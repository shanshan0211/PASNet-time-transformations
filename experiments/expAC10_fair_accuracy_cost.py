"""Run the sealed AC10 exact-H accuracy--cost comparison.

Each method configuration runs in a fresh subprocess. This makes maximum RSS
comparable at configuration level while amortising model loading over the same
18 registered scenarios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC10")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from expAC2_corrected_long import run_adaptive, run_direct  # noqa: E402
from expAC9_end_to_end_monitor_generalisation import (  # noqa: E402
    Monitor,
    kepler_state,
    rotate_state,
    run_rollout,
)
from systems import Kepler2D  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_protocol_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def verify_protocol():
    with open(PROTOCOL_PATH, encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected = protocol["integrity"]["protocol_sha256"]
    actual = canonical_protocol_hash(protocol)
    if actual != expected:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {expected}")
    for source in protocol["frozen_sources"]:
        actual_source = sha256_file(os.path.join(ROOT, source["path"]))
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


def scenarios(protocol):
    design = protocol["evaluation"]
    result = []
    for semi_major_axis in design["semi_major_axes"]:
        for eccentricity in design["eccentricities"]:
            for anomaly in design["initial_mean_anomalies"]:
                result.append(
                    {
                        "semi_major_axis": semi_major_axis,
                        "eccentricity": eccentricity,
                        "initial_mean_anomaly": anomaly,
                        "orientation_radians": 0.0,
                        "momentum_direction": "forward",
                        "energy_shell": -0.5 / semi_major_axis,
                        "same_training_energy": semi_major_axis == 1.0,
                    }
                )
    return result


def method_configurations(protocol):
    methods = protocol["methods"]
    configurations = []
    for step in methods["direct_implicit_midpoint"]["dt"]:
        configurations.append(
            {
                "configuration_id": f"direct_dt_{step}",
                "method": "direct_implicit_midpoint",
                "dt": step,
            }
        )
    for tolerance in methods["adaptive_implicit_midpoint"]["rtol"]:
        configurations.append(
            {
                "configuration_id": f"adaptive_rtol_{tolerance}",
                "method": "adaptive_implicit_midpoint",
                "rtol": tolerance,
            }
        )
    for step in methods["manual_sundman"]["dtau"]:
        configurations.append(
            {
                "configuration_id": f"manual_r_dtau_{step}",
                "method": "manual_sundman",
                "dtau": step,
            }
        )
    for architecture in ["raw", "invariant"]:
        for seed in methods["learned_monitors"]["checkpoint_seeds"]:
            for step in methods["learned_monitors"]["dtau"]:
                configurations.append(
                    {
                        "configuration_id":
                            f"{architecture}_seed_{seed}_dtau_{step}",
                        "method": "learned_monitor",
                        "architecture": architecture,
                        "checkpoint_seed": seed,
                        "dtau": step,
                    }
                )
    return configurations


def initial_condition(scenario):
    state = kepler_state(
        scenario["semi_major_axis"],
        scenario["eccentricity"],
        scenario["initial_mean_anomaly"],
    )
    state = rotate_state(state, scenario["orientation_radians"])
    if scenario["momentum_direction"] == "reversed":
        state[2:] *= -1.0
    return state[:2], state[2:]


def normalize_physical_record(record, configuration, scenario):
    return {
        "configuration_id": configuration["configuration_id"],
        "method": configuration["method"],
        "architecture": None,
        "checkpoint_seed": None,
        "completion_status": "complete",
        "steps": record["steps"],
        "gradient_evaluations": record["gradient_evaluations"],
        "nonlinear_iterations": record["gradient_evaluations"]
        - record["steps"],
        "rejected_steps": record["rejected_steps"],
        "maximum_residual": record["max_fp_residual"],
        "maximum_iterations": record["max_fp_iterations"],
        "max_energy_error": record["max_energy_drift"],
        "max_angular_momentum_error":
            record["max_angular_momentum_error"],
        "final_state_closure_error": record["final_state_error"],
        "physical_time": record["physical_time"],
        "final_time_error": record["final_time_error"],
        "wall_seconds": record["wall_seconds"],
        **scenario,
    }


def run_configuration(protocol, configuration):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    design = protocol["evaluation"]
    evaluation = {
        "periods": design["periods"],
        "dtau": configuration.get("dtau"),
        "fixed_point": design["fixed_point"],
        "maximum_steps": design["maximum_steps"],
    }
    monitor = None
    if configuration["method"] == "learned_monitor":
        monitor = Monitor(configuration["architecture"])
        checkpoint = os.path.join(
            ROOT,
            "results",
            "expAC9",
            f"monitor_{configuration['architecture']}_"
            f"seed{configuration['checkpoint_seed']}.pt",
        )
        package = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        monitor.load_state_dict(package["state_dict"])
        monitor.eval()
    elif configuration["method"] == "manual_sundman":
        monitor = lambda value: torch.linalg.vector_norm(  # noqa: E731
            value[:, :2], dim=1
        )

    records = []
    for scenario in scenarios(protocol):
        try:
            system = Kepler2D(scenario["eccentricity"])
            q0, p0 = initial_condition(scenario)
            horizon = (
                2.0
                * math.pi
                * scenario["semi_major_axis"] ** 1.5
                * design["periods"]
            )
            if configuration["method"] == "direct_implicit_midpoint":
                source = run_direct(
                    system, q0, p0, configuration["dt"], horizon
                )
                record = normalize_physical_record(
                    source, configuration, scenario
                )
            elif configuration["method"] == "adaptive_implicit_midpoint":
                source = run_adaptive(
                    system, q0, p0, configuration["rtol"], horizon
                )
                record = normalize_physical_record(
                    source, configuration, scenario
                )
            else:
                architecture = configuration.get(
                    "architecture", "manual_r"
                )
                seed = configuration.get("checkpoint_seed")
                record = run_rollout(
                    monitor,
                    architecture,
                    seed,
                    scenario,
                    evaluation,
                )
                record["configuration_id"] = configuration[
                    "configuration_id"
                ]
                record["method"] = configuration["method"]
                record["rejected_steps"] = 0
                record["gradient_evaluations"] = record[
                    "vector_field_evaluations"
                ]
        except Exception as error:
            record = {
                "configuration_id": configuration["configuration_id"],
                "method": configuration["method"],
                "architecture": configuration.get("architecture"),
                "checkpoint_seed": configuration.get("checkpoint_seed"),
                "completion_status": "unexpected_error",
                "error": str(error),
                "traceback": traceback.format_exc(),
                "steps": 0,
                "gradient_evaluations": 0,
                "rejected_steps": 0,
                "maximum_residual": 0.0,
                "max_energy_error": 0.0,
                "max_angular_momentum_error": 0.0,
                "final_state_closure_error": 0.0,
                "physical_time": 0.0,
                "final_time_error": None,
                "wall_seconds": 0.0,
                **scenario,
            }
        records.append(record)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for record in records:
        record["configuration_peak_rss_kib"] = peak_rss
        record["memory_scope"] = (
            "fresh subprocess peak across the configuration's 18 scenarios"
        )
    return {
        "configuration": configuration,
        "records": records,
        "configuration_peak_rss_kib": peak_rss,
    }


def summarize(protocol, artifact):
    records = artifact["records"]
    statuses = {}
    for record in records:
        status = record["completion_status"]
        statuses[status] = statuses.get(status, 0) + 1
    complete = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    expected = protocol["evaluation"]["total_registered_rollouts"]
    acceptance = {
        "all_configurations_attempted": len(
            artifact["completed_configuration_ids"]
        )
        == protocol["evaluation"]["total_configurations"],
        "all_rollouts_attempted_and_retained": len(records) == expected,
        "completed_final_times_aligned": all(
            record["final_time_error"] < 1e-10 for record in complete
        ),
        "completed_residuals_valid": all(
            record["maximum_residual"] < 1e-11
            for record in complete
        ),
        "configuration_peak_memory_recorded": all(
            record["configuration_peak_rss_kib"] > 0
            for record in records
        ),
        "source_hashes_valid": True,
        "all_failures_reported": sum(statuses.values()) == len(records),
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "counts": {
            "registered": expected,
            "attempted": len(records),
            "by_status": statuses,
        },
        "acceptance": acceptance,
        "raw_results_sha256": sha256_file(RAW_PATH),
        "memory_definition":
            "ru_maxrss from one fresh subprocess per method configuration; "
            "each subprocess evaluates the same 18 scenarios",
    }


def orchestrate(protocol):
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "records": [],
        "configuration_memory": [],
        "completed_configuration_ids": [],
    }
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if (
            artifact["protocol_sha256"]
            != protocol["integrity"]["protocol_sha256"]
        ):
            raise RuntimeError("existing AC10 protocol mismatch")
    completed = set(artifact["completed_configuration_ids"])
    for configuration in method_configurations(protocol):
        identifier = configuration["configuration_id"]
        if identifier in completed:
            continue
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker-config-json",
            json.dumps(configuration, separators=(",", ":")),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"worker failed for {identifier}: {result.stderr}"
            )
        package = json.loads(result.stdout)
        artifact["records"].extend(package["records"])
        artifact["configuration_memory"].append(
            {
                "configuration_id": identifier,
                "peak_rss_kib": package["configuration_peak_rss_kib"],
            }
        )
        artifact["completed_configuration_ids"].append(identifier)
        atomic_json(RAW_PATH, artifact)
        complete = sum(
            record["completion_status"] == "complete"
            for record in package["records"]
        )
        print(
            f"{identifier}: {complete}/{len(package['records'])} complete, "
            f"peak={package['configuration_peak_rss_kib']} KiB",
            flush=True,
        )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    summary = summarize(protocol, artifact)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("Exp AC10 artifact gate failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config-json")
    arguments = parser.parse_args()
    protocol = verify_protocol()
    if arguments.worker_config_json:
        configuration = json.loads(arguments.worker_config_json)
        print(
            json.dumps(
                run_configuration(protocol, configuration),
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        return
    orchestrate(protocol)


if __name__ == "__main__":
    main()
