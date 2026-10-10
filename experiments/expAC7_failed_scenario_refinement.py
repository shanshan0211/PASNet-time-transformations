"""Run the sealed post-hoc refinement of all AC6 learned-g failures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
PROTOCOL_PATH = os.path.join(RESULTS, "expAC7", "preregistration.json")
RAW_PATH = os.path.join(RESULTS, "expAC7", "raw_results.json")
SUMMARY_PATH = os.path.join(RESULTS, "expAC7", "summary.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from expAC6_monitor_orbit_generalisation import (  # noqa: E402
    atomic_json,
    load_model,
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


def verify_protocol():
    with open(PROTOCOL_PATH, encoding="utf-8") as handle:
        protocol = json.load(handle)
    if canonical_protocol_hash(protocol) != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("AC7 protocol hash mismatch")
    for source in [protocol["frozen_checkpoint"], *protocol["frozen_sources"]]:
        if sha256_file(os.path.join(ROOT, source["path"])) != source["sha256"]:
            raise RuntimeError(f"frozen source changed: {source['path']}")
    return protocol


def finite_ratio(numerator, denominator):
    if (
        numerator > 0.0
        and denominator > 0.0
        and math.isfinite(numerator)
        and math.isfinite(denominator)
    ):
        return numerator / denominator
    return None


def geometric_mean(values):
    values = [
        value
        for value in values
        if value is not None and value > 0.0 and math.isfinite(value)
    ]
    return float(math.exp(np.mean(np.log(values)))) if values else None


def summarize(records, protocol):
    counts = {}
    for record in records:
        key = (record["method"], str(record["dtau"]))
        counts.setdefault(key, {})
        status = record["completion_status"]
        counts[key][status] = counts[key].get(status, 0) + 1

    index = {
        (
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
            record["method"],
            record["dtau"],
        ): record
        for record in records
    }
    learned_manual = {
        str(dtau): {"energy": [], "closure": [], "steps": [], "wall": []}
        for dtau in protocol["design"]["dtau_values"]
    }
    refinement = {
        method: {"energy_orders": [], "closure_orders": []}
        for method in protocol["design"]["methods"]
    }
    for scenario in protocol["failed_scenarios"]:
        scenario_key = (
            scenario["eccentricity"],
            scenario["orientation_radians"],
            scenario["momentum_direction"],
        )
        for dtau in protocol["design"]["dtau_values"]:
            manual = index[(*scenario_key, "manual_r", dtau)]
            learned = index[(*scenario_key, "learned_g", dtau)]
            if (
                manual["completion_status"] == "complete"
                and learned["completion_status"] == "complete"
            ):
                target = learned_manual[str(dtau)]
                target["energy"].append(
                    finite_ratio(
                        learned["max_energy_drift"],
                        manual["max_energy_drift"],
                    )
                )
                target["closure"].append(
                    finite_ratio(
                        learned["final_state_closure_error"],
                        manual["final_state_closure_error"],
                    )
                )
                target["steps"].append(
                    finite_ratio(learned["steps"], manual["steps"])
                )
                target["wall"].append(
                    finite_ratio(
                        learned["wall_seconds"], manual["wall_seconds"]
                    )
                )
        for method in protocol["design"]["methods"]:
            coarse = index[(*scenario_key, method, 0.02)]
            fine = index[(*scenario_key, method, 0.01)]
            if (
                coarse["completion_status"] == "complete"
                and fine["completion_status"] == "complete"
            ):
                energy_ratio = finite_ratio(
                    coarse["max_energy_drift"], fine["max_energy_drift"]
                )
                closure_ratio = finite_ratio(
                    coarse["final_state_closure_error"],
                    fine["final_state_closure_error"],
                )
                refinement[method]["energy_orders"].append(
                    math.log(energy_ratio, 2.0)
                    if energy_ratio is not None
                    else None
                )
                refinement[method]["closure_orders"].append(
                    math.log(closure_ratio, 2.0)
                    if closure_ratio is not None
                    else None
                )

    completed = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    acceptance = {
        "all_60_attempted_and_retained": len(records)
        == protocol["design"]["total_registered_rollouts"],
        "completed_final_times_aligned": all(
            record["final_time_error"]
            < protocol["acceptance"]["completed_final_time_error_below"]
            for record in completed
        ),
        "completed_solver_residuals_valid": all(
            record["max_fp_residual"]
            < protocol["acceptance"]["completed_solver_residual_below"]
            for record in completed
        ),
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "counts_by_method_dtau": [
            {"method": key[0], "dtau": float(key[1]), **statuses}
            for key, statuses in sorted(counts.items())
        ],
        "paired_learned_over_manual": {
            dtau: {
                metric: {
                    "geometric_mean_ratio": geometric_mean(values),
                    "n_finite_pairs": sum(
                        value is not None for value in values
                    ),
                }
                for metric, values in metrics.items()
            }
            for dtau, metrics in learned_manual.items()
        },
        "refinement": {
            method: {
                metric: {
                    "mean_observed_order": (
                        float(np.mean([v for v in values if v is not None]))
                        if any(v is not None for v in values)
                        else None
                    ),
                    "n_finite_pairs": sum(v is not None for v in values),
                }
                for metric, values in metrics.items()
            }
            for method, metrics in refinement.items()
        },
        "acceptance": acceptance,
        "claim_limits": {
            "post_hoc_failed_scenario_refinement": True,
            "does_not_replace_ac6_primary_estimands": True,
            "true_hamiltonian_only": True,
            "one_frozen_checkpoint": True,
        },
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    model = load_model(protocol)
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "records": [],
    }
    base_design = protocol["design"]
    for scenario in protocol["failed_scenarios"]:
        eccentricity = scenario["eccentricity"]
        angle = scenario["orientation_radians"]
        direction = scenario["momentum_direction"]
        system = Kepler2D(eccentricity)
        base_q, base_p = system.initial_condition()
        q0, p0 = rotate_state(base_q, base_p, angle)
        if direction == "reversed":
            p0 = -p0
        for dtau in base_design["dtau_values"]:
            design = {
                **base_design,
                "dtau": dtau,
            }
            for method in base_design["methods"]:
                print(
                    f"e={eccentricity:.2f} angle={angle:.3f} {direction} "
                    f"dtau={dtau:g} {method}",
                    flush=True,
                )
                result = run_rollout(
                    system, model, q0, p0, method, design
                )
                result.update(
                    eccentricity=eccentricity,
                    orientation_radians=angle,
                    momentum_direction=direction,
                    dtau=dtau,
                    ac6_status=scenario["ac6_status"],
                )
                artifact["records"].append(result)
                atomic_json(RAW_PATH, artifact)
                print(
                    f"  {result['completion_status']} "
                    f"dE={result['max_energy_drift']:.3e} "
                    f"closure={result['final_state_closure_error']:.3e}",
                    flush=True,
                )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    summary = summarize(artifact["records"], protocol)
    summary["raw_results_sha256"] = sha256_file(RAW_PATH)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("Exp AC7 artifact gate failed")


if __name__ == "__main__":
    main()
