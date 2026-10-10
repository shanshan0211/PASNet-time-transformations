"""Replay the sealed AC6--AC10 endpoint-residual correction set.

AC15 derives its manifest from immutable parent raw artifacts, accepts an
implicit solve only after recomputing the defining residual at the returned
endpoint, and atomically persists every attempted replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC15")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SMOKE_PATH = os.path.join(OUT, "smoke_results.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from expAC8_monitor_architecture_solver_factorial import (  # noqa: E402
    StudentMonitor,
)
from expAC9_end_to_end_monitor_generalisation import (  # noqa: E402
    Monitor,
    kepler_state,
    rotate_state as rotate_flat_state,
)
from implicit_midpoint_solvers import (  # noqa: E402
    ImplicitMidpointSolverError,
    fixed_point_endpoint,
)
from models.pasnet import PASNet  # noqa: E402
from systems import Kepler2D  # noqa: E402


PARENT_COUNTS = {"AC6": 80, "AC7": 60, "AC8": 132, "AC9": 792, "AC10": 684}
TOTAL_REPLAYS = sum(PARENT_COUNTS.values())


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
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_protocol():
    protocol = load_json(PROTOCOL_PATH)
    expected = protocol["integrity"]["protocol_sha256"]
    actual = canonical_protocol_hash(protocol)
    if actual != expected:
        raise RuntimeError(f"AC15 protocol hash mismatch: {actual} != {expected}")
    for source in protocol["frozen_sources"]:
        actual_source = sha256_file(os.path.join(ROOT, source["path"]))
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


def replay_manifest():
    manifest = []
    for number in range(6, 11):
        parent = f"AC{number}"
        raw_path = os.path.join(
            ROOT, "results", f"expAC{number}", "raw_results.json"
        )
        records = load_json(raw_path)["records"]
        for parent_index, record in enumerate(records):
            if number == 8 and record["solver"] != "fixed_point":
                continue
            manifest.append(
                {
                    "replay_id": f"{parent}:{parent_index:04d}",
                    "parent": parent,
                    "parent_index": parent_index,
                    "parent_record": record,
                }
            )
    counts = {
        parent: sum(item["parent"] == parent for item in manifest)
        for parent in PARENT_COUNTS
    }
    if counts != PARENT_COUNTS or len(manifest) != TOTAL_REPLAYS:
        raise RuntimeError(f"invalid replay manifest: {counts}")
    return manifest


def _rotate_pair(q, p, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return rotation @ q, rotation @ p


def _h_features(z):
    radius = torch.linalg.vector_norm(z[:, :2], dim=1, keepdim=True)
    return torch.cat([radius, 1.0 / radius], dim=1)


class ModelCache:
    def __init__(self):
        self.models = {}

    def ac6(self):
        key = ("AC6",)
        if key not in self.models:
            model = PASNet(
                dim=2,
                hidden_H=128,
                n_hidden_H=3,
                hidden_g=64,
                n_hidden_g=2,
                g_max=2.0,
                h_features=_h_features,
            )
            package = torch.load(
                os.path.join(ROOT, "results", "expE2", "pasnet_weights.pt"),
                map_location="cpu",
                weights_only=True,
            )
            model.H_net.load_state_dict(package["H_net"])
            model.g_net.load_state_dict(package["g_net"])
            self.models[key] = model.double().eval()
        return self.models[key]

    def ac8(self, architecture, seed):
        if architecture == "manual_r":
            return None
        key = ("AC8", architecture, seed)
        if key not in self.models:
            model = StudentMonitor(architecture)
            path = os.path.join(
                ROOT,
                "results",
                "expAC8",
                f"monitor_{architecture}_seed{seed}.pt",
            )
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            self.models[key] = model.double().eval()
        return self.models[key]

    def ac9(self, architecture, seed):
        if architecture == "manual_r":
            return None
        key = ("AC9", architecture, seed)
        if key not in self.models:
            model = Monitor(architecture)
            path = os.path.join(
                ROOT,
                "results",
                "expAC9",
                f"monitor_{architecture}_seed{seed}.pt",
            )
            package = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(package["state_dict"])
            self.models[key] = model.double().eval()
        return self.models[key]


def _monitor_function(model, manual):
    if manual:
        return lambda value: torch.linalg.vector_norm(  # noqa: E731
            value[:, :2], dim=1
        )
    return model


def transformed_case(spec, cache):
    parent = spec["parent"]
    source = spec["parent_record"]
    eccentricity = source["eccentricity"]
    system = Kepler2D(eccentricity)
    if parent in {"AC6", "AC7", "AC8"}:
        q0, p0 = system.initial_condition()
        q0, p0 = _rotate_pair(q0, p0, source["orientation_radians"])
        if source["momentum_direction"] == "reversed":
            p0 = -p0
        initial = np.concatenate([q0, p0])
    else:
        initial = kepler_state(
            source["semi_major_axis"],
            eccentricity,
            source["initial_mean_anomaly"],
        )
        initial = rotate_flat_state(initial, source["orientation_radians"])
        if source["momentum_direction"] == "reversed":
            initial[2:] *= -1.0
        q0, p0 = initial[:2], initial[2:]

    if parent in {"AC6", "AC7"}:
        architecture = source["method"]
        model = cache.ac6()
        monitor = _monitor_function(model.g, architecture == "manual_r")
        step_size = source.get("dtau", 0.04)
        horizon = 62.83185307179586
        alignment_iterations = 45
        step_cap = 500000 if parent == "AC7" else 250000
    elif parent == "AC8":
        architecture = source["architecture"]
        model = cache.ac8(architecture, source["seed"])
        monitor = _monitor_function(model, architecture == "manual_r")
        step_size = 0.04
        horizon = 18.84955592153876
        alignment_iterations = 40
        step_cap = 100000
    else:
        architecture = source["architecture"]
        seed = source["checkpoint_seed"]
        model = cache.ac9(architecture, seed)
        monitor = _monitor_function(model, architecture == "manual_r")
        step_size = (
            float(source["configuration_id"].rsplit("_dtau_", 1)[1])
            if parent == "AC10"
            else 0.04
        )
        horizon = (
            2.0
            * math.pi
            * source["semi_major_axis"] ** 1.5
        )
        alignment_iterations = 40
        step_cap = 100000
    return {
        "system": system,
        "initial": initial,
        "monitor": monitor,
        "step_size": step_size,
        "horizon": horizon,
        "alignment_iterations": alignment_iterations,
        "step_cap": step_cap,
        "architecture": architecture,
    }


def angular_momentum(state):
    return float(state[0] * state[3] - state[1] * state[2])


def _failure_payload(error):
    payload = {
        "error": str(error),
        "failure_type": type(error).__name__,
    }
    if isinstance(error, ImplicitMidpointSolverError):
        payload["failure_solver_diagnostics"] = error.diagnostics.to_dict()
        payload["failure_endpoint"] = (
            error.endpoint.detach().cpu().reshape(-1).tolist()
        )
    return payload


def run_transformed(spec, cache, smoke_steps=None):
    case = transformed_case(spec, cache)
    source = spec["parent_record"]
    system = case["system"]
    initial = case["initial"]
    state = torch.tensor(initial[None, :], dtype=torch.float64)
    energy0 = float(system.H(initial[:2], initial[2:]))
    momentum0 = angular_momentum(initial)
    p0_extended = torch.tensor([-energy0], dtype=torch.float64)
    monitor = case["monitor"]
    vector_evaluations = 0

    def vector_field(value):
        nonlocal vector_evaluations
        vector_evaluations += 1
        with torch.enable_grad():
            differentiable = value.detach().requires_grad_(True)
            extended = monitor(differentiable) * (
                system.H_torch(differentiable) + p0_extended
            )
            gradient = torch.autograd.grad(
                extended.sum(), differentiable
            )[0]
        return torch.cat(
            [gradient[:, 2:], -gradient[:, :2]], dim=1
        ).detach()

    record = {
        "completion_status": "running",
        "steps": 0,
        "rejected_steps": 0,
        "nonlinear_iterations": 0,
        "maximum_returned_endpoint_residual_inf": 0.0,
        "maximum_stopping_update_inf": 0.0,
        "maximum_nonlinear_iterations": 0,
        "max_energy_error": 0.0,
        "max_angular_momentum_error": 0.0,
        "physical_time": 0.0,
        "solver_tolerance": 1e-12,
        "solver_maximum_iterations": 100,
    }
    started = time.perf_counter()
    target = case["horizon"]
    try:
        while record["physical_time"] < target:
            if smoke_steps is not None and record["steps"] >= smoke_steps:
                record["completion_status"] = "smoke_step_limit"
                break
            if record["steps"] >= case["step_cap"]:
                record["completion_status"] = "step_cap"
                break
            step = case["step_size"]
            trial, diagnostics = fixed_point_endpoint(
                state,
                step,
                vector_field,
                tolerance=record["solver_tolerance"],
                maximum_iterations=record["solver_maximum_iterations"],
            )
            physical_dt = step * float(
                monitor(0.5 * (state + trial)).detach()
            )
            if not math.isfinite(physical_dt) or physical_dt <= 0.0:
                raise FloatingPointError(
                    f"invalid physical increment {physical_dt}"
                )
            if record["physical_time"] + physical_dt > target:
                remaining = target - record["physical_time"]
                lower, upper = 0.0, step
                for _ in range(case["alignment_iterations"]):
                    candidate_step = 0.5 * (lower + upper)
                    candidate, candidate_diagnostics = fixed_point_endpoint(
                        state,
                        candidate_step,
                        vector_field,
                        tolerance=record["solver_tolerance"],
                        maximum_iterations=record[
                            "solver_maximum_iterations"
                        ],
                    )
                    candidate_dt = candidate_step * float(
                        monitor(0.5 * (state + candidate)).detach()
                    )
                    if candidate_dt < remaining:
                        lower = candidate_step
                    else:
                        upper = candidate_step
                    trial = candidate
                    diagnostics = candidate_diagnostics
                physical_dt = remaining
            state = trial
            record["physical_time"] += physical_dt
            record["steps"] += 1
            record["nonlinear_iterations"] += diagnostics.iterations
            residual = diagnostics.returned_endpoint_residual_inf
            record["maximum_returned_endpoint_residual_inf"] = max(
                record["maximum_returned_endpoint_residual_inf"], residual
            )
            record["maximum_stopping_update_inf"] = max(
                record["maximum_stopping_update_inf"],
                diagnostics.stopping_update_inf or 0.0,
            )
            record["maximum_nonlinear_iterations"] = max(
                record["maximum_nonlinear_iterations"],
                diagnostics.iterations,
            )
            current = state.detach().cpu().numpy()[0]
            record["max_energy_error"] = max(
                record["max_energy_error"],
                abs(
                    float(system.H(current[:2], current[2:])) - energy0
                ),
            )
            record["max_angular_momentum_error"] = max(
                record["max_angular_momentum_error"],
                abs(angular_momentum(current) - momentum0),
            )
        if record["completion_status"] == "running":
            record["completion_status"] = "complete"
    except ImplicitMidpointSolverError as error:
        record["completion_status"] = "solver_failure"
        record.update(_failure_payload(error))
    except FloatingPointError as error:
        record["completion_status"] = "nonfinite"
        record.update(_failure_payload(error))
    except Exception as error:
        record["completion_status"] = "unexpected_error"
        record.update(_failure_payload(error))
        record["traceback"] = traceback.format_exc()
    final = state.detach().cpu().numpy()[0]
    record.update(
        vector_field_evaluations=vector_evaluations,
        final_state_closure_error=float(np.linalg.norm(final - initial)),
        final_time_error=abs(record["physical_time"] - target),
        final_state=final.tolist(),
        wall_seconds=time.perf_counter() - started,
        method=source.get("method"),
        architecture=source.get("architecture", source.get("method")),
        checkpoint_seed=source.get("checkpoint_seed", source.get("seed")),
        configuration_id=source.get("configuration_id"),
    )
    return record


def direct_vector_field(state):
    q, p = state[:, :2], state[:, 2:]
    radius = torch.linalg.vector_norm(q, dim=1, keepdim=True)
    return torch.cat([p, -q / radius**3], dim=1)


def _physical_solve(state, step):
    return fixed_point_endpoint(
        state,
        step,
        direct_vector_field,
        tolerance=1e-13,
        maximum_iterations=100,
    )


def _physical_base(spec, initial):
    source = spec["parent_record"]
    system = Kepler2D(source["eccentricity"])
    energy0 = float(system.H(initial[:2], initial[2:]))
    return source, system, energy0, angular_momentum(initial)


def run_direct(spec, smoke_steps=None):
    source = spec["parent_record"]
    initial = kepler_state(
        source["semi_major_axis"],
        source["eccentricity"],
        source["initial_mean_anomaly"],
    )
    initial = rotate_flat_state(initial, source["orientation_radians"])
    state = torch.tensor(initial[None, :], dtype=torch.float64)
    source, system, energy0, momentum0 = _physical_base(spec, initial)
    horizon = 2.0 * math.pi * source["semi_major_axis"] ** 1.5
    step = float(source["configuration_id"].rsplit("_dt_", 1)[1])
    record = _new_physical_record(source)
    started = time.perf_counter()
    try:
        while record["physical_time"] < horizon:
            if smoke_steps is not None and record["steps"] >= smoke_steps:
                record["completion_status"] = "smoke_step_limit"
                break
            current_step = min(step, horizon - record["physical_time"])
            endpoint, diagnostics = _physical_solve(state, current_step)
            state = endpoint
            record["physical_time"] += current_step
            _accept_physical(
                record, diagnostics, state, system, energy0, momentum0
            )
        if record["completion_status"] == "running":
            record["completion_status"] = "complete"
    except ImplicitMidpointSolverError as error:
        record["completion_status"] = "solver_failure"
        record.update(_failure_payload(error))
    except Exception as error:
        record["completion_status"] = "unexpected_error"
        record.update(_failure_payload(error))
        record["traceback"] = traceback.format_exc()
    return _finish_physical(record, state, initial, horizon, started)


def _new_physical_record(source):
    return {
        "completion_status": "running",
        "method": source["method"],
        "configuration_id": source["configuration_id"],
        "architecture": source.get("architecture"),
        "checkpoint_seed": source.get("checkpoint_seed"),
        "steps": 0,
        "rejected_steps": 0,
        "nonlinear_iterations": 0,
        "vector_field_evaluations": 0,
        "maximum_returned_endpoint_residual_inf": 0.0,
        "maximum_stopping_update_inf": 0.0,
        "maximum_nonlinear_iterations": 0,
        "max_energy_error": 0.0,
        "max_angular_momentum_error": 0.0,
        "physical_time": 0.0,
        "solver_tolerance": 1e-13,
        "solver_maximum_iterations": 100,
    }


def _accept_physical(
    record, diagnostics, state, system, energy0, momentum0, count_step=True
):
    if count_step:
        record["steps"] += 1
    record["nonlinear_iterations"] += diagnostics.iterations
    record["vector_field_evaluations"] += (
        diagnostics.vector_field_evaluations
    )
    record["maximum_returned_endpoint_residual_inf"] = max(
        record["maximum_returned_endpoint_residual_inf"],
        diagnostics.returned_endpoint_residual_inf,
    )
    record["maximum_stopping_update_inf"] = max(
        record["maximum_stopping_update_inf"],
        diagnostics.stopping_update_inf or 0.0,
    )
    record["maximum_nonlinear_iterations"] = max(
        record["maximum_nonlinear_iterations"], diagnostics.iterations
    )
    current = state.detach().cpu().numpy()[0]
    record["max_energy_error"] = max(
        record["max_energy_error"],
        abs(float(system.H(current[:2], current[2:])) - energy0),
    )
    record["max_angular_momentum_error"] = max(
        record["max_angular_momentum_error"],
        abs(angular_momentum(current) - momentum0),
    )


def _finish_physical(record, state, initial, horizon, started):
    final = state.detach().cpu().numpy()[0]
    record.update(
        final_state_closure_error=float(np.linalg.norm(final - initial)),
        final_time_error=abs(record["physical_time"] - horizon),
        final_state=final.tolist(),
        wall_seconds=time.perf_counter() - started,
    )
    return record


def run_adaptive(spec, smoke_steps=None):
    source = spec["parent_record"]
    initial = kepler_state(
        source["semi_major_axis"],
        source["eccentricity"],
        source["initial_mean_anomaly"],
    )
    initial = rotate_flat_state(initial, source["orientation_radians"])
    state = torch.tensor(initial[None, :], dtype=torch.float64)
    source, system, energy0, momentum0 = _physical_base(spec, initial)
    horizon = 2.0 * math.pi * source["semi_major_axis"] ** 1.5
    rtol = float(source["configuration_id"].rsplit("_rtol_", 1)[1])
    atol = 1e-12
    step = 0.005
    record = _new_physical_record(source)
    started = time.perf_counter()
    try:
        while record["physical_time"] < horizon:
            if smoke_steps is not None and record["steps"] >= smoke_steps:
                record["completion_status"] = "smoke_step_limit"
                break
            step = min(step, horizon - record["physical_time"])
            full, full_diagnostics = _physical_solve(state, step)
            half, first_diagnostics = _physical_solve(state, step / 2.0)
            doubled, second_diagnostics = _physical_solve(half, step / 2.0)
            for diagnostics in (
                full_diagnostics,
                first_diagnostics,
                second_diagnostics,
            ):
                _accept_physical(
                    record,
                    diagnostics,
                    state,
                    system,
                    energy0,
                    momentum0,
                    count_step=False,
                )
            scale = atol + rtol * torch.maximum(
                torch.abs(state), torch.abs(doubled)
            )
            difference = (full - doubled) / 3.0
            error = float(torch.sqrt(torch.mean((difference / scale) ** 2)))
            factor = min(
                5.0,
                max(0.2, 0.9 * max(error, 1e-16) ** (-1.0 / 3.0)),
            )
            if error <= 1.0:
                state = doubled
                record["physical_time"] += step
                record["steps"] += 1
                current = state.detach().cpu().numpy()[0]
                record["max_energy_error"] = max(
                    record["max_energy_error"],
                    abs(
                        float(system.H(current[:2], current[2:])) - energy0
                    ),
                )
                record["max_angular_momentum_error"] = max(
                    record["max_angular_momentum_error"],
                    abs(angular_momentum(current) - momentum0),
                )
            else:
                record["rejected_steps"] += 1
            step = min(0.05, max(1e-7, step * factor))
        if record["completion_status"] == "running":
            record["completion_status"] = "complete"
    except ImplicitMidpointSolverError as error:
        record["completion_status"] = "solver_failure"
        record.update(_failure_payload(error))
    except Exception as error:
        record["completion_status"] = "unexpected_error"
        record.update(_failure_payload(error))
        record["traceback"] = traceback.format_exc()
    return _finish_physical(record, state, initial, horizon, started)


def run_replay(spec, cache, smoke_steps=None):
    source = spec["parent_record"]
    started = datetime.now(timezone.utc).isoformat()
    if (
        spec["parent"] == "AC10"
        and source["method"] == "direct_implicit_midpoint"
    ):
        result = run_direct(spec, smoke_steps)
    elif (
        spec["parent"] == "AC10"
        and source["method"] == "adaptive_implicit_midpoint"
    ):
        result = run_adaptive(spec, smoke_steps)
    else:
        result = run_transformed(spec, cache, smoke_steps)
    result.update(
        replay_id=spec["replay_id"],
        parent_experiment=spec["parent"],
        parent_record_index=spec["parent_index"],
        parent_completion_status=source["completion_status"],
        parent_maximum_reported_residual=source.get(
            "maximum_residual",
            source.get(
                "maximum_nonlinear_residual",
                source.get("max_fp_residual"),
            ),
        ),
        parent_steps=source.get("steps"),
        parent_max_energy_error=source.get(
            "max_energy_error", source.get("max_energy_drift")
        ),
        parent_final_state_closure_error=source.get(
            "final_state_closure_error", source.get("final_state_error")
        ),
        started_at_utc=started,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        smoke_only=smoke_steps is not None,
    )
    return result


def new_artifact(protocol, smoke):
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_only": smoke,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "records": [],
    }


def execute(protocol, selected, output_path, smoke_steps=None):
    smoke = smoke_steps is not None
    artifact = new_artifact(protocol, smoke)
    if os.path.exists(output_path) and not smoke:
        artifact = load_json(output_path)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError("existing AC15 raw artifact protocol mismatch")
        if artifact.get("smoke_only"):
            raise RuntimeError("refusing to resume full run from smoke artifact")
    completed = {record["replay_id"] for record in artifact["records"]}
    cache = ModelCache()
    for spec in selected:
        if spec["replay_id"] in completed:
            continue
        record = run_replay(spec, cache, smoke_steps)
        artifact["records"].append(record)
        atomic_json(output_path, artifact)
        print(
            f"{record['replay_id']} "
            f"{record['parent_completion_status']} -> "
            f"{record['completion_status']} "
            f"steps={record['steps']} "
            f"R={record['maximum_returned_endpoint_residual_inf']:.3e}",
            flush=True,
        )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, artifact)
    return artifact


def smoke_selection(manifest):
    selected = []
    for parent in PARENT_COUNTS:
        selected.append(next(item for item in manifest if item["parent"] == parent))
    direct = next(
        item
        for item in manifest
        if item["parent"] == "AC10"
        and item["parent_record"]["method"] == "direct_implicit_midpoint"
    )
    adaptive = next(
        item
        for item in manifest
        if item["parent"] == "AC10"
        and item["parent_record"]["method"] == "adaptive_implicit_midpoint"
    )
    for item in (direct, adaptive):
        if item["replay_id"] not in {entry["replay_id"] for entry in selected}:
            selected.append(item)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--parent", choices=list(PARENT_COUNTS))
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    manifest = replay_manifest()
    if arguments.smoke:
        selected = smoke_selection(manifest)
        execute(protocol, selected, SMOKE_PATH, arguments.smoke_steps)
        return
    selected = [
        item
        for item in manifest
        if arguments.parent is None or item["parent"] == arguments.parent
    ]
    if arguments.limit is not None:
        selected = selected[: arguments.limit]
    execute(protocol, selected, RAW_PATH)


if __name__ == "__main__":
    main()
