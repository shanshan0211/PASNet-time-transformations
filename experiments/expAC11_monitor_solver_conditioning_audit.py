"""Read-only AC11 replay of frozen AC8/AC9 monitor evaluations.

This runner intentionally reproduces the parent stopping semantics to replay
their completion labels.  It separately evaluates the true endpoint residual
at every recorded endpoint.  It never trains or writes parent artifacts.
"""

from __future__ import annotations

import hashlib
import importlib
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
OUT = os.path.join(ROOT, "results", "expAC11")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from implicit_midpoint_solvers import endpoint_residual  # noqa: E402
from monitor_conditioning import conditioning_diagnostics  # noqa: E402
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


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def verify_protocol():
    protocol = load_json(PROTOCOL_PATH)
    actual = canonical_protocol_hash(protocol)
    expected = protocol["integrity"]["protocol_sha256"]
    if actual != expected:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {expected}")
    for source in protocol["frozen_sources"]:
        path = os.path.join(ROOT, source["path"])
        if sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"sealed source changed: {source['path']}")
    return protocol


class ParentSolverFailure(RuntimeError):
    def __init__(self, reason, message, endpoint, diagnostics):
        super().__init__(message)
        self.reason = reason
        self.endpoint = endpoint
        self.diagnostics = diagnostics


def _inf_norm(value):
    return float(torch.max(torch.abs(value.detach())).cpu())


def parent_fixed_point_step(state, step_size, vector_field, config):
    """Reproduce AC8/AC9 update stopping, then measure true residual."""
    trial = state + step_size * vector_field(state)
    history = []
    for iteration in range(1, config["maximum_iterations"] + 1):
        proposal = state + step_size * vector_field(0.5 * (state + trial))
        stopping_update = _inf_norm(proposal - trial)
        trial = proposal
        history.append(stopping_update)
        if not bool(torch.isfinite(trial).all()):
            raise ParentSolverFailure(
                "nonfinite_iterate",
                "nonfinite fixed-point iterate",
                trial,
                {
                    "iterations": iteration,
                    "stopping_update_history_inf": history,
                },
            )
        if stopping_update < config["tolerance"]:
            returned_residual = _inf_norm(
                endpoint_residual(state, trial, step_size, vector_field)
            )
            return trial, {
                "iterations": iteration,
                "backtracks": 0,
                "stopping_update_inf": stopping_update,
                "returned_endpoint_residual_inf": returned_residual,
                "stopping_update_history_inf": history,
            }
    returned_residual = _inf_norm(
        endpoint_residual(state, trial, step_size, vector_field)
    )
    raise ParentSolverFailure(
        "fixed_point_iteration_cap",
        f"fixed point did not converge: h={step_size:.6g}, "
        f"stopping_update_inf={history[-1]:.3e}",
        trial,
        {
            "iterations": config["maximum_iterations"],
            "backtracks": 0,
            "stopping_update_inf": history[-1],
            "returned_endpoint_residual_inf": returned_residual,
            "stopping_update_history_inf": history,
        },
    )


def _residual_and_jacobian(state, trial, step_size, vector_field):
    shape = trial.shape

    def function(flat_trial):
        endpoint = flat_trial.reshape(shape)
        midpoint = 0.5 * (state + endpoint)
        return (
            endpoint
            - state
            - step_size * vector_field(midpoint, create_graph=True)
        ).reshape(-1)

    flat = trial.detach().reshape(-1).requires_grad_(True)
    residual = function(flat)
    jacobian = torch.autograd.functional.jacobian(
        function, flat, vectorize=True
    )
    return residual.detach(), jacobian.detach()


def parent_newton_step(state, step_size, vector_field, config):
    """Reproduce AC8 monotone-residual Newton semantics."""
    trial = state + step_size * vector_field(state)
    residual_history = []
    damping_history = []
    total_backtracks = 0
    for iteration in range(1, config["maximum_iterations"] + 1):
        residual, jacobian = _residual_and_jacobian(
            state, trial, step_size, vector_field
        )
        residual_inf = _inf_norm(residual)
        residual_history.append(residual_inf)
        if residual_inf < config["tolerance"]:
            return trial, {
                "iterations": iteration,
                "backtracks": total_backtracks,
                "stopping_update_inf": None,
                "returned_endpoint_residual_inf": _inf_norm(
                    endpoint_residual(state, trial, step_size, vector_field)
                ),
                "residual_history_inf": residual_history,
                "accepted_damping_history": damping_history,
            }
        try:
            direction = torch.linalg.solve(jacobian, -residual).reshape_as(trial)
        except torch.linalg.LinAlgError as error:
            raise ParentSolverFailure(
                "singular_newton_jacobian",
                f"singular Newton Jacobian: {error}",
                trial,
                {
                    "iterations": iteration,
                    "backtracks": total_backtracks,
                    "residual_history_inf": residual_history,
                },
            ) from error
        damping = 1.0
        accepted = False
        for _ in range(config["maximum_backtracks"] + 1):
            candidate = trial + damping * direction
            candidate_inf = _inf_norm(
                endpoint_residual(state, candidate, step_size, vector_field)
            )
            if math.isfinite(candidate_inf) and candidate_inf < residual_inf:
                trial = candidate.detach()
                accepted = True
                damping_history.append(damping)
                break
            damping *= 0.5
            total_backtracks += 1
        if not accepted or damping < config["minimum_damping"]:
            raise ParentSolverFailure(
                "newton_monotone_line_search_failed",
                f"Newton line search failed: h={step_size:.6g}, "
                f"residual={residual_inf:.3e}",
                trial,
                {
                    "iterations": iteration,
                    "backtracks": total_backtracks,
                    "returned_endpoint_residual_inf": residual_inf,
                    "residual_history_inf": residual_history,
                    "accepted_damping_history": damping_history,
                },
            )
    returned_residual = _inf_norm(
        endpoint_residual(state, trial, step_size, vector_field)
    )
    raise ParentSolverFailure(
        "newton_iteration_cap",
        f"Newton did not converge: h={step_size:.6g}, "
        f"residual={returned_residual:.3e}",
        trial,
        {
            "iterations": config["maximum_iterations"],
            "backtracks": total_backtracks,
            "returned_endpoint_residual_inf": returned_residual,
            "residual_history_inf": residual_history,
            "accepted_damping_history": damping_history,
        },
    )


def classify_unexpected(error):
    if isinstance(error, ParentSolverFailure):
        return error.reason
    if isinstance(error, FloatingPointError):
        return "invalid_physical_time_increment"
    return "unexpected_error"


def load_monitor(arm, record, ac8, ac9):
    architecture = record["architecture"]
    if architecture == "manual_r":
        return lambda value: torch.linalg.vector_norm(
            value[..., :2], dim=-1
        )
    if arm == "AC8":
        monitor = ac8.StudentMonitor(architecture)
        path = os.path.join(
            ROOT,
            "results",
            "expAC8",
            f"monitor_{architecture}_seed{record['seed']}.pt",
        )
        monitor.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    else:
        monitor = ac9.Monitor(architecture)
        path = os.path.join(
            ROOT,
            "results",
            "expAC9",
            f"monitor_{architecture}_seed{record['checkpoint_seed']}.pt",
        )
        package = torch.load(path, map_location="cpu", weights_only=True)
        monitor.load_state_dict(package["state_dict"])
    return monitor.double().eval()


def initial_state_and_system(arm, record, ac8, ac9):
    if arm == "AC8":
        system = Kepler2D(record["eccentricity"])
        q0, p0 = system.initial_condition()
        q0, p0 = ac8.rotate_state(
            q0, p0, record["orientation_radians"]
        )
        if record["momentum_direction"] == "reversed":
            p0 = -p0
        semi_major_axis = 1.0
        initial = np.concatenate([q0, p0])
    else:
        system = Kepler2D(record["eccentricity"])
        semi_major_axis = record["semi_major_axis"]
        initial = ac9.kepler_state(
            semi_major_axis,
            record["eccentricity"],
            record["initial_mean_anomaly"],
        )
        initial = ac9.rotate_state(
            initial, record["orientation_radians"]
        )
        if record["momentum_direction"] == "reversed":
            initial[2:] *= -1.0
    return (
        torch.tensor(initial[None, :], dtype=torch.float64),
        system,
        semi_major_axis,
    )


def condition_sample(
    state,
    endpoint,
    step_size,
    monitor,
    system,
    p0,
    semi_major_axis,
    *,
    step,
    physical_time,
    horizon_trim,
    sample_kind,
    solver_diagnostics,
):
    midpoint = 0.5 * (state + endpoint)
    sample = {
        "sample_kind": sample_kind,
        "accepted_physical_step": step,
        "physical_time": physical_time,
        "step_size": step_size,
        "horizon_trim": bool(horizon_trim),
        "state": state.detach().reshape(-1).cpu().tolist(),
        "endpoint_or_iterate": endpoint.detach().reshape(-1).cpu().tolist(),
        "midpoint": midpoint.detach().reshape(-1).cpu().tolist(),
        "solver": solver_diagnostics,
    }
    try:
        sample["conditioning"] = conditioning_diagnostics(
            midpoint,
            monitor,
            system.H_torch,
            p0,
            step_size,
            semi_major_axis,
        )
    except Exception as error:
        sample["conditioning_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return sample


def replay_rollout(arm, parent, protocol, ac8, ac9):
    monitor = load_monitor(arm, parent, ac8, ac9)
    state, system, semi_major_axis = initial_state_and_system(
        arm, parent, ac8, ac9
    )
    initial = state.clone()
    energy0 = float(system.H(initial.numpy()[0, :2], initial.numpy()[0, 2:]))
    p0 = torch.tensor([-energy0], dtype=torch.float64)
    parent_protocol = protocol["parents"][arm]["protocol"]
    evaluation = parent_protocol["evaluation"]
    if arm == "AC8":
        target_time = evaluation["physical_horizon"]
        maximum_steps = evaluation["maximum_steps_per_rollout"]
        solver_name = parent["solver"]
    else:
        target_time = (
            2.0
            * math.pi
            * semi_major_axis**1.5
            * evaluation["periods"]
        )
        maximum_steps = evaluation["maximum_steps"]
        solver_name = "fixed_point"
    stride = protocol["sampling"]["accepted_physical_step_stride"]
    samples = []
    physical_time = 0.0
    steps = 0
    completion_status = "running"
    failure = None
    started = time.perf_counter()

    def vector_field(value, create_graph=False):
        with torch.enable_grad():
            differentiable = value
            if not differentiable.requires_grad:
                differentiable = value.detach().requires_grad_(True)
            extended = monitor(differentiable) * (
                system.H_torch(differentiable) + p0
            )
            gradient = torch.autograd.grad(
                extended.sum(), differentiable, create_graph=create_graph
            )[0]
        field = torch.cat(
            [gradient[:, 2:], -gradient[:, :2]], dim=1
        )
        return field if create_graph else field.detach()

    def solve(value, h):
        if solver_name == "damped_newton":
            return parent_newton_step(
                value, h, vector_field, evaluation["damped_newton"]
            )
        return parent_fixed_point_step(
            value, h, vector_field, evaluation["fixed_point"]
        )

    try:
        while physical_time < target_time:
            if steps >= maximum_steps:
                completion_status = "step_cap"
                failure = {
                    "reason": "maximum_steps",
                    "message": f"reached maximum_steps={maximum_steps}",
                }
                break
            old_state = state
            step_size = evaluation["dtau"]
            horizon_trim = False
            try:
                endpoint, solver_diagnostics = solve(old_state, step_size)
                physical_dt = step_size * float(
                    monitor(0.5 * (old_state + endpoint)).detach()
                )
                if not math.isfinite(physical_dt) or physical_dt <= 0.0:
                    raise FloatingPointError(
                        f"invalid physical increment {physical_dt}"
                    )
                if physical_time + physical_dt > target_time:
                    horizon_trim = True
                    remaining = target_time - physical_time
                    lower, upper = 0.0, step_size
                    for _ in range(40):
                        candidate_size = 0.5 * (lower + upper)
                        candidate, candidate_diagnostics = solve(
                            old_state, candidate_size
                        )
                        candidate_dt = candidate_size * float(
                            monitor(
                                0.5 * (old_state + candidate)
                            ).detach()
                        )
                        if candidate_dt < remaining:
                            lower = candidate_size
                        else:
                            upper = candidate_size
                        endpoint = candidate
                        step_size = candidate_size
                        solver_diagnostics = candidate_diagnostics
                    physical_dt = remaining
            except Exception as error:
                completion_status = (
                    "nonfinite"
                    if isinstance(error, FloatingPointError)
                    or (
                        isinstance(error, ParentSolverFailure)
                        and error.reason == "nonfinite_iterate"
                    )
                    else "solver_failure"
                    if isinstance(error, ParentSolverFailure)
                    else "unexpected_error"
                )
                failure = {
                    "reason": classify_unexpected(error),
                    "type": type(error).__name__,
                    "message": str(error),
                }
                if isinstance(error, ParentSolverFailure):
                    failure["solver"] = error.diagnostics
                    failure_sample = condition_sample(
                        old_state,
                        error.endpoint,
                        step_size,
                        monitor,
                        system,
                        p0,
                        semi_major_axis,
                        step=steps + 1,
                        physical_time=physical_time,
                        horizon_trim=horizon_trim,
                        sample_kind="failed_step_final_iterate",
                        solver_diagnostics=error.diagnostics,
                    )
                    if samples:
                        failure_sample["preceding_accepted_sample"] = samples[-1]
                        failure_sample[
                            "preceding_accepted_sample_feasible"
                        ] = True
                    else:
                        failure_sample[
                            "preceding_accepted_sample_feasible"
                        ] = False
                    samples.append(failure_sample)
                break

            state = endpoint
            steps += 1
            physical_time += physical_dt
            if steps % stride == 0 or horizon_trim:
                samples.append(
                    condition_sample(
                        old_state,
                        state,
                        step_size,
                        monitor,
                        system,
                        p0,
                        semi_major_axis,
                        step=steps,
                        physical_time=physical_time,
                        horizon_trim=horizon_trim,
                        sample_kind=(
                            "accepted_horizon_trim"
                            if horizon_trim
                            else "accepted_stride"
                        ),
                        solver_diagnostics=solver_diagnostics,
                    )
                )
        if completion_status == "running":
            completion_status = "complete"
    except Exception as error:
        completion_status = "unexpected_error"
        failure = {
            "reason": "replay_wrapper_error",
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    return {
        "arm": arm,
        "rollout_key": rollout_key(arm, parent),
        "parent_completion_status": parent["completion_status"],
        "replay_completion_status": completion_status,
        "completion_label_match": (
            parent["completion_status"] == completion_status
        ),
        "architecture": parent["architecture"],
        "checkpoint_seed": (
            parent.get("seed")
            if arm == "AC8"
            else parent.get("checkpoint_seed")
        ),
        "solver": solver_name,
        "semi_major_axis": semi_major_axis,
        "eccentricity": parent["eccentricity"],
        "orientation_radians": parent["orientation_radians"],
        "momentum_direction": parent["momentum_direction"],
        "initial_mean_anomaly": parent.get("initial_mean_anomaly"),
        "steps": steps,
        "physical_time": physical_time,
        "target_physical_time": target_time,
        "failure": failure,
        "samples": samples,
        "wall_seconds": time.perf_counter() - started,
    }


def rollout_key(arm, record):
    if arm == "AC8":
        values = [
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
            record["architecture"],
            record.get("seed"),
            record["solver"],
        ]
    else:
        values = [
            record["architecture"],
            record.get("checkpoint_seed"),
            record["semi_major_axis"],
            record["eccentricity"],
            record["initial_mean_anomaly"],
            record["orientation_radians"],
            record["momentum_direction"],
        ]
    return arm + "|" + "|".join(str(value) for value in values)


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    ac8 = importlib.import_module(
        "expAC8_monitor_architecture_solver_factorial"
    )
    ac9 = importlib.import_module(
        "expAC9_end_to_end_monitor_generalisation"
    )
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
        },
        "records": [],
    }
    if os.path.exists(RAW_PATH):
        artifact = load_json(RAW_PATH)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError("existing raw artifact protocol mismatch")
    existing = {record["rollout_key"] for record in artifact["records"]}

    for arm in ["AC8", "AC9"]:
        parent_path = os.path.join(
            ROOT, protocol["parents"][arm]["raw_results_path"]
        )
        parent_records = load_json(parent_path)["records"]
        for parent in parent_records:
            key = rollout_key(arm, parent)
            if key in existing:
                continue
            print(f"replay {key}", flush=True)
            record = replay_rollout(arm, parent, protocol, ac8, ac9)
            artifact["records"].append(record)
            existing.add(key)
            atomic_json(RAW_PATH, artifact)
            print(
                f"  parent={record['parent_completion_status']} "
                f"replay={record['replay_completion_status']} "
                f"samples={len(record['samples'])} "
                f"wall={record['wall_seconds']:.2f}s",
                flush=True,
            )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    artifact["all_completion_labels_match"] = all(
        record["completion_label_match"] for record in artifact["records"]
    )
    atomic_json(RAW_PATH, artifact)


if __name__ == "__main__":
    main()
