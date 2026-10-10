"""Run the sealed Exp AC6 orbit-generalisation audit.

The experiment keeps the Hamiltonian exact and compares the frozen learned
monitor with the radial Sundman positive control across eccentricity,
orientation, and momentum reversal. Every registered rollout is retained.
"""

from __future__ import annotations

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
RESULTS = os.path.join(ROOT, "results")
PROTOCOL_PATH = os.path.join(RESULTS, "expAC6", "preregistration.json")
RAW_PATH = os.path.join(RESULTS, "expAC6", "raw_results.json")
SUMMARY_PATH = os.path.join(RESULTS, "expAC6", "summary.json")
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.pasnet import PASNet  # noqa: E402
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
    paths = [protocol["frozen_checkpoint"], *protocol["frozen_sources"]]
    for source in paths:
        actual_source = sha256_file(os.path.join(ROOT, source["path"]))
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


def h_features(z):
    q = z[:, :2]
    radius = torch.linalg.vector_norm(q, dim=1, keepdim=True)
    return torch.cat([radius, 1.0 / radius], dim=1)


def load_model(protocol):
    model = PASNet(
        dim=2,
        hidden_H=128,
        n_hidden_H=3,
        hidden_g=64,
        n_hidden_g=2,
        g_max=2.0,
        h_features=h_features,
    )
    checkpoint = os.path.join(
        ROOT, protocol["frozen_checkpoint"]["path"]
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.H_net.load_state_dict(state["H_net"])
    model.g_net.load_state_dict(state["g_net"])
    return model.double().eval()


def angular_momentum(z):
    return float(z[0] * z[3] - z[1] * z[2])


def rotate_state(q, p, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return rotation @ q, rotation @ p


def transformed_step(z, h, vector_field, tolerance, max_iterations):
    evaluations = 1
    z_next = z + h * vector_field(z)
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        proposal = z + h * vector_field(0.5 * (z + z_next))
        evaluations += 1
        residual = float(torch.max(torch.abs(proposal - z_next)))
        z_next = proposal
        if not torch.isfinite(z_next).all():
            raise FloatingPointError("nonfinite midpoint iterate")
        if residual < tolerance:
            return z_next, residual, iteration, evaluations
    raise RuntimeError(
        f"fixed point did not converge: h={h:.6g}, residual={residual:.3e}"
    )


def run_rollout(system, model, q0, p0, method, design):
    z0_numpy = np.concatenate([q0, p0])
    z = torch.tensor(z0_numpy[None, :], dtype=torch.float64)
    energy0 = float(system.H(q0, p0))
    momentum0 = angular_momentum(z0_numpy)
    p0_extended = torch.tensor([-energy0], dtype=torch.float64)
    tolerance = design["fixed_point_tolerance"]
    max_iterations = design["fixed_point_max_iterations"]
    dtau = design["dtau"]
    target_time = design["physical_horizon"]

    def monitor(state):
        if method == "manual_r":
            return torch.linalg.vector_norm(state[:, :2], dim=1)
        return model.g(state)

    def vector_field(state):
        with torch.enable_grad():
            differentiable = state.detach().requires_grad_(True)
            extended_hamiltonian = monitor(differentiable) * (
                system.H_torch(differentiable) + p0_extended
            )
            gradient = torch.autograd.grad(
                extended_hamiltonian.sum(), differentiable
            )[0]
        return torch.cat(
            [gradient[:, 2:], -gradient[:, :2]], dim=1
        ).detach()

    record = {
        "completion_status": "running",
        "method": method,
        "steps": 0,
        "gradient_evaluations": 0,
        "alignment_gradient_evaluations": 0,
        "max_fp_residual": 0.0,
        "max_fp_iterations": 0,
        "max_energy_drift": 0.0,
        "max_angular_momentum_error": 0.0,
        "min_monitor": math.inf,
        "max_monitor": 0.0,
        "physical_time": 0.0,
    }
    started = time.perf_counter()
    try:
        while record["physical_time"] < target_time:
            if record["steps"] >= design["maximum_steps_per_rollout"]:
                record["completion_status"] = "step_cap"
                break
            trial, residual, iterations, evaluations = transformed_step(
                z, dtau, vector_field, tolerance, max_iterations
            )
            midpoint = 0.5 * (z + trial)
            physical_dt = dtau * float(monitor(midpoint))
            if not math.isfinite(physical_dt) or physical_dt <= 0.0:
                raise FloatingPointError(
                    f"invalid physical time increment {physical_dt}"
                )

            if record["physical_time"] + physical_dt > target_time:
                remaining = target_time - record["physical_time"]
                lower, upper = 0.0, dtau
                alignment_evaluations = 0
                for _ in range(45):
                    candidate_h = 0.5 * (lower + upper)
                    candidate, cand_res, cand_iter, cand_evals = (
                        transformed_step(
                            z,
                            candidate_h,
                            vector_field,
                            tolerance,
                            max_iterations,
                        )
                    )
                    candidate_midpoint = 0.5 * (z + candidate)
                    candidate_dt = candidate_h * float(
                        monitor(candidate_midpoint)
                    )
                    alignment_evaluations += cand_evals
                    if candidate_dt < remaining:
                        lower = candidate_h
                    else:
                        upper = candidate_h
                    trial = candidate
                    residual = cand_res
                    iterations = cand_iter
                    physical_dt = candidate_dt
                record["alignment_gradient_evaluations"] += (
                    alignment_evaluations
                )
                evaluations += alignment_evaluations
                physical_dt = remaining

            z = trial
            record["physical_time"] += physical_dt
            record["steps"] += 1
            record["gradient_evaluations"] += evaluations
            record["max_fp_residual"] = max(
                record["max_fp_residual"], residual
            )
            record["max_fp_iterations"] = max(
                record["max_fp_iterations"], iterations
            )
            monitor_value = float(monitor(z))
            record["min_monitor"] = min(
                record["min_monitor"], monitor_value
            )
            record["max_monitor"] = max(
                record["max_monitor"], monitor_value
            )
            state_numpy = z.detach().numpy()[0]
            q, p = state_numpy[:2], state_numpy[2:]
            record["max_energy_drift"] = max(
                record["max_energy_drift"],
                abs(float(system.H(q, p)) - energy0),
            )
            record["max_angular_momentum_error"] = max(
                record["max_angular_momentum_error"],
                abs(angular_momentum(state_numpy) - momentum0),
            )

        if record["completion_status"] == "running":
            record["completion_status"] = "complete"
    except RuntimeError as error:
        record["completion_status"] = "solver_failure"
        record["error"] = str(error)
    except FloatingPointError as error:
        record["completion_status"] = "nonfinite"
        record["error"] = str(error)
    except Exception as error:  # retain unexpected failures in the artifact
        record["completion_status"] = "unexpected_error"
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()

    final_state = z.detach().numpy()[0]
    record.update(
        final_state_closure_error=float(
            np.linalg.norm(final_state - z0_numpy)
        ),
        final_time_error=abs(record["physical_time"] - target_time),
        wall_seconds=time.perf_counter() - started,
    )
    if not math.isfinite(record["min_monitor"]):
        record["min_monitor"] = None
    return record


def safe_ratio(numerator, denominator):
    if (
        numerator is None
        or denominator is None
        or numerator <= 0.0
        or denominator <= 0.0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return None
    return numerator / denominator


def geometric_mean(values):
    finite = [
        value
        for value in values
        if value is not None and value > 0.0 and math.isfinite(value)
    ]
    if not finite:
        return None
    return float(math.exp(np.mean(np.log(finite))))


def summarize(records, protocol):
    complete = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    failure_counts = {}
    for record in records:
        status = record["completion_status"]
        failure_counts[status] = failure_counts.get(status, 0) + 1

    paired_method_ratios = {"energy": [], "closure": [], "steps": [], "wall": []}
    reversal_ratios = {"manual_r": [], "learned_g": []}
    rotation_groups = []
    index = {
        (
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
            record["method"],
        ): record
        for record in records
    }
    design = protocol["design"]
    for eccentricity in design["eccentricities"]:
        for angle in design["orientation_radians"]:
            for direction in design["momentum_directions"]:
                manual = index[(eccentricity, angle, direction, "manual_r")]
                learned = index[(eccentricity, angle, direction, "learned_g")]
                if (
                    manual["completion_status"] == "complete"
                    and learned["completion_status"] == "complete"
                ):
                    paired_method_ratios["energy"].append(
                        safe_ratio(
                            learned["max_energy_drift"],
                            manual["max_energy_drift"],
                        )
                    )
                    paired_method_ratios["closure"].append(
                        safe_ratio(
                            learned["final_state_closure_error"],
                            manual["final_state_closure_error"],
                        )
                    )
                    paired_method_ratios["steps"].append(
                        safe_ratio(learned["steps"], manual["steps"])
                    )
                    paired_method_ratios["wall"].append(
                        safe_ratio(
                            learned["wall_seconds"], manual["wall_seconds"]
                        )
                    )
            for method in design["methods"]:
                forward = index[(eccentricity, angle, "forward", method)]
                reversed_record = index[
                    (eccentricity, angle, "reversed", method)
                ]
                if (
                    forward["completion_status"] == "complete"
                    and reversed_record["completion_status"] == "complete"
                ):
                    reversal_ratios[method].append(
                        safe_ratio(
                            forward["max_energy_drift"],
                            reversed_record["max_energy_drift"],
                        )
                    )

        for direction in design["momentum_directions"]:
            for method in design["methods"]:
                group = [
                    index[(eccentricity, angle, direction, method)]
                    for angle in design["orientation_radians"]
                ]
                finite_group = [
                    record
                    for record in group
                    if record["completion_status"] == "complete"
                    and record["max_energy_drift"] > 0.0
                ]
                energy_ratio = None
                closure_ratio = None
                if len(finite_group) == len(group):
                    energy_values = [
                        record["max_energy_drift"] for record in finite_group
                    ]
                    closure_values = [
                        record["final_state_closure_error"]
                        for record in finite_group
                    ]
                    energy_ratio = max(energy_values) / min(energy_values)
                    closure_ratio = max(closure_values) / min(closure_values)
                rotation_groups.append(
                    {
                        "eccentricity": eccentricity,
                        "momentum_direction": direction,
                        "method": method,
                        "n_registered": len(group),
                        "n_complete": len(finite_group),
                        "max_min_energy_error_ratio": energy_ratio,
                        "max_min_closure_error_ratio": closure_ratio,
                    }
                )

    final_time_valid = all(
        record["final_time_error"] < 1e-10 for record in complete
    )
    residual_valid = all(
        record["max_fp_residual"] < 1e-11 for record in complete
    )
    acceptance = {
        "all_80_attempted_and_retained": len(records)
        == design["total_registered_rollouts"],
        "completed_final_times_aligned": final_time_valid,
        "completed_solver_residuals_valid": residual_valid,
        "failure_counts_reported": sum(failure_counts.values()) == len(records),
        "protocol_and_sources_valid": True,
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "counts": {
            "registered": design["total_registered_rollouts"],
            "attempted": len(records),
            "complete": len(complete),
            "by_status": failure_counts,
        },
        "paired_learned_over_manual": {
            metric: {
                "geometric_mean_ratio": geometric_mean(values),
                "n_finite_pairs": sum(value is not None for value in values),
            }
            for metric, values in paired_method_ratios.items()
        },
        "forward_over_reversed_energy": {
            method: {
                "geometric_mean_ratio": geometric_mean(values),
                "n_finite_pairs": sum(value is not None for value in values),
                "minimum_ratio": min(
                    (value for value in values if value is not None),
                    default=None,
                ),
                "maximum_ratio": max(
                    (value for value in values if value is not None),
                    default=None,
                ),
            }
            for method, values in reversal_ratios.items()
        },
        "rotation_groups": rotation_groups,
        "acceptance": acceptance,
        "claim_limits": {
            "true_hamiltonian_only": True,
            "one_frozen_checkpoint": True,
            "learned_hamiltonian_generalisation_tested": False,
            "registered_orbits_are_analysis_units_not_random_seeds": True,
        },
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    design = protocol["design"]
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

    for eccentricity in design["eccentricities"]:
        system = Kepler2D(eccentricity)
        base_q, base_p = system.initial_condition()
        for angle in design["orientation_radians"]:
            rotated_q, rotated_p = rotate_state(base_q, base_p, angle)
            for direction in design["momentum_directions"]:
                p0 = (
                    rotated_p
                    if direction == "forward"
                    else -rotated_p
                )
                for method in design["methods"]:
                    print(
                        f"e={eccentricity:.2f} angle={angle:.3f} "
                        f"{direction} {method}",
                        flush=True,
                    )
                    result = run_rollout(
                        system, model, rotated_q, p0, method, design
                    )
                    result.update(
                        eccentricity=eccentricity,
                        orientation_radians=angle,
                        momentum_direction=direction,
                    )
                    artifact["records"].append(result)
                    atomic_json(RAW_PATH, artifact)
                    print(
                        f"  {result['completion_status']} "
                        f"dE={result['max_energy_drift']:.3e} "
                        f"closure={result['final_state_closure_error']:.3e} "
                        f"steps={result['steps']}",
                        flush=True,
                    )

    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    summary = summarize(artifact["records"], protocol)
    summary["raw_results_sha256"] = sha256_file(RAW_PATH)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("Exp AC6 artifact gate failed")


if __name__ == "__main__":
    main()
