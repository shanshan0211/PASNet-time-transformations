"""Run sealed AC18, the robust state-generation remediation of AC17."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC18")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
H_AUDIT_PATH = os.path.join(OUT, "hamiltonian_audit.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
INFRASTRUCTURE_FAILURE_PATH = os.path.join(OUT, "infrastructure_failure.json")
sys.path.insert(0, HERE)

import expAC17_multi_learned_h_factorial as ac17  # noqa: E402


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


def verify_protocol():
    if not os.path.isfile(PROTOCOL_PATH):
        raise RuntimeError("seal AC18 with register_expAC18.py before execution")
    protocol = load_json(PROTOCOL_PATH)
    if canonical_protocol_hash(protocol) != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("AC18 protocol hash mismatch")
    for source in protocol["frozen_sources"]:
        if sha256_file(os.path.join(ROOT, source["path"])) != source["sha256"]:
            raise RuntimeError(f"frozen source changed: {source['path']}")
    return protocol


def solve_eccentric_anomaly_bisection(
    mean_anomaly, eccentricity, *, residual_tolerance=2e-14, iterations=80
):
    """Solve E - e sin(E) = M by a guaranteed monotone bracket for 0<=e<1."""
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("elliptic eccentricity must satisfy 0 <= e < 1")
    original = np.asarray(mean_anomaly, dtype=np.float64)
    if not np.isfinite(original).all():
        raise ValueError("mean anomaly must be finite")
    two_pi = 2.0 * math.pi
    cycles = np.floor(original / two_pi)
    reduced = original - cycles * two_pi
    lower = np.zeros_like(reduced)
    upper = np.full_like(reduced, two_pi)
    exact_zero = reduced == 0.0
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        residual = midpoint - eccentricity * np.sin(midpoint) - reduced
        upper = np.where(residual >= 0.0, midpoint, upper)
        lower = np.where(residual < 0.0, midpoint, lower)
    solution_reduced = np.where(exact_zero, 0.0, 0.5 * (lower + upper))
    solution = solution_reduced + cycles * two_pi
    residual = solution - eccentricity * np.sin(solution) - original
    maximum_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    if maximum_residual > residual_tolerance:
        raise RuntimeError(
            "bracketed eccentric-anomaly residual exceeded tolerance: "
            f"{maximum_residual:.3e} > {residual_tolerance:.3e}"
        )
    return float(solution) if original.ndim == 0 else solution


def registered_anomaly_grids(protocol):
    h_config = protocol["hamiltonian_training"]
    gate = protocol["hamiltonian_fidelity_gate"]
    training = (
        2.0 * math.pi
        * (np.arange(h_config["states_per_eccentricity"])
           + h_config["phase_offset"])
        / h_config["states_per_eccentricity"]
    )
    held_out = (
        2.0 * math.pi
        * (np.arange(gate["held_out_states_per_eccentricity"])
           + gate["held_out_phase_offset"])
        / gate["held_out_states_per_eccentricity"]
    )
    return {
        "hamiltonian_training": (
            h_config["training_eccentricities"], training
        ),
        "hamiltonian_fidelity_gate": (
            gate["held_out_eccentricities"], held_out
        ),
        "factorial_evaluation": (
            protocol["evaluation"]["eccentricities"],
            np.asarray([protocol["evaluation"]["initial_mean_anomaly"]]),
        ),
        "monitor_training_and_validation": (
            [protocol["monitor_training"]["training_orbit_eccentricity"]],
            np.concatenate([training, held_out]),
        ),
    }


def validate_registered_state_generation(protocol):
    tolerance = protocol["state_generation"]["residual_tolerance"]
    records = []
    for purpose, (eccentricities, anomalies) in registered_anomaly_grids(
        protocol
    ).items():
        for eccentricity in eccentricities:
            solution = solve_eccentric_anomaly_bisection(
                anomalies,
                eccentricity,
                residual_tolerance=tolerance,
                iterations=protocol["state_generation"]["bisection_iterations"],
            )
            residual = (
                solution - eccentricity * np.sin(solution) - anomalies
            )
            states = robust_kepler_state(1.0, eccentricity, anomalies)
            records.append({
                "purpose": purpose,
                "eccentricity": eccentricity,
                "count": len(anomalies),
                "maximum_residual": float(np.max(np.abs(residual))),
                "all_states_finite": bool(np.isfinite(states).all()),
                "minimum_radius": float(
                    np.min(np.linalg.norm(states[:, :2], axis=1))
                ),
            })
    if not all(
        record["maximum_residual"] <= tolerance
        and record["all_states_finite"]
        and record["minimum_radius"] > 0.0
        for record in records
    ):
        raise RuntimeError("registered state-generation preflight failed")
    return records


def robust_kepler_state(semi_major_axis, eccentricity, mean_anomaly):
    mean_anomaly = np.asarray(mean_anomaly, dtype=np.float64)
    eccentric_anomaly = solve_eccentric_anomaly_bisection(
        mean_anomaly, eccentricity
    )
    root = math.sqrt(1.0 - eccentricity**2)
    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    scale = math.sqrt(semi_major_axis)
    qx = semi_major_axis * (eccentricity - np.cos(eccentric_anomaly))
    qy = -semi_major_axis * root * np.sin(eccentric_anomaly)
    px = np.sin(eccentric_anomaly) / (scale * denominator)
    py = -root * np.cos(eccentric_anomaly) / (scale * denominator)
    return np.stack([qx, qy, px, py], axis=-1)


def configure_ac17_runtime():
    ac17.OUT = OUT
    ac17.PROTOCOL_PATH = PROTOCOL_PATH
    ac17.H_AUDIT_PATH = H_AUDIT_PATH
    ac17.TRAINING_PATH = TRAINING_PATH
    ac17.RAW_PATH = RAW_PATH
    ac17.SUMMARY_PATH = SUMMARY_PATH
    ac17.ac12.solve_eccentric_anomaly = solve_eccentric_anomaly_bisection


def runtime_protocol(protocol):
    configured = copy.deepcopy(protocol)
    configured["training"] = configured["monitor_training"]
    return configured


def train_and_gate_hamiltonians(protocol):
    validate_registered_state_generation(protocol)
    audit = ac17.train_and_gate_hamiltonians(protocol)
    infrastructure_failures = [
        record for record in audit["records"]
        if record["training_status"] == "training_or_audit_failure"
    ]
    if infrastructure_failures:
        failure = {
            "schema_version": "1.0",
            "experiment": protocol["experiment"],
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "invalid_infrastructure_failure",
            "scientific_gate_result_available": False,
            "records": infrastructure_failures,
        }
        ac17.atomic_json(INFRASTRUCTURE_FAILURE_PATH, failure)
        raise RuntimeError(
            "AC18 infrastructure failure; no fidelity-gate null may be claimed"
        )
    return audit


def smoke_h_training_and_gating(protocol):
    """Enter one real H optimizer step and compute held-out gate metrics."""
    preflight = validate_registered_state_generation(protocol)
    config = protocol["hamiltonian_training"]
    seed = config["candidate_seeds"][0]
    torch.manual_seed(seed)
    model = ac17.make_h_template()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    anomalies = (
        2.0 * math.pi
        * (np.arange(8) + config["phase_offset"]) / 8
    )
    states_np, gradients_np, _, _ = ac17.states_and_fields(
        config["training_eccentricities"], anomalies
    )
    states = torch.tensor(states_np, dtype=torch.float64)
    targets = torch.tensor(gradients_np, dtype=torch.float64)
    prediction = ac17.learned_gradient(model, states, create_graph=True)
    loss = torch.mean((prediction - targets) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    gate = protocol["hamiltonian_fidelity_gate"]
    held_anomalies = (
        2.0 * math.pi
        * (np.arange(8) + gate["held_out_phase_offset"]) / 8
    )
    held_states, held_gradients, held_fields, _ = ac17.states_and_fields(
        gate["held_out_eccentricities"], held_anomalies
    )
    predicted_gradient = ac17.learned_gradient(
        model,
        torch.tensor(held_states, dtype=torch.float64),
        create_graph=False,
    ).detach().numpy()
    predicted_field = np.concatenate(
        [predicted_gradient[:, 2:], -predicted_gradient[:, :2]], axis=1
    )
    gradient_nrmse = float(
        np.sqrt(np.mean((predicted_gradient - held_gradients) ** 2))
        / np.sqrt(np.mean(held_gradients**2))
    )
    field_nrmse = float(
        np.sqrt(np.mean((predicted_field - held_fields) ** 2))
        / np.sqrt(np.mean(held_fields**2))
    )
    return {
        "status": "smoke_complete",
        "registered_state_generation_cases": len(preflight),
        "optimizer_steps_completed": 1,
        "finite_training_loss": bool(math.isfinite(float(loss.detach()))),
        "gate_metrics_computed": True,
        "held_out_gradient_normalized_rmse": gradient_nrmse,
        "held_out_vector_field_normalized_rmse": field_nrmse,
        "scientific_thresholds_applied_to_smoke": False,
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    configure_ac17_runtime()
    protocol = runtime_protocol(verify_protocol())
    if "--smoke" in sys.argv[1:]:
        print(json.dumps(smoke_h_training_and_gating(protocol), indent=2))
        return
    audit = train_and_gate_hamiltonians(protocol)
    training = ac17.train_monitors(protocol, audit)
    raw = ac17.evaluate_all(protocol, audit, training)
    summary = ac17.summarize(protocol, audit, training, raw)
    ac17.atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("AC18 artifact gate failed")


if __name__ == "__main__":
    main()
