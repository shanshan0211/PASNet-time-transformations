"""Register the AC8 monitor-architecture by nonlinear-solver factorial."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
OUTPUT = os.path.join(RESULTS, "expAC8", "preregistration.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")
    checkpoint = os.path.join(RESULTS, "expE2", "pasnet_weights.pt")
    ac6_raw = os.path.join(RESULTS, "expAC6", "raw_results.json")
    ac6_summary = os.path.join(RESULTS, "expAC6", "summary.json")
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC8_monitor_architecture_solver_factorial",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_training_or_evaluation",
        "question":
            "Do symmetry-preserving monitor inputs improve orbit "
            "generalisation, and which failures remain solver-dependent?",
        "frozen_teacher": {
            "path": "results/expE2/pasnet_weights.pt",
            "sha256": sha256_file(checkpoint),
            "role":
                "orbit-conditioned monitor target; H network is not used",
        },
        "frozen_sources": [
            {
                "path": "results/expAC6/raw_results.json",
                "sha256": sha256_file(ac6_raw),
            },
            {
                "path": "results/expAC6/summary.json",
                "sha256": sha256_file(ac6_summary),
            },
        ],
        "training": {
            "task":
                "distill the same frozen teacher g values on the canonical "
                "forward e=0.9 orbit; no rollout result enters training",
            "state_generation":
                "exact Kepler a=1 states at 4096 equally spaced mean "
                "anomalies; eccentric anomaly solved to 1e-14",
            "train_indices": "first 3 of every 4 samples",
            "validation_indices": "every fourth sample",
            "teacher_dtype": "float64",
            "student_dtype": "float64",
            "architectures": {
                "raw": {
                    "features": ["qx", "qy", "px", "py"],
                    "input_dimension": 4,
                },
                "invariant": {
                    "features": [
                        "r",
                        "1/r",
                        "p_squared",
                        "(q_dot_p)^2",
                        "angular_momentum^2",
                    ],
                    "input_dimension": 5,
                    "guarantees": [
                        "SO(2) rotation invariance",
                        "momentum-reversal evenness",
                    ],
                },
            },
            "shared_network": {
                "hidden_width": 64,
                "hidden_layers": 2,
                "activation": "tanh",
                "output": "2*sigmoid(logit)",
            },
            "seeds": [6101, 6102, 6103, 6104, 6105],
            "epochs": 800,
            "batch_size": 512,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "scheduler": "CosineAnnealingLR",
            "loss": "mean squared error to frozen teacher g",
            "checkpoint_selection":
                "minimum validation MSE evaluated every 20 epochs",
            "no_result_conditioned_tuning": True,
        },
        "evaluation": {
            "hamiltonian": "exact Kepler H",
            "eccentricities": [0.7, 0.9, 0.99],
            "orientations_radians": [0.0, 1.5707963267948966],
            "momentum_directions": ["forward", "reversed"],
            "periods": 3,
            "physical_horizon": 18.84955592153876,
            "dtau": 0.04,
            "student_architectures": ["raw", "invariant"],
            "solvers": ["fixed_point", "damped_newton"],
            "manual_r_positive_control": True,
            "learned_rollouts": 240,
            "manual_control_rollouts": 24,
            "total_registered_rollouts": 264,
            "dtype": "float64",
            "torch_threads": 1,
            "fixed_point": {
                "tolerance": 1e-12,
                "maximum_iterations": 100,
            },
            "damped_newton": {
                "tolerance": 1e-12,
                "maximum_iterations": 12,
                "maximum_backtracks": 8,
                "minimum_damping": 0.00390625,
                "linear_solve": "torch.linalg.solve on 4x4 Jacobian",
            },
            "maximum_steps_per_rollout": 100000,
            "matched_final_physical_time": True,
        },
        "frozen_outcomes": {
            "training": [
                "train_mse",
                "validation_mse",
                "checkpoint_sha256",
                "symmetry_unit_tests",
            ],
            "rollout": [
                "completion_status",
                "energy_error",
                "closure_error",
                "angular_momentum_error",
                "steps",
                "vector_field_evaluations",
                "jacobian_evaluations",
                "nonlinear_iterations",
                "backtracks",
                "wall_seconds",
                "final_time_error",
            ],
            "primary_estimands": [
                "completion fraction by architecture and solver",
                "invariant/raw paired error and cost ratios",
                "Newton/fixed-point recovery counts",
                "eccentricity and symmetry-stratified completion",
            ],
        },
        "acceptance": {
            "artifact_gate": [
                "all ten registered students trained and retained",
                "all 264 rollouts attempted and retained",
                "completed final-time error below 1e-10",
                "completed nonlinear residual below 1e-11",
                "invariant monitor symmetry relative error below 1e-12",
                "source hashes unchanged",
            ],
            "scientific_success_not_required": True,
            "claim_rule":
                "report architecture and solver factors regardless of sign",
        },
        "censoring": {
            "all_failures_retained": True,
            "finite_pair_ratios_report_denominator": True,
        },
        "hardware_registration": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "integrity": {"protocol_sha256": None},
    }
    protocol["integrity"]["protocol_sha256"] = canonical_hash(protocol)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=False)
    with open(OUTPUT, "x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
        handle.write("\n")
    print(protocol["integrity"]["protocol_sha256"])


if __name__ == "__main__":
    main()
