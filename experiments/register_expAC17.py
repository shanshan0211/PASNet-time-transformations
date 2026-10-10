"""Prospectively seal AC17 before H or monitor training."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC17", "preregistration.json")


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
    source_roles = [
        ("experiments/expAC17_multi_learned_h_factorial.py", "runner"),
        ("experiments/analyze_expAC17.py", "analysis"),
        ("tests/test_multi_learned_h_factorial_artifacts.py", "tests"),
        (
            "experiments/expAC12_learned_h_monitor_solver_factorial.py",
            "reused_monitor_training_and_rollout_primitives",
        ),
        ("src/implicit_midpoint_solvers.py", "corrected_solvers"),
        ("src/models/pasnet.py", "model_definition"),
        ("src/systems.py", "exact_kepler_definition"),
    ]
    monitor_training = {
        "architectures": ["raw", "invariant"],
        "replicates": [1, 2],
        "h_modes": [],
        "seed_manifest": [],
        "training_states": 4096,
        "periapsis_sampling_power": 1.5,
        "periapsis_sampling_offset": 0.05,
        "epochs": 600,
        "optimizer": "Adam",
        "learning_rate": 0.0002,
        "scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 1.0,
        "dtau": 0.08,
        "rollout_steps": 100,
        "unrolled_fixed_point_iterations": 4,
        "checkpoint_physical_dt": 0.3,
        "weight_energy_variance": 0.1,
        "weight_checkpoint": 5.0,
        "weight_inverse_monitor": 0.02,
        "validation_starts": 8,
        "validation_every_epochs": 50,
        "checkpoint_selection": "minimum validation state MSE",
        "dtype": "float64",
        "torch_threads": 1,
    }
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC17_multi_learned_h_factorial",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_hamiltonian_or_monitor_training",
        "remediation_target":
            "AC12 used one shared learned-H checkpoint and could not estimate "
            "learned-H checkpoint uncertainty.",
        "question":
            "Among prospectively fidelity-qualified independently trained "
            "learned Hamiltonians, how do H checkpoint, monitor architecture, "
            "and corrected solver affect rollout completion?",
        "hypotheses": {
            "fidelity":
                "At least three of five independently trained learned-H "
                "checkpoints may pass all held-out gates.",
            "architecture":
                "Invariant monitors may improve H-checkpoint-averaged completion.",
            "solver":
                "Newton may recover fixed-point failures, with checkpoint-level "
                "variation retained.",
        },
        "frozen_sources": [
            {
                "path": path,
                "role": role,
                "sha256": sha256_file(os.path.join(ROOT, path)),
            }
            for path, role in source_roles
        ],
        "hamiltonian_training": {
            "candidate_seeds": [1701, 1702, 1703, 1704, 1705],
            "independently_initialized": True,
            "model":
                "PASNet H_net, inputs [qx,qy,px,py,r,1/r], width 128, "
                "three hidden tanh layers",
            "target": "exact Hamiltonian gradient",
            "training_eccentricities": [0.72, 0.81, 0.9, 0.96, 0.985],
            "states_per_eccentricity": 512,
            "phase_offset": 0.5,
            "epochs": 800,
            "batch_size": 512,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "scheduler": "CosineAnnealingLR",
            "gradient_clip_norm": 10.0,
            "dtype": "float64",
            "monitor_initialization_forbidden_until_gate_complete": True,
        },
        "hamiltonian_fidelity_gate": {
            "prospectively_fixed": True,
            "held_out_from_h_training": True,
            "held_out_eccentricities": [0.7, 0.75, 0.85, 0.95, 0.99],
            "held_out_states_per_eccentricity": 257,
            "held_out_phase_offset": 0.1732050807568877,
            "metrics": [
                "Hamiltonian-gradient normalized RMSE",
                "Hamiltonian-gradient pointwise relative-error p95 and maximum",
                "canonical vector-field normalized RMSE",
                "canonical vector-field pointwise relative-error p95 and maximum",
                "per-eccentricity normalized RMSE",
                "all predictions finite",
            ],
            "maximum_normalized_rmse": 0.01,
            "maximum_stratum_normalized_rmse": 0.02,
            "maximum_relative_error_p95": 0.03,
            "maximum_relative_error_max": 0.15,
            "minimum_accepted_checkpoints": 3,
            "decision_rule":
                "Accept a checkpoint only if every gradient and vector-field "
                "gate passes. Proceed only with at least three accepted "
                "checkpoints; otherwise declare that no checkpoint ensemble "
                "qualifies and skip all monitor training and evaluation.",
            "rejected_checkpoints_retained_with_metrics_and_reasons": True,
            "no_replacement_candidates": True,
            "no_result_conditioned_threshold_changes": True,
        },
        "monitor_training": monitor_training,
        "training": monitor_training,
        "evaluation": {
            "eccentricities": [0.7, 0.9, 0.99],
            "orientations_radians": [0.0, 1.5707963267948966],
            "momentum_directions": ["forward", "reversed"],
            "initial_mean_anomaly": 3.141592653589793,
            "periods": 3,
            "dtau": 0.04,
            "solvers": ["fixed_point", "monotone_residual_newton"],
            "fixed_point": {
                "tolerance": 1e-12,
                "maximum_iterations": 100,
            },
            "monotone_residual_newton": {
                "tolerance": 1e-12,
                "maximum_iterations": 12,
                "maximum_backtracks": 8,
                "minimum_damping": 0.00390625,
            },
            "acceptance_tolerance": 1e-12,
            "horizon_bisection_iterations": 40,
            "maximum_steps": 100000,
            "scenarios_per_factorial_cell": 12,
            "rollouts_per_accepted_h_checkpoint": 96,
            "maximum_rollouts_if_all_five_h_checkpoints_pass": 480,
            "conditional_registration":
                "zero rollouts if fewer than three H checkpoints pass; "
                "otherwise 96 times the number accepted",
        },
        "design": {
            "crossed_factors": [
                "accepted learned-H checkpoint",
                "monitor architecture",
                "solver",
            ],
            "nested_factor":
                "two independently initialized monitor replicates nested within "
                "each H checkpoint by architecture",
            "primary_cluster": "learned-H checkpoint",
            "pseudoreplication_rule":
                "Average scenario outcomes within monitor replicate, average "
                "monitor replicates within H checkpoint cells, and bootstrap "
                "only H checkpoints for primary intervals.",
        },
        "primary_estimands": [
            "per-H-checkpoint completion by monitor architecture and solver",
            "invariant-minus-raw completion averaged within H checkpoint",
            "Newton-minus-fixed-point completion averaged within H checkpoint",
            "architecture-by-solver interaction within H checkpoint",
            "H-checkpoint bootstrap intervals and between-checkpoint SD",
        ],
        "censoring": {
            "all_H_training_and_gate_failures_retained": True,
            "all_monitor_training_failures_retained_as_unavailable_rollouts": True,
            "all_solver_failures_retained": True,
            "no_failed_checkpoint_replacement": True,
        },
        "acceptance": {
            "artifact_success_independent_of_scientific_success": True,
            "gate_failure_is_a_valid_terminal_result": True,
            "requirements": [
                "all five H candidates and decisions retained",
                "held-out H gates applied before monitor initialization",
                "all conditionally registered rollouts retained",
                "accepted endpoint residual strictly below 1e-12",
                "all failures preserved",
            ],
        },
        "claim_limits": [
            "Kepler a=1 planar CPU scope",
            "five H candidates and at most five H-level clusters",
            "two nested monitor replicates do not increase H-level sample size",
            "rejected checkpoints are excluded by prospective fidelity gates",
            "no factorial claim is made if fewer than three H checkpoints pass",
            "survivor-conditioned accuracy and cost cannot replace completion",
        ],
        "expected_runtime": {
            "single_cpu_core_hours_low": 3,
            "single_cpu_core_hours_high": 6,
            "basis":
                "Five float64 H fits (800 epochs over 2,560 states), up to "
                "twenty AC12-sized monitor fits, and up to 480 three-period "
                "solver rollouts.",
            "not_empirically_timed": True,
        },
        "commands": {
            "register":
                "python experiments/register_expAC17.py",
            "unit_and_smoke":
                "python -m unittest tests.test_multi_learned_h_factorial_artifacts",
            "full_run":
                "python experiments/expAC17_multi_learned_h_factorial.py",
            "analysis":
                "python experiments/analyze_expAC17.py",
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
