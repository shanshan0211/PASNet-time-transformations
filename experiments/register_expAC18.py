"""Prospectively seal AC18 before H or monitor training."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC18", "preregistration.json")


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
        (
            "experiments/expAC18_robust_multi_learned_h_factorial.py",
            "runner_and_robust_state_generation",
        ),
        ("experiments/analyze_expAC18.py", "analysis_entrypoint"),
        (
            "tests/test_robust_multi_learned_h_factorial_artifacts.py",
            "regression_protocol_and_smoke_tests",
        ),
        (
            "experiments/expAC17_multi_learned_h_factorial.py",
            "preserved_factorial_primitives",
        ),
        ("experiments/analyze_expAC17.py", "checkpoint_analysis_primitives"),
        (
            "experiments/expAC12_learned_h_monitor_solver_factorial.py",
            "monitor_and_rollout_primitives",
        ),
        ("results/expAC17/invalidation.json", "ac17_invalidation_evidence"),
        ("src/implicit_midpoint_solvers.py", "corrected_solvers"),
        ("src/models/pasnet.py", "model_definition"),
        ("src/systems.py", "exact_kepler_definition"),
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC18_robust_multi_learned_h_factorial",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_hamiltonian_or_monitor_training",
        "remediation_target": {
            "experiment": "expAC17_multi_learned_h_factorial",
            "protocol_sha256":
                "fc37688fa7e7a65d053fac17795c3ee84db2df39593f87cf054f30a7dbddafe5",
            "failure":
                "all candidates failed before optimization because shared "
                "eccentric-anomaly Newton iteration did not converge",
            "classification": "infrastructure_failure_not_fidelity_gate_null",
        },
        "question":
            "Among prospectively fidelity-qualified independently trained "
            "learned Hamiltonians, how do H checkpoint, monitor architecture, "
            "and corrected solver affect rollout completion?",
        "frozen_sources": [
            {
                "path": path,
                "role": role,
                "sha256": sha256_file(os.path.join(ROOT, path)),
            }
            for path, role in source_roles
        ],
        "state_generation": {
            "equation": "E - e sin(E) = M",
            "domain": "elliptic 0 <= e < 1",
            "algorithm":
                "reduce M by integer 2*pi cycles; bracket reduced E in "
                "[0,2*pi]; run componentwise monotone bisection; restore cycles",
            "monotonicity_basis": "1 - e cos(E) >= 1 - e > 0",
            "bisection_iterations": 80,
            "residual_tolerance": 2e-14,
            "preflight_required": True,
            "preflight_scope": [
                "every registered H-training eccentricity and anomaly",
                "every registered held-out-gate eccentricity and anomaly",
                "every factorial evaluation eccentricity and initial anomaly",
                "monitor training and validation state grids",
            ],
            "preflight_requirements": [
                "Kepler-equation residual <= 2e-14",
                "all generated states finite",
                "strictly positive radius",
            ],
        },
        "hamiltonian_training": {
            "candidate_seeds": [1801, 1802, 1803, 1804, 1805],
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
            "threshold_relation_to_ac17": "unchanged",
            "decision_rule":
                "Accept only checkpoints passing every gradient and vector-field "
                "gate. Proceed only if at least three checkpoints are accepted; "
                "otherwise report a valid fidelity null and skip the factorial.",
            "rejected_checkpoints_retained_with_metrics_and_reasons": True,
            "no_replacement_candidates": True,
            "no_result_conditioned_threshold_changes": True,
        },
        "monitor_training": {
            "architectures": ["raw", "invariant"],
            "replicates": [1, 2],
            "training_orbit_eccentricity": 0.9,
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
        },
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
        },
        "design": {
            "crossed_factors": [
                "accepted learned-H checkpoint",
                "monitor architecture",
                "solver",
            ],
            "nested_factor":
                "two monitor replicates nested within H checkpoint and architecture",
            "primary_cluster": "learned-H checkpoint",
            "pseudoreplication_rule":
                "Average scenarios within monitor replicate and monitor "
                "replicates within H cells; bootstrap only H checkpoints.",
        },
        "infrastructure_failure_policy": {
            "cannot_be_interpreted_as_fidelity_rejection": True,
            "cannot_be_interpreted_as_valid_gate_null": True,
            "write_separate_failure_artifact_and_abort": True,
            "scientific_gate_rejection_requires_completed_fidelity_metrics": True,
        },
        "smoke_test": {
            "command":
                ".venv/bin/python experiments/"
                "expAC18_robust_multi_learned_h_factorial.py --smoke",
            "must_complete_registered_state_preflight": True,
            "must_complete_real_optimizer_steps": 1,
            "must_compute_held_out_gradient_and_vector_field_metrics": True,
            "does_not_apply_scientific_acceptance_thresholds": True,
            "writes_no_experiment_result_artifacts": True,
        },
        "primary_estimands": [
            "per-H-checkpoint completion by monitor architecture and solver",
            "invariant-minus-raw completion within H checkpoint",
            "Newton-minus-fixed-point completion within H checkpoint",
            "architecture-by-solver interaction within H checkpoint",
            "H-checkpoint bootstrap intervals and between-checkpoint SD",
        ],
        "censoring": {
            "all_H_gate_rejections_retained": True,
            "all_monitor_and_solver_failures_retained": True,
            "no_failed_checkpoint_replacement": True,
            "infrastructure_failures_separated_from_scientific_rejections": True,
        },
        "claim_limits": [
            "Kepler a=1 planar CPU scope",
            "five H candidates and at most five H-level clusters",
            "two nested monitor replicates do not increase H-level sample size",
            "no factorial claim if fewer than three H checkpoints pass",
            "survivor-conditioned accuracy and cost cannot replace completion",
        ],
        "expected_runtime": {
            "single_cpu_core_hours_low": 3,
            "single_cpu_core_hours_high": 6,
            "basis":
                "AC17 design unchanged except robust state generation and "
                "preflight; five H fits, up to twenty monitor fits, and up to "
                "480 three-period rollouts.",
            "not_empirically_timed": True,
        },
        "commands": {
            "register": ".venv/bin/python experiments/register_expAC18.py",
            "unit_and_smoke":
                ".venv/bin/python -m unittest "
                "tests.test_robust_multi_learned_h_factorial_artifacts",
            "smoke_runner":
                ".venv/bin/python experiments/"
                "expAC18_robust_multi_learned_h_factorial.py --smoke",
            "full_run":
                ".venv/bin/python experiments/"
                "expAC18_robust_multi_learned_h_factorial.py",
            "analysis": ".venv/bin/python experiments/analyze_expAC18.py",
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
