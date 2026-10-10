"""Preregister and seal the AC12 learned-H monitor/solver factorial."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC12", "preregistration.json")


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


def sealed_entry(relative_path, role):
    absolute = os.path.join(ROOT, relative_path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"required {role} is absent: {relative_path}")
    return {
        "path": relative_path,
        "sha256": sha256_file(absolute),
        "role": role,
    }


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")
    learned_checkpoint = "results/expE2/pasnet_weights.pt"
    source_roles = [
        (
            "experiments/expAC12_learned_h_monitor_solver_factorial.py",
            "ac12_runner",
        ),
        ("experiments/analyze_expAC12.py", "ac12_analysis"),
        ("tests/test_learned_h_factorial_artifacts.py", "ac12_tests"),
        ("src/implicit_midpoint_solvers.py", "corrected_solver_module"),
        ("src/models/pasnet.py", "model_definition"),
        ("src/systems.py", "exact_kepler_definition"),
        (learned_checkpoint, "single_frozen_learned_h_checkpoint"),
    ]
    seed_manifest = [
        {
            "checkpoint_seed": seed,
            "sampling_seed": seed + 10000,
            "validation_seed": seed + 20000,
            "raw_initialization_seed": seed + 30000,
            "invariant_initialization_seed": seed + 40000,
        }
        for seed in [8101, 8102, 8103, 8104, 8105]
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC12_learned_h_monitor_solver_factorial",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_ac12_training_or_evaluation",
        "question":
            "How do exact versus one frozen learned Hamiltonian, raw versus "
            "invariant monitor architecture, and corrected fixed-point versus "
            "Newton solvers interact on Kepler rollout completion?",
        "hypotheses": {
            "hamiltonian_architecture_interaction":
                "The architecture completion contrast may differ between exact "
                "and learned H.",
            "hamiltonian_solver_interaction":
                "Newton recovery may differ between exact and learned H.",
            "positive_control":
                "Manual g=r is evaluated in every H-by-solver arm without "
                "being assumed to complete.",
        },
        "frozen_sources": [
            sealed_entry(path, role) for path, role in source_roles
        ],
        "hamiltonians": {
            "exact": {
                "definition": "Kepler2D.H_torch",
                "dtype": "float64",
            },
            "learned": {
                "definition":
                    "H_net only from the E2 PASNet; inputs [z,r,1/r]",
                "checkpoint_path": learned_checkpoint,
                "checkpoint_sha256": sha256_file(
                    os.path.join(ROOT, learned_checkpoint)
                ),
                "entire_checkpoint_hash_frozen": True,
                "H_net_parameters_frozen": True,
                "g_net_not_used": True,
                "dtype_conversion": "load state, convert H_net to float64",
            },
        },
        "training": {
            "design":
                "five paired seeds per H mode by architecture; each monitor "
                "is trained through its assigned H mode",
            "h_modes": ["exact", "learned"],
            "architectures": ["raw", "invariant"],
            "seed_manifest": seed_manifest,
            "checkpoint_count": 20,
            "p0": "-H_mode(z0), detached",
            "training_orbit": {
                "semi_major_axis": 1.0,
                "eccentricity": 0.9,
                "direction": "forward",
                "orientation_radians": 0.0,
            },
            "task":
                "AC9 end-to-end unrolled implicit-midpoint rollout training "
                "with the same objective and selection in both H modes",
            "raw_features": ["qx", "qy", "px", "py"],
            "invariant_features": [
                "r",
                "1/r",
                "p_squared",
                "(q_dot_p)^2",
                "angular_momentum^2",
            ],
            "shared_network": {
                "hidden_width": 64,
                "hidden_layers": 2,
                "activation": "tanh",
                "output": "2*sigmoid(logit)",
                "cold_start_final_bias": -2.2,
            },
            "training_states": 4096,
            "state_generation":
                "exact a=1,e=0.9 states equally spaced in mean anomaly",
            "sampling":
                "probability proportional to 1/(r^1.5+0.05)",
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
            "no_result_conditioned_tuning": True,
        },
        "evaluation": {
            "semi_major_axis": 1.0,
            "initial_mean_anomaly": 3.141592653589793,
            "eccentricities": [0.7, 0.9, 0.99],
            "orientations_radians": [0.0, 1.5707963267948966],
            "momentum_directions": ["forward", "reversed"],
            "periods": 3,
            "physical_horizon": 18.84955592153876,
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
            "acceptance_criterion":
                "returned endpoint residual infinity norm strictly below "
                "1e-12; update norm is not an acceptance criterion",
            "horizon_bisection_iterations": 40,
            "maximum_steps": 100000,
            "manual_monitor": "g=r positive control under every H and solver",
            "learned_rollouts": 480,
            "manual_control_rollouts": 48,
            "total_registered_rollouts": 528,
            "dtype": "float64",
            "torch_threads": 1,
        },
        "primary_estimands": [
            "completion by H mode by architecture by solver",
            "H-mode by architecture interaction on completion",
            "H-mode by solver interaction on completion",
            "checkpoint-cluster uncertainty with n=5 paired monitor seeds",
            "rotation and momentum-reversal symmetry stratification",
            "Newton recovery of fixed-point failures",
            "exact-H versus learned-H completion contrast",
        ],
        "secondary_estimands": [
            "survivor-conditioned exact-H and mode-H errors",
            "survivor-conditioned closure, angular-momentum, cost, and wall time",
            "failure reason counts",
        ],
        "uncertainty": {
            "cluster_unit": "checkpoint seed within H mode and architecture",
            "n_clusters_per_H_by_architecture": 5,
            "paired_cluster_bootstrap_draws": 20000,
            "bootstrap_seed": 12012,
            "rollout_level_intervals": "secondary descriptive only",
        },
        "censoring": {
            "all_training_and_rollout_failures_retained": True,
            "survivor_conditioned_errors_explicitly_labelled": True,
            "resume":
                "atomic artifacts after each checkpoint/rollout; matching "
                "completed keys are skipped under the same protocol hash",
        },
        "acceptance": {
            "artifact_gate": [
                "all twenty newly trained checkpoints retained and hashed",
                "all 528 registered rollouts attempted and retained",
                "completed final-time error below 1e-10",
                "every accepted endpoint residual strictly below 1e-12",
                "invariant monitor symmetry relative error below 1e-12",
                "all frozen source and checkpoint hashes unchanged",
                "all failures reported",
            ],
            "scientific_success_not_required": True,
            "claim_rule": "report every preregistered arm regardless of sign",
        },
        "claim_limits": [
            "one learned-H checkpoint is frozen and shared across all monitor seeds",
            "the learned-H interaction does not represent learned-H population uncertainty",
            "five monitor checkpoints per H-by-architecture arm are a small cluster sample",
            "one training orbit and one energy shell",
            "planar CPU Kepler scope",
            "survivor-conditioned errors are secondary and cannot replace completion",
        ],
        "expected_runtime": {
            "estimate": "approximately 1-2 hours on one comparable CPU core",
            "basis":
                "AC9 training took about 26 minutes for ten exact-H checkpoints; "
                "AC12 doubles checkpoints and adds learned-H network cost, while "
                "prior 264-rollout solver factorial evaluation took about 14 minutes",
            "long_training_or_evaluation_not_started_by_registration": True,
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
