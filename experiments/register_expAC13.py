"""Seal the AC13 cross-Hamiltonian scale benchmark before execution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC13", "preregistration.json")


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
    source_paths = [
        "experiments/expAC13_cross_hamiltonian_scale.py",
        "experiments/analyze_expAC13.py",
        "src/implicit_midpoint_solvers.py",
        "src/models/pasnet.py",
        "src/systems.py",
    ]
    seeds = [
        {
            "checkpoint_seed": seed,
            "sampling_seed": seed + 10000,
            "validation_seed": seed + 20000,
            "raw_initialization_seed": seed + 30000,
            "reversal_even_initialization_seed": seed + 40000,
        }
        for seed in [9101, 9102, 9103]
    ]
    shared_training = {
        "architectures": ["raw", "reversal_even"],
        "seed_manifest": seeds,
        "epochs": 300,
        "optimizer": "Adam",
        "learning_rate": 0.0002,
        "scheduler": "CosineAnnealingLR",
        "gradient_clip_norm": 1.0,
        "rollout_steps": 40,
        "unrolled_fixed_point_iterations": 4,
        "checkpoint_physical_dt": 0.1,
        "weight_energy_variance": 0.1,
        "weight_checkpoint": 5.0,
        "weight_inverse_monitor": 0.02,
        "validation_starts": 4,
        "validation_every_epochs": 25,
        "checkpoint_selection": "minimum validation state MSE",
        "reference_method": "resolved velocity Verlet",
        "reference_physical_dt": 0.001,
        "reference_horizon": 40.0,
        "reference_sample_dt": 0.01,
        "dtype": "float64",
        "torch_threads": 1,
        "no_result_conditioned_tuning": True,
    }
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC13_cross_hamiltonian_scale",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_training_or_evaluation",
        "question":
            "Does momentum-reversal-even monitor conditioning transfer across "
            "energy shells and initial conditions for a 32D modal oscillator "
            "and a one-dimensional Duffing Hamiltonian?",
        "hypotheses": {
            "architecture":
                "Momentum-reversal-even features may improve independently "
                "trained checkpoint completion relative to raw state.",
            "scale":
                "The N=16 modal benchmark tests scale beyond the historical "
                "N=8 gate.",
            "cross_system_limit":
                "System results are descriptive across Hamiltonians and are "
                "not pooled into a universal effect.",
        },
        "frozen_sources": [
            {"path": path, "sha256": sha256_file(os.path.join(ROOT, path))}
            for path in source_paths
        ],
        "systems": {
            "modal": {
                "class": "ModalOscillator",
                "parameters": {
                    "n_modes": 16,
                    "omega_min": 1.0,
                    "kappa": 10.0,
                    "beta": 0.25,
                    "seed": 0,
                    "orthogonal_mixing": True,
                },
                "state_dimension": 32,
                "hamiltonian": "exact H",
                "training_energy": 1.0,
                "training_initial_condition_seed": 0,
                "training_dtau": 0.05,
            },
            "duffing": {
                "class": "Duffing1D",
                "parameters": {"beta": 0.1},
                "state_dimension": 2,
                "hamiltonian": "exact H",
                "training_energy": 1.0,
                "training_initial_condition_seed": 0,
                "training_dtau": 0.1,
            },
        },
        "training": {
            **shared_training,
            "task":
                "end-to-end differentiation through exact-H transformed "
                "implicit-midpoint rollouts, adapted from AC9",
            "shared_network": {
                "hidden_width": 64,
                "hidden_layers": 2,
                "activation": "tanh",
                "output": "2*sigmoid(logit)",
                "cold_start_final_bias": -2.2,
            },
            "features": {
                "modal_raw": "concat(u,v)",
                "modal_reversal_even": "concat(u,v^2)",
                "duffing_raw": "concat(x,p)",
                "duffing_reversal_even": "concat(x,p^2)",
            },
            "symmetry_claim":
                "reversal-even inputs enforce momentum-reversal evenness "
                "only; no global invariance is claimed",
            "registered_training_rollouts": 3600,
            "registered_validation_rollouts": 576,
            "total_registered_training_and_validation_rollouts": 4176,
        },
        "evaluation": {
            "solver": "fixed_point",
            "newton_included": False,
            "solver_scope":
                "Corrected residual-safe fixed point only; Newton is outside "
                "AC13 scope.",
            "fixed_point": {
                "tolerance": 1e-12,
                "maximum_iterations": 100,
                "acceptance_uses_returned_endpoint_residual": True,
            },
            "modal": {
                "energy_shells": [0.75, 1.0, 1.25],
                "initial_condition_seeds": [0, 1],
                "kappas": [10.0, 20.0],
                "momentum_directions": ["forward", "reversed"],
                "slow_periods": 5,
                "dtau": 0.05,
                "dtau_basis":
                    "corrected expAC5 FPU_like_modal gate: 0.05 converged "
                    "with returned historical residual 9.99e-13",
                "resolved_verlet_dt": 0.001,
                "scenarios": 24,
            },
            "duffing": {
                "energy_shells": [0.75, 1.0, 1.25],
                "initial_condition_seeds": [0, 1],
                "momentum_directions": ["forward", "reversed"],
                "physical_horizon": 20.0,
                "dtau": 0.1,
                "dtau_basis":
                    "corrected expAC5 Duffing gate: 0.1 converged with "
                    "returned historical residual 9.92e-13",
                "resolved_verlet_dt": 0.001,
                "scenarios": 12,
            },
            "controls": [
                "prescribed_monitor_from_corrected_expAC5",
                "resolved_velocity_verlet",
            ],
            "learned_rollouts": 216,
            "control_rollouts": 72,
            "total_registered_rollouts": 288,
            "maximum_steps": 100000,
            "matched_final_physical_time": True,
            "completion_first": True,
            "dtype": "float64",
            "torch_threads": 1,
        },
        "primary_estimands": [
            "per-checkpoint completion by system and architecture",
            "reversal-even minus raw completion with n=3 checkpoint-cluster uncertainty",
            "shell, initial-condition, and momentum-reversal transfer",
            "modal final and maximum checkpoint state error, not energy alone",
            "steps, vector-field evaluations, nonlinear iterations, and wall time",
        ],
        "censoring": {
            "all_training_and_rollout_failures_retained": True,
            "completed_metrics_are_survivor_conditioned": True,
            "checkpoint_is_primary_statistical_cluster": True,
        },
        "acceptance": {
            "artifact_gate": [
                "all twelve independently initialized checkpoints retained",
                "all 288 registered evaluation rollouts attempted and retained",
                "completed final-time error below 1e-10",
                "completed returned endpoint residual below 1e-12",
                "reversal-even monitor reversal relative error below 1e-12",
                "all frozen source hashes unchanged",
                "all failures reported",
            ],
            "scientific_success_not_required": True,
            "claim_rule":
                "report every system, architecture, shell, IC, reversal, and "
                "modal-kappa stratum regardless of sign",
        },
        "claim_limits": [
            "exact-H monitor benchmark; no learned Hamiltonian",
            "three checkpoints per architecture and system",
            "momentum-reversal evenness only, not global invariance",
            "fixed-point evaluation only; Newton not included",
            "cross-system comparison descriptive only; no pooled universal claim",
        ],
        "runtime_estimate": {
            "basis":
                "AC5 CPU timings scaled by 288 evaluation rollouts plus 4176 "
                "short differentiable training/validation rollouts",
            "cpu_single_thread_hours_low": 8,
            "cpu_single_thread_hours_high": 24,
            "not_empirically_timed": True,
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
