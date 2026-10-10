"""Register the AC9 end-to-end monitor generalisation experiment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC9", "preregistration.json")


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
        "experiments/expAC9_end_to_end_monitor_generalisation.py",
        "src/models/pasnet.py",
        "src/systems.py",
    ]
    seed_manifest = [
        {
            "checkpoint_seed": seed,
            "sampling_seed": seed + 10000,
            "validation_seed": seed + 20000,
            "raw_initialization_seed": seed + 30000,
            "invariant_initialization_seed": seed + 40000,
        }
        for seed in [7101, 7102, 7103, 7104, 7105]
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC9_end_to_end_monitor_generalisation",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_training_or_evaluation",
        "question":
            "Do independently rollout-trained symmetry-preserving monitors "
            "generalise better than raw-state monitors across initial "
            "conditions and Kepler energy shells under exact H?",
        "hypotheses": {
            "architecture":
                "Invariant inputs improve checkpoint-level completion and "
                "remove orientation/momentum-reversal variation.",
            "boundary":
                "Invariant inputs need not eliminate high-eccentricity or "
                "unseen-energy-shell failures.",
            "positive_control":
                "Manual g=r completes all registered exact-H controls.",
        },
        "frozen_sources": [
            {
                "path": path,
                "sha256": sha256_file(os.path.join(ROOT, path)),
            }
            for path in source_paths
        ],
        "training": {
            "hamiltonian": "exact Kepler H; no learned H network",
            "training_orbit": {
                "semi_major_axis": 1.0,
                "eccentricity": 0.9,
                "direction": "forward",
                "orientation_radians": 0.0,
            },
            "task":
                "differentiate through unrolled exact-H implicit-midpoint "
                "rollouts using the historical Stage-B checkpoint, energy-"
                "variance, and inverse-monitor objective",
            "architectures": ["raw", "invariant"],
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
            "seed_manifest": seed_manifest,
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
            "hamiltonian": "exact Kepler H",
            "semi_major_axes": [0.75, 1.0, 1.25],
            "energy_shells": [
                -0.6666666666666666,
                -0.5,
                -0.4,
            ],
            "eccentricities": [0.7, 0.9, 0.99],
            "initial_mean_anomalies": [
                3.141592653589793,
                1.5707963267948966,
            ],
            "orientations_radians": [0.0, 1.5707963267948966],
            "momentum_directions": ["forward", "reversed"],
            "periods": 1,
            "dtau": 0.04,
            "solver": "fixed_point",
            "fixed_point": {
                "tolerance": 1e-12,
                "maximum_iterations": 100,
            },
            "maximum_steps": 100000,
            "learned_rollouts": 720,
            "manual_control_rollouts": 72,
            "total_registered_rollouts": 792,
            "matched_final_physical_time": True,
            "dtype": "float64",
            "torch_threads": 1,
        },
        "primary_estimands": [
            "per-checkpoint completion fraction by architecture",
            "checkpoint-cluster bootstrap invariant-minus-raw completion",
            "same-energy versus unseen-energy completion",
            "eccentricity-stratified completion",
            "symmetry-transform completion and error variation",
        ],
        "secondary_estimands": [
            "completed-pair energy, closure, cost, and wall-time ratios",
            "rollout-level Wilson completion intervals",
            "failure reason counts",
        ],
        "censoring": {
            "all_training_and_rollout_failures_retained": True,
            "completed-pair_metrics_labelled_survivor_conditioned": True,
            "checkpoint_is_primary_statistical_cluster": True,
        },
        "acceptance": {
            "artifact_gate": [
                "all ten independently initialized checkpoints retained",
                "all 792 registered rollouts attempted and retained",
                "completed final-time error below 1e-10",
                "completed nonlinear residual below 1e-11",
                "invariant monitor symmetry relative error below 1e-12",
                "frozen source hashes unchanged",
                "all failures reported",
            ],
            "scientific_success_not_required": True,
            "claim_rule":
                "report all architecture, shell, eccentricity, and "
                "checkpoint strata regardless of sign",
        },
        "claim_limits": [
            "exact-H monitor study, not learned-H interaction",
            "one training orbit and one training energy shell",
            "five checkpoints per architecture",
            "fixed-point evaluation only; AC8 supplies solver factorial",
            "CPU low-dimensional Kepler scope",
        ],
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
