"""Register the AC10 fair exact-H accuracy--cost comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC10", "preregistration.json")


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
        "experiments/expAC10_fair_accuracy_cost.py",
        "experiments/expAC2_corrected_long.py",
        "experiments/expAC9_end_to_end_monitor_generalisation.py",
        "src/kepler_integrators.py",
        "src/systems.py",
    ]
    checkpoint_paths = [
        f"results/expAC9/monitor_{architecture}_seed{seed}.pt"
        for architecture in ["raw", "invariant"]
        for seed in [7101, 7102, 7103, 7104, 7105]
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC10_fair_accuracy_cost",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_ac10_rollout",
        "question":
            "What accuracy--cost trade-offs remain when direct, adaptive, "
            "manual-Sundman, and independently trained learned monitors use "
            "the same exact H, scenarios, dtype, hardware, horizon, and "
            "nonlinear tolerance?",
        "frozen_sources": [
            {
                "path": path,
                "sha256": sha256_file(os.path.join(ROOT, path)),
                "role": "source",
            }
            for path in source_paths
        ]
        + [
            {
                "path": path,
                "sha256": sha256_file(os.path.join(ROOT, path)),
                "role": "checkpoint",
            }
            for path in checkpoint_paths
        ],
        "evaluation": {
            "semi_major_axes": [0.75, 1.0, 1.25],
            "eccentricities": [0.7, 0.9, 0.99],
            "initial_mean_anomalies": [
                3.141592653589793,
                1.5707963267948966,
            ],
            "orientations_radians": [0.0],
            "momentum_directions": ["forward"],
            "scenarios_per_configuration": 18,
            "periods": 1,
            "hamiltonian": "exact Kepler H",
            "dtype": "float64",
            "hardware": "same CPU host; one Torch thread",
            "matched_final_physical_time": True,
            "fixed_point": {
                "tolerance": 1e-12,
                "maximum_iterations": 100,
            },
            "maximum_steps": 100000,
            "total_configurations": 38,
            "total_registered_rollouts": 684,
        },
        "methods": {
            "direct_implicit_midpoint": {
                "dt": [0.02, 0.01, 0.005],
                "configurations": 3,
            },
            "adaptive_implicit_midpoint": {
                "rtol": [1e-5, 1e-7],
                "atol": 1e-12,
                "step_doubling": True,
                "configurations": 2,
            },
            "manual_sundman": {
                "monitor": "g=r",
                "dtau": [0.08, 0.04, 0.02],
                "configurations": 3,
            },
            "learned_monitors": {
                "architectures": ["raw", "invariant"],
                "checkpoint_seeds": [7101, 7102, 7103, 7104, 7105],
                "dtau": [0.08, 0.04, 0.02],
                "configurations": 30,
            },
        },
        "recorded_outcomes": [
            "completion status and failure reason",
            "maximum energy error",
            "one-period closure error",
            "maximum angular-momentum error",
            "steps and rejected steps",
            "Hamiltonian/vector-field evaluations",
            "nonlinear iterations and maximum residual",
            "method-only wall time",
            "configuration-level maximum RSS",
        ],
        "memory_protocol":
            "one fresh subprocess per method configuration; each subprocess "
            "evaluates the same 18 scenarios; Linux ru_maxrss is assigned "
            "as configuration-level peak, not per-rollout allocation",
        "primary_estimands": [
            "completion-aware configuration summaries",
            "robust Pareto front restricted to 18/18-complete configurations",
            "energy error versus vector-field evaluations",
            "closure error versus vector-field evaluations",
            "closure error versus wall time",
            "completion versus configuration peak RSS",
        ],
        "aggregation": {
            "controls": "18 scenario units per configuration",
            "learned_architecture_dtau":
                "90 units: five checkpoint clusters by 18 scenarios",
            "checkpoint_cluster_bootstrap_draws": 20000,
            "bootstrap_seed": 10110,
            "error_aggregation": "geometric mean over positive finite values",
            "cost_aggregation": "geometric mean",
            "pareto_completion_rule":
                "only configurations complete on all 18 scenarios enter "
                "the robust configuration-level front",
        },
        "censoring": {
            "all_failures_retained": True,
            "failed configurations not silently dropped": True,
            "completed-only error summaries labelled survivor-conditioned":
                True,
        },
        "acceptance": {
            "artifact_gate": [
                "all 38 configurations attempted in fresh subprocesses",
                "all 684 rollouts attempted and retained",
                "completed final-time error below 1e-10",
                "completed nonlinear residual below 1e-11",
                "configuration-level peak RSS recorded",
                "frozen source and checkpoint hashes unchanged",
                "all failures reported",
            ],
            "scientific_success_not_required": True,
            "claim_rule":
                "no method with incomplete registered scenarios may be "
                "called robustly Pareto-superior",
        },
        "claim_limits": [
            "exact-H one-period CPU Kepler comparison",
            "adaptive step doubling is not globally symplectic",
            "wall time is implementation- and host-specific",
            "maximum RSS is configuration-level",
            "learned methods have five checkpoints; controls are deterministic",
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
