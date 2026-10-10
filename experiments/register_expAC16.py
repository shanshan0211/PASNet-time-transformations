"""Seal the AC16 correction experiment without modifying sealed AC13."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC16", "preregistration.json")


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
        "experiments/expAC16_reference_convergence.py",
        "experiments/analyze_expAC16.py",
        "src/systems.py",
    ]
    ac13_artifacts = [
        "results/expAC13/preregistration.json",
        "results/expAC13/raw_results.json",
        "results/expAC13/summary.json",
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC16_reference_convergence",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_execution",
        "correction_of": "expAC13_cross_hamiltonian_scale",
        "question":
            "Are AC13 state-error rankings robust to velocity-Verlet "
            "reference discretization error at dt=0.001?",
        "hypotheses": {
            "reference_convergence":
                "Velocity Verlet should exhibit approximately second-order "
                "state and energy convergence on every registered AC13 "
                "scenario, but all irregular or negative orders are retained.",
            "ranking_stability":
                "A published learned/prescribed ordering is called stable "
                "only when conservative reference-error intervals do not "
                "overlap; overlap is reported as unresolved.",
        },
        "frozen_sources": [
            {"path": path, "sha256": sha256_file(os.path.join(ROOT, path))}
            for path in source_paths
        ],
        "frozen_ac13_artifacts": [
            {"path": path, "sha256": sha256_file(os.path.join(ROOT, path))}
            for path in ac13_artifacts
        ],
        "systems": {
            "modal": {
                "class": "ModalOscillator",
                "parameters": {
                    "n_modes": 16,
                    "omega_min": 1.0,
                    "beta": 0.25,
                    "seed": 0,
                    "orthogonal_mixing": True,
                    "kappas": [10.0, 20.0],
                },
                "state_dimension": 32,
            },
            "duffing": {
                "class": "Duffing1D",
                "parameters": {"beta": 0.1},
                "state_dimension": 2,
            },
        },
        "reference": {
            "method": "velocity Verlet",
            "dtype": "float64",
            "dt_values": [0.002, 0.001, 0.0005],
            "ac13_dt": 0.001,
            "finest_dt": 0.0005,
            "comparison_grid":
                "each finer trajectory grid; coarser states linearly "
                "interpolated exactly as AC13 interpolated its reference",
            "state_uncertainty":
                "Richardson estimate from adjacent-level maximum and final "
                "trajectory discrepancies, with observed order capped at 4 "
                "and conservative order 0.5 for nonpositive/undefined order",
            "energy_uncertainty":
                "exact-H maximum and final drift at every level plus observed "
                "order and finest-level Richardson estimate",
            "zero_reference_points_forbidden": True,
        },
        "evaluation": {
            "modal": {
                "energy_shells": [0.75, 1.0, 1.25],
                "initial_condition_seeds": [0, 1],
                "kappas": [10.0, 20.0],
                "momentum_directions": ["forward", "reversed"],
                "slow_periods": 5,
                "scenarios": 24,
            },
            "duffing": {
                "energy_shells": [0.75, 1.0, 1.25],
                "initial_condition_seeds": [0, 1],
                "momentum_directions": ["forward", "reversed"],
                "physical_horizon": 20.0,
                "scenarios": 12,
            },
            "total_scenarios": 36,
            "all_ac13_scenarios_and_full_horizons": True,
        },
        "analysis": {
            "ac13_methods": ["learned_monitor", "prescribed_monitor"],
            "metrics": ["max_state_error", "final_state_error"],
            "record_interval":
                "[max(0, reported_error-reference_uncertainty), "
                "reported_error+reference_uncertainty]",
            "aggregate":
                "median reported error and medians of recordwise interval "
                "endpoints, stratified by system and learned architecture",
            "stable_total_order_rule":
                "every adjacent pair in the reported median ordering has "
                "strictly separated conservative intervals",
            "scenario_pair_rule":
                "learned versus prescribed direction is stable only for "
                "strictly separated intervals",
            "resolved_reference_display":
                "if displayed, finest reference uses positive Richardson "
                "uncertainty and is never plotted as exact zero",
        },
        "acceptance": {
            "artifact_gate": [
                "all 36 registered scenarios retained and complete",
                "all three dt levels retained",
                "nonzero finite state uncertainty for every scenario",
                "AC13 input artifact hashes unchanged",
                "frozen source hashes unchanged",
            ],
            "scientific_success_not_required": True,
            "observed_order_near_two_not_required": True,
            "ranking_stability_not_required": True,
        },
        "runtime_estimate": {
            "integration_steps": 3478944,
            "cpu_single_process_minutes_low": 1,
            "cpu_single_process_minutes_high": 8,
            "basis":
                "3.48 million vectorized NumPy velocity-Verlet steps over "
                "36 scenarios; excludes analysis and environment contention",
            "smoke_seconds": 2,
        },
        "claim_limits": [
            "corrects reference-discretization uncertainty only",
            "does not retrain or rerun learned/prescribed AC13 methods",
            "ranking stability follows triangle-inequality intervals and may "
            "be unresolved even when the reported medians differ",
            "Richardson uncertainty is model-based and empirical order is "
            "reported without suppressing irregular convergence",
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
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(protocol["integrity"]["protocol_sha256"])


if __name__ == "__main__":
    main()
