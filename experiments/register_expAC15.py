"""Seal the AC15 returned-endpoint-residual correction protocol."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC15")
OUTPUT = os.path.join(OUT, "preregistration.json")


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


def load(relative_path):
    with open(os.path.join(ROOT, relative_path), encoding="utf-8") as handle:
        return json.load(handle)


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")

    parent_counts = {}
    observed_seconds = {}
    for parent in range(6, 11):
        raw = load(f"results/expAC{parent}/raw_results.json")
        records = raw["records"]
        if parent == 8:
            records = [
                record
                for record in records
                if record["solver"] == "fixed_point"
            ]
        parent_counts[f"AC{parent}"] = len(records)
        observed_seconds[f"AC{parent}"] = sum(
            record.get("wall_seconds", 0.0) for record in records
        )
    expected_counts = {
        "AC6": 80,
        "AC7": 60,
        "AC8": 132,
        "AC9": 792,
        "AC10": 684,
    }
    if parent_counts != expected_counts:
        raise RuntimeError(
            f"parent replay counts changed: {parent_counts} != {expected_counts}"
        )

    parent_files = [
        f"results/expAC{parent}/{name}"
        for parent in range(6, 11)
        for name in ("preregistration.json", "raw_results.json")
    ]
    parent_files += [
        "results/expAC8/training.json",
        "results/expAC9/training.json",
        "results/expE2/pasnet_weights.pt",
    ]
    parent_files += [
        f"results/expAC8/monitor_{architecture}_seed{seed}.pt"
        for architecture in ("raw", "invariant")
        for seed in (6101, 6102, 6103, 6104, 6105)
    ]
    parent_files += [
        f"results/expAC9/monitor_{architecture}_seed{seed}.pt"
        for architecture in ("raw", "invariant")
        for seed in (7101, 7102, 7103, 7104, 7105)
    ]
    implementation_files = [
        "experiments/expAC15_returned_endpoint_residual_correction.py",
        "experiments/analyze_expAC15.py",
        "src/implicit_midpoint_solvers.py",
        "src/models/pasnet.py",
        "src/systems.py",
    ]

    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC15_returned_endpoint_residual_correction",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_ac15_replay",
        "coordination": {
            "number_selection":
                "AC15 is used to avoid collision with the conceptually "
                "adjacent AC14 workstream; no AC14 artifact is read or changed",
            "correction_only":
                "AC15 does not overwrite, relabel, or silently replace any "
                "AC6--AC10 artifact",
        },
        "problem":
            "AC6--AC10 fixed-point paths stopped on the norm of the update "
            "that produced the returned iterate, while their artifact fields "
            "and gates called that quantity a nonlinear residual",
        "acceptance_definition": {
            "equation":
                "R(z_plus)=z_plus-z-h*f((z+z_plus)/2)",
            "norm": "infinity norm",
            "rule":
                "a solve is complete only when R is freshly recomputed at "
                "the exact returned endpoint and ||R||_inf < solver tolerance",
            "strict_inequality": True,
            "update_norm_is_diagnostic_only": True,
        },
        "minimum_rerun_decision": {
            "included": {
                "AC6": "all 80 manual/learned fixed-point rollouts",
                "AC7": "all 60 refined manual/learned fixed-point rollouts",
                "AC8":
                    "132 fixed-point cells, including all raw, invariant, "
                    "and manual configurations",
                "AC9":
                    "all 792 fixed-point evaluations; frozen training and "
                    "checkpoints are reused",
                "AC10":
                    "all 684 direct, adaptive, manual, and learned "
                    "configuration-scenario evaluations",
            },
            "excluded": {
                "AC8_damped_newton":
                    "its runner already checks the defining residual at the "
                    "returned trial; rerunning 132 cells cannot remediate the "
                    "identified endpoint/update mismatch",
                "AC8_and_AC9_training":
                    "training checkpoint selection does not declare rollout "
                    "completion from the affected stopping test",
            },
            "registered_replays_by_parent": parent_counts,
            "total_registered_replays": sum(parent_counts.values()),
        },
        "solver_semantics": {
            "transformed_fixed_point": {
                "tolerance": 1e-12,
                "maximum_iterations": 100,
                "initial_guess": "explicit Euler endpoint",
            },
            "AC10_direct_and_adaptive_fixed_point": {
                "tolerance": 1e-13,
                "maximum_iterations": 100,
                "note":
                    "preserves the effective parent implementation, which "
                    "was stricter than AC10's top-level 1e-12 description",
                "initial_guess": "explicit Euler endpoint",
            },
            "steps_horizons_alignment_and_adaptive_controller":
                "copied from parent protocols/runners without tuning",
        },
        "replay_and_censoring": {
            "manifest":
                "one immutable replay id per selected parent raw record",
            "resume":
                "raw_results.json is atomically replaced after every attempt; "
                "all previously attempted replay ids are skipped",
            "failures":
                "every solver, nonfinite, step-cap, and unexpected failure is "
                "retained with the last state and structured diagnostics",
            "comparison":
                "every correction record carries its parent status and label; "
                "analysis reports transition counts and paired outcome changes",
        },
        "runtime": {
            "parent_observed_selected_wall_seconds": observed_seconds,
            "parent_observed_selected_total_wall_seconds":
                sum(observed_seconds.values()),
            "expected_single_thread_wall_minutes": [25, 45],
            "basis":
                "selected parent method-only wall time plus one fresh residual "
                "evaluation per nonlinear iteration and atomic serialization",
            "smoke_not_full":
                "the development command executes a small representative "
                "subset only; the sealed full replay is not launched here",
        },
        "frozen_sources": [
            {
                "path": path,
                "sha256": sha256_file(os.path.join(ROOT, path)),
                "role": (
                    "implementation"
                    if path in implementation_files
                    else "parent_artifact_or_checkpoint"
                ),
            }
            for path in [*parent_files, *implementation_files]
        ],
        "acceptance": {
            "all_1748_attempted_and_retained": True,
            "all_completed_returned_endpoint_residuals_below_own_tolerance":
                True,
            "all_parent_labels_linked": True,
            "all_failures_retained": True,
            "source_hashes_unchanged": True,
            "scientific_success_not_required": True,
        },
        "claim_limits": [
            "AC15 corrects completion and residual evidence only",
            "it does not retroactively change AC6--AC10 sealed labels",
            "wall-time comparisons cross implementations and are descriptive",
            "AC8 damped-Newton parent labels are not replayed",
            "training procedures and checkpoints are not rerun",
        ],
        "hardware_registration": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "integrity": {"protocol_sha256": None},
    }
    protocol["integrity"]["protocol_sha256"] = canonical_hash(protocol)
    os.makedirs(OUT, exist_ok=True)
    with open(OUTPUT, "x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(protocol["integrity"]["protocol_sha256"])


if __name__ == "__main__":
    main()
