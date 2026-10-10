"""Checkpoint-level analysis for sealed AC17."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC17")
AUDIT_PATH = os.path.join(OUT, "hamiltonian_audit.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def completion(records):
    complete = sum(
        record["completion_status"] == "complete" for record in records
    )
    return {
        "complete": complete,
        "registered": len(records),
        "fraction": complete / len(records) if records else None,
    }


def h_level_estimands(records, n_draws=20000):
    h_seeds = sorted({record["h_seed"] for record in records})
    architectures = ["raw", "invariant"]
    solvers = ["fixed_point", "monotone_residual_newton"]
    per_h = {}
    nested_monitor = {}
    for h_seed in h_seeds:
        cell = {}
        for architecture in architectures:
            for solver in solvers:
                selected = [
                    record for record in records
                    if record["h_seed"] == h_seed
                    and record["architecture"] == architecture
                    and record["solver"] == solver
                ]
                fractions = {}
                for replicate in sorted({
                    record["monitor_replicate"] for record in selected
                }):
                    replicate_records = [
                        record for record in selected
                        if record["monitor_replicate"] == replicate
                    ]
                    fractions[str(replicate)] = completion(
                        replicate_records
                    )["fraction"]
                nested_monitor[f"{h_seed}|{architecture}|{solver}"] = fractions
                values = [value for value in fractions.values() if value is not None]
                cell[(architecture, solver)] = (
                    float(np.mean(values)) if values else np.nan
                )
        architecture_contrast = np.mean([
            cell[("invariant", solver)] - cell[("raw", solver)]
            for solver in solvers
        ])
        solver_contrast = np.mean([
            cell[(architecture, "monotone_residual_newton")]
            - cell[(architecture, "fixed_point")]
            for architecture in architectures
        ])
        interaction = (
            cell[("invariant", "monotone_residual_newton")]
            - cell[("raw", "monotone_residual_newton")]
            - cell[("invariant", "fixed_point")]
            + cell[("raw", "fixed_point")]
        )
        per_h[h_seed] = {
            "invariant_minus_raw": float(architecture_contrast),
            "newton_minus_fixed_point": float(solver_contrast),
            "architecture_by_solver": float(interaction),
        }
    rng = np.random.default_rng(17017)
    estimates = {}
    for name in [
        "invariant_minus_raw",
        "newton_minus_fixed_point",
        "architecture_by_solver",
    ]:
        values = np.asarray([per_h[seed][name] for seed in h_seeds])
        if len(values):
            draws = np.asarray([
                np.mean(values[rng.integers(0, len(values), len(values))])
                for _ in range(n_draws)
            ])
            interval = [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ]
            mean = float(np.mean(values))
            standard_deviation = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        else:
            interval = [None, None]
            mean = None
            standard_deviation = None
        estimates[name] = {
            "mean_across_h_checkpoints": mean,
            "h_checkpoint_standard_deviation": standard_deviation,
            "h_checkpoint_bootstrap_95": interval,
            "per_h_checkpoint": {
                str(seed): per_h[seed][name] for seed in h_seeds
            },
        }
    return {
        "primary_cluster": "independently trained accepted H checkpoint",
        "n_h_checkpoints": len(h_seeds),
        "monitor_replicates_are_nested_not_independent_h_replicates": True,
        "bootstrap_draws": n_draws,
        "bootstrap_seed": 17017,
        "nested_monitor_completion": nested_monitor,
        "estimands": estimates,
    }


def fidelity_uncertainty(audit):
    accepted = [record for record in audit["records"] if record["accepted"]]
    rejected = [record for record in audit["records"] if not record["accepted"]]
    fields = ["normalized_rmse", "relative_error_p95", "relative_error_max"]
    return {
        "accepted_checkpoint_metrics": {
            str(record["h_seed"]): {
                kind: {
                    field: record[kind][field] for field in fields
                }
                for kind in ["hamiltonian_gradient", "vector_field"]
            }
            for record in accepted
        },
        "rejected_checkpoints": {
            str(record["h_seed"]): {
                "rejection_reasons": record["rejection_reasons"],
                "metrics": {
                    kind: record.get(kind)
                    for kind in ["hamiltonian_gradient", "vector_field"]
                },
                "error": record.get("error"),
            }
            for record in rejected
        },
    }


def main():
    audit = load(AUDIT_PATH)
    training = load(TRAINING_PATH)
    raw = load(RAW_PATH)
    summary = load(SUMMARY_PATH)
    records = raw["records"]
    analysis = {
        "schema_version": "1.0",
        "experiment": raw["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": raw["protocol_sha256"],
        "source_sha256": {
            "hamiltonian_audit": sha256_file(AUDIT_PATH),
            "training": sha256_file(TRAINING_PATH),
            "raw": sha256_file(RAW_PATH),
            "summary": sha256_file(SUMMARY_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "gate_decision": {
            "ensemble_qualified": audit["ensemble_qualified"],
            "decision": audit["decision"],
            "accepted_h_seeds": [
                record["h_seed"] for record in audit["records"]
                if record["accepted"]
            ],
        },
        "fidelity_checkpoint_uncertainty": fidelity_uncertainty(audit),
        "factorial_checkpoint_analysis": (
            h_level_estimands(records)
            if audit["ensemble_qualified"]
            else {
                "status": "not_run_no_qualified_h_ensemble",
                "primary_cluster":
                    "independently trained accepted H checkpoint",
            }
        ),
        "failure_reasons": dict(Counter(
            record["completion_status"] for record in records
            if record["completion_status"] != "complete"
        )),
        "training_failures": [
            record for record in training["records"]
            if record["status"] == "training_failure"
        ],
        "artifact_gate": summary["acceptance"],
        "interpretation_limits": [
            "Inference resamples H checkpoints, never individual rollouts.",
            "Monitor replicates are nested within H checkpoint and architecture.",
            "Rejected H checkpoints remain visible but do not enter the factorial.",
            "If fewer than three H checkpoints pass, no monitor result is produced.",
            "Completed error and cost summaries are not primary estimands.",
        ],
    }
    atomic_json(ANALYSIS_PATH, analysis)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
