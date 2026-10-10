"""Analyze the sealed AC11 conditioning replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC11")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(ROOT, "fig_conditioning_audit.pdf")
FIGURE_PNG = os.path.join(ROOT, "fig_conditioning_audit.png")


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


def finite(values):
    return np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )


def describe(values):
    values = finite(values)
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def sample_rows(records):
    rows = []
    for record in records:
        outcome = (
            "completed_accepted"
            if record["replay_completion_status"] == "complete"
            else "failed_rollout_sample"
        )
        for sample in record["samples"]:
            conditioning = sample.get("conditioning")
            if conditioning is None:
                continue
            solver = sample.get("solver", {})
            stopping = solver.get("stopping_update_inf")
            residual = solver.get("returned_endpoint_residual_inf")
            ratio = None
            if (
                stopping is not None
                and residual is not None
                and stopping > 0.0
            ):
                ratio = residual / stopping
            rows.append(
                {
                    "rollout_key": record["rollout_key"],
                    "arm": record["arm"],
                    "architecture": record["architecture"],
                    "checkpoint_seed": record["checkpoint_seed"],
                    "solver": record["solver"],
                    "semi_major_axis": record["semi_major_axis"],
                    "eccentricity": record["eccentricity"],
                    "rollout_outcome": outcome,
                    "sample_kind": sample["sample_kind"],
                    "q": conditioning["contraction_q"],
                    "sigma_min": conditioning[
                        "scaled_endpoint_jacobian_sigma_min"
                    ],
                    "condition_2": conditioning[
                        "scaled_endpoint_jacobian_condition_2"
                    ],
                    "off_surface_defect_abs": conditioning[
                        "off_surface_defect_abs"
                    ],
                    "returned_residual_over_stopping_update": ratio,
                }
            )
    return rows


METRICS = [
    "q",
    "sigma_min",
    "condition_2",
    "off_surface_defect_abs",
    "returned_residual_over_stopping_update",
]


def stratified(rows):
    dimensions = [
        ("arm",),
        ("arm", "architecture"),
        ("arm", "solver"),
        ("arm", "architecture", "solver"),
        ("arm", "eccentricity"),
        ("arm", "semi_major_axis"),
    ]
    output = {}
    for fields in dimensions:
        groups = {}
        for row in rows:
            key = "|".join(str(row[field]) for field in fields)
            groups.setdefault(key, []).append(row)
        output["+".join(fields)] = {
            key: {
                outcome: {
                    metric: describe(
                        [
                            row[metric]
                            for row in selected
                            if row["rollout_outcome"] == outcome
                        ]
                    )
                    for metric in METRICS
                }
                for outcome in ["completed_accepted", "failed_rollout_sample"]
            }
            for key, selected in sorted(groups.items())
        }
    return output


def rollout_summaries(records, rows):
    by_rollout = {}
    for row in rows:
        by_rollout.setdefault(row["rollout_key"], []).append(row)
    summaries = []
    for record in records:
        selected = by_rollout.get(record["rollout_key"], [])
        if not selected:
            continue
        summaries.append(
            {
                "rollout_key": record["rollout_key"],
                "arm": record["arm"],
                "architecture": record["architecture"],
                "checkpoint_seed": record["checkpoint_seed"],
                "solver": record["solver"],
                "complete": record["replay_completion_status"] == "complete",
                "q": float(np.max([row["q"] for row in selected])),
                "sigma_min": float(
                    np.min([row["sigma_min"] for row in selected])
                ),
                "condition_2": float(
                    np.max([row["condition_2"] for row in selected])
                ),
            }
        )
    return summaries


def cluster_bootstrap(summaries, draws=20000, seed=11111):
    """Bootstrap checkpoint clusters; deterministic controls stay descriptive."""
    output = {}
    rng = np.random.default_rng(seed)
    groups = {}
    for row in summaries:
        if row["checkpoint_seed"] is None:
            continue
        key = (row["arm"], row["architecture"], row["solver"])
        groups.setdefault(key, []).append(row)
    for key, selected in sorted(groups.items()):
        clusters = sorted({row["checkpoint_seed"] for row in selected})
        label = "|".join(key)
        result = {
            "cluster_unit": "checkpoint_seed",
            "n_clusters": len(clusters),
            "draws_requested": draws,
            "metrics": {},
        }
        for metric in ["q", "sigma_min", "condition_2"]:
            observed_complete = [
                row[metric] for row in selected if row["complete"]
            ]
            observed_failed = [
                row[metric] for row in selected if not row["complete"]
            ]
            observed = (
                float(np.median(observed_failed) - np.median(observed_complete))
                if observed_complete and observed_failed
                else None
            )
            bootstrap = []
            if len(clusters) >= 2 and observed is not None:
                cluster_rows = {
                    cluster: [
                        row
                        for row in selected
                        if row["checkpoint_seed"] == cluster
                    ]
                    for cluster in clusters
                }
                for _ in range(draws):
                    sampled = rng.choice(clusters, len(clusters), replace=True)
                    replayed = [
                        row
                        for cluster in sampled
                        for row in cluster_rows[cluster]
                    ]
                    complete = [
                        row[metric] for row in replayed if row["complete"]
                    ]
                    failed = [
                        row[metric] for row in replayed if not row["complete"]
                    ]
                    if complete and failed:
                        bootstrap.append(
                            float(np.median(failed) - np.median(complete))
                        )
            result["metrics"][metric] = {
                "estimand":
                    "failed minus completed median of rollout-level sampled "
                    "extreme",
                "observed": observed,
                "valid_bootstrap_draws": len(bootstrap),
                "checkpoint_cluster_bootstrap_95": (
                    [
                        float(np.quantile(bootstrap, 0.025)),
                        float(np.quantile(bootstrap, 0.975)),
                    ]
                    if bootstrap
                    else None
                ),
            }
        output[label] = result
    return output


def completion_join(records, protocol):
    expected = protocol["evaluation"]["total_registered_rollouts"]
    mismatches = [
        record["rollout_key"]
        for record in records
        if not record["completion_label_match"]
    ]
    return {
        "expected": expected,
        "joined": len(records),
        "unique_keys": len({record["rollout_key"] for record in records}),
        "parent_statuses": dict(
            Counter(record["parent_completion_status"] for record in records)
        ),
        "replay_statuses": dict(
            Counter(record["replay_completion_status"] for record in records)
        ),
        "matching": len(records) - len(mismatches),
        "mismatching": len(mismatches),
        "mismatch_keys": mismatches,
        "complete_and_unique": (
            len(records) == expected
            and len({record["rollout_key"] for record in records}) == expected
        ),
    }


def make_figure(rows):
    completed = [
        row for row in rows if row["rollout_outcome"] == "completed_accepted"
    ]
    failed = [
        row for row in rows if row["rollout_outcome"] == "failed_rollout_sample"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    configurations = [
        ("q", "Scaled contraction $q$"),
        ("sigma_min", "Scaled endpoint $\\sigma_{\\min}$"),
        (
            "returned_residual_over_stopping_update",
            "Returned residual / stopping update",
        ),
    ]
    for axis, (metric, label) in zip(axes, configurations):
        values = [
            finite([row[metric] for row in completed]),
            finite([row[metric] for row in failed]),
        ]
        nonempty = [value for value in values if len(value)]
        if nonempty:
            axis.boxplot(
                nonempty,
                tick_labels=[
                    name
                    for name, value in zip(["Completed", "Failed"], values)
                    if len(value)
                ],
                showfliers=False,
            )
        if all(np.all(value > 0.0) for value in nonempty):
            axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_title("(a) Fixed-point proxy")
    axes[1].set_title("(b) Newton conditioning")
    axes[2].set_title("(c) Stopping semantics")
    fig.tight_layout()
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    protocol = load(PROTOCOL_PATH)
    raw = load(RAW_PATH)
    if raw["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("raw artifact protocol mismatch")
    records = raw["records"]
    join = completion_join(records, protocol)
    if not join["complete_and_unique"]:
        raise RuntimeError(
            f"AC11 replay incomplete: {join['joined']}/"
            f"{join['expected']} records"
        )
    rows = sample_rows(records)
    summaries = rollout_summaries(records, rows)
    analysis = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "source_sha256": {
            "raw": sha256_file(RAW_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "completion_label_join": join,
        "sample_counts": {
            "total_with_conditioning": len(rows),
            "by_kind": dict(Counter(row["sample_kind"] for row in rows)),
            "by_rollout_outcome": dict(
                Counter(row["rollout_outcome"] for row in rows)
            ),
        },
        "stratified_sample_summaries": stratified(rows),
        "checkpoint_cluster_bootstrap": cluster_bootstrap(summaries),
        "manual_control_uncertainty":
            "descriptive scenario summaries only; no checkpoint cluster exists",
        "failure_reasons": dict(
            Counter(
                record["failure"]["reason"]
                for record in records
                if record["failure"] is not None
            )
        ),
        "interpretation_limits": [
            "Conditioning and completion are correlated replay measurements; "
            "the analysis does not establish causation.",
            "No conditioning intervention or randomized solver intervention "
            "was performed.",
            "Accepted-step diagnostics are sampled every ten steps plus horizon "
            "trims; failures retain the final failed iterate.",
            "Within-rollout samples are dependent and are not treated as "
            "independent rollout replicates.",
            "Completed accepted-step comparisons are survivor-conditioned.",
            "The contraction proxy is sufficient locally when below one under "
            "additional mapping assumptions; it is not necessary.",
        ],
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(rows)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
