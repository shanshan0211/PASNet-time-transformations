"""Analyze the prospectively sealed AC14 correction experiment."""

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
OUT = os.path.join(ROOT, "results", "expAC14")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(OUT, "conditioning_association.pdf")
FIGURE_PNG = os.path.join(OUT, "conditioning_association.png")
METRICS = ("q", "sigma_min", "condition_2")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(record, sample):
    corrected = sample["corrected_conditioning"]
    return {
        "rollout_key": record["rollout_key"],
        "outcome": (
            "completed"
            if record["eventual_completion_status"] == "complete"
            else "failed"
        ),
        "q": corrected["contraction_q"],
        "sigma_min": corrected["scaled_endpoint_jacobian_sigma_min"],
        "condition_2": corrected["scaled_endpoint_jacobian_condition_2"],
    }


def accepted_stride_samples(record, maximum_step=None):
    return [
        sample
        for sample in record["samples"]
        if sample["sample_kind"] == "accepted_stride"
        and "corrected_conditioning" in sample
        and (
            maximum_step is None
            or sample["accepted_physical_step"] <= maximum_step
        )
    ]


def exact_step_rows(records, step):
    rows = []
    for record in records:
        selected = [
            sample
            for sample in accepted_stride_samples(record)
            if sample["accepted_physical_step"] == step
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"{record['rollout_key']} has {len(selected)} samples at {step}"
            )
        rows.append(metric_row(record, selected[0]))
    return rows


def window_rows(records, step):
    rows = []
    for record in records:
        selected = accepted_stride_samples(record, step)
        expected = step // 10
        if len(selected) != expected:
            raise RuntimeError(
                f"{record['rollout_key']} has {len(selected)}/{expected} "
                f"accepted samples through step {step}"
            )
        values = [metric_row(record, sample) for sample in selected]
        rows.append(
            {
                "rollout_key": record["rollout_key"],
                "outcome": values[0]["outcome"],
                "q": max(row["q"] for row in values),
                "sigma_min": min(row["sigma_min"] for row in values),
                "condition_2": max(row["condition_2"] for row in values),
            }
        )
    return rows


def terminal_rows(records):
    rows = []
    for record in records:
        if record["eventual_completion_status"] == "complete":
            selected = [
                sample
                for sample in record["samples"]
                if sample["sample_kind"].startswith("accepted_")
                and "corrected_conditioning" in sample
            ]
            sample = selected[-1]
        else:
            selected = [
                sample
                for sample in record["samples"]
                if sample["sample_kind"] == "failed_step_final_iterate"
                and "corrected_conditioning" in sample
            ]
            if len(selected) != 1:
                raise RuntimeError("failed rollout lacks one terminal iterate")
            sample = selected[0]
        rows.append(metric_row(record, sample))
    return rows


def summarize(rows):
    output = {"n_rollouts": len(rows), "outcomes": {}, "contrasts": {}}
    for outcome in ("completed", "failed"):
        selected = [row for row in rows if row["outcome"] == outcome]
        output["outcomes"][outcome] = {
            "n": len(selected),
            **{
                metric: {
                    "median": float(np.median([row[metric] for row in selected])),
                    "q25": float(np.quantile(
                        [row[metric] for row in selected], 0.25
                    )),
                    "q75": float(np.quantile(
                        [row[metric] for row in selected], 0.75
                    )),
                }
                for metric in METRICS
            },
        }
    for metric in METRICS:
        completed = [
            row[metric] for row in rows if row["outcome"] == "completed"
        ]
        failed = [row[metric] for row in rows if row["outcome"] == "failed"]
        output["contrasts"][metric] = {
            "estimand": "failed minus completed rollout-level median",
            "value": float(np.median(failed) - np.median(completed)),
        }
    return output


def residual_summary(records):
    stopping = []
    returned = []
    paired_ratios = []
    for record in records:
        for sample in record["samples"]:
            update = sample.get("stopping_update_inf")
            residual = sample.get("returned_endpoint_residual_inf")
            if update is not None and math.isfinite(update):
                stopping.append(update)
            if residual is not None and math.isfinite(residual):
                returned.append(residual)
            if (
                update is not None
                and residual is not None
                and math.isfinite(update)
                and math.isfinite(residual)
                and update > 0.0
            ):
                paired_ratios.append(residual / update)
    return {
        "stopping_update_inf": {
            "n": len(stopping),
            "median": float(np.median(stopping)),
        },
        "returned_endpoint_residual_inf": {
            "n": len(returned),
            "median": float(np.median(returned)),
        },
        "secondary_paired_residual_over_update": {
            "n": len(paired_ratios),
            "median": float(np.median(paired_ratios)),
        },
    }


def make_figure(rows, step):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    labels = {
        "q": "Corrected contraction $q$",
        "sigma_min": "Corrected endpoint $\\sigma_{\\min}$",
        "condition_2": "Corrected endpoint condition number",
    }
    for axis, metric in zip(axes, METRICS):
        groups = [
            [row[metric] for row in rows if row["outcome"] == outcome]
            for outcome in ("completed", "failed")
        ]
        axis.boxplot(groups, tick_labels=["Completed", "Failed"], showfliers=False)
        if all(value > 0.0 for group in groups for value in group):
            axis.set_yscale("log")
        axis.set_ylabel(labels[metric])
        axis.grid(axis="y", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        f"Eventual outcome association at accepted step {step} "
        "(one observation per rollout)"
    )
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
    expected = protocol["evaluation"]["expected_rollouts"]
    if len(records) != expected or len(
        {record["rollout_key"] for record in records}
    ) != expected:
        raise RuntimeError("AC14 rollout join is incomplete or duplicated")
    landmarks = protocol["analysis"]["accepted_step_landmarks"]
    matched_step = {
        str(step): summarize(exact_step_rows(records, step))
        for step in landmarks
    }
    matched_window = {
        str(step): summarize(window_rows(records, step))
        for step in landmarks
    }
    terminal = summarize(terminal_rows(records))
    primary_step = protocol["analysis"]["primary_figure_step"]
    primary_rows = exact_step_rows(records, primary_step)
    analysis = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "source_sha256": {
            "raw_results": sha256_file(RAW_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "rollout_counts": {
            "total": len(records),
            "by_eventual_status": dict(Counter(
                record["eventual_completion_status"] for record in records
            )),
        },
        "matched_step_accepted_only": matched_step,
        "matched_window_accepted_only": matched_window,
        "terminal_iterate_sensitivity_separate": terminal,
        "solver_quantities_reported_separately": residual_summary(records),
        "figure_estimand": {
            "accepted_step": primary_step,
            "observations": len(primary_rows),
            "observation_unit": "one exact-step observation per rollout",
            "terminal_failed_iterates_included": False,
        },
        "interpretation_limits": [
            "All completion comparisons are associations, not causal effects.",
            "Matched-step and matched-window analyses use accepted iterates only.",
            "Terminal failed iterates appear only in the separately labeled "
            "terminal sensitivity analysis.",
            "Eventual-outcome groups are observational and not randomized.",
            "AC11 remains invalidated diagnostic history and is not overwritten.",
        ],
    }
    temporary = ANALYSIS_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, ANALYSIS_PATH)
    make_figure(primary_rows, primary_step)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
