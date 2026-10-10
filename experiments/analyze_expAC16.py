"""Analyze AC16 convergence and conservatively reassess AC13 rankings."""

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
OUT = os.path.join(ROOT, "results", "expAC16")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
AC13_RAW_PATH = os.path.join(ROOT, "results", "expAC13", "raw_results.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(OUT, "fig_reference_convergence.pdf")
FIGURE_PNG = os.path.join(OUT, "fig_reference_convergence.png")


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


def scenario_key(record):
    return (
        record["system"],
        record["energy_shell"],
        record["initial_condition_seed"],
        record.get("kappa"),
        record["momentum_direction"],
    )


def method_label(record):
    if record["method"] == "learned_monitor":
        return f"learned_{record['architecture']}"
    return "prescribed_monitor"


def uncertainty_for_metric(reference, metric):
    field = "maximum" if metric == "max_state_error" else "final"
    return reference["state_reference_uncertainty"]["ac13_dt"][field][
        "absolute"
    ]


def corrected_records(ac13_records, reference_by_scenario):
    corrected = []
    for record in ac13_records:
        if (
            record["method"] not in {"learned_monitor", "prescribed_monitor"}
            or record["completion_status"] != "complete"
        ):
            continue
        reference = reference_by_scenario[scenario_key(record)]
        entry = {
            "scenario": list(scenario_key(record)),
            "system": record["system"],
            "method": record["method"],
            "architecture": record.get("architecture"),
            "checkpoint_seed": record.get("checkpoint_seed"),
            "label": method_label(record),
            "metrics": {},
        }
        for metric in ["max_state_error", "final_state_error"]:
            value = record[metric]
            uncertainty = uncertainty_for_metric(reference, metric)
            entry["metrics"][metric] = {
                "reported": value,
                "reference_uncertainty": uncertainty,
                "lower": max(0.0, value - uncertainty),
                "upper": value + uncertainty,
            }
        corrected.append(entry)
    return corrected


def aggregate_rankings(corrected):
    result = {}
    for system_name in ["modal", "duffing"]:
        for metric in ["max_state_error", "final_state_error"]:
            groups = {}
            for label in [
                "learned_raw",
                "learned_reversal_even",
                "prescribed_monitor",
            ]:
                selected = [
                    entry["metrics"][metric]
                    for entry in corrected
                    if entry["system"] == system_name and entry["label"] == label
                ]
                groups[label] = {
                    "records": len(selected),
                    "median_reported": float(
                        np.median([value["reported"] for value in selected])
                    ),
                    "conservative_median_interval": [
                        float(np.median([value["lower"] for value in selected])),
                        float(np.median([value["upper"] for value in selected])),
                    ],
                }
            ordering = sorted(groups, key=lambda label: groups[label]["median_reported"])
            adjacent = []
            for better, worse in zip(ordering, ordering[1:]):
                separated = (
                    groups[better]["conservative_median_interval"][1]
                    < groups[worse]["conservative_median_interval"][0]
                )
                adjacent.append(
                    {
                        "reported_better": better,
                        "reported_worse": worse,
                        "strictly_separated": separated,
                    }
                )
            result[f"{system_name}|{metric}"] = {
                "groups": groups,
                "reported_order_best_to_worst": ordering,
                "adjacent_comparisons": adjacent,
                "reported_total_order_stable":
                    all(item["strictly_separated"] for item in adjacent),
            }
    return result


def learned_prescribed_pairs(corrected):
    prescribed = {}
    for entry in corrected:
        if entry["label"] == "prescribed_monitor":
            prescribed[tuple(entry["scenario"])] = entry
    result = {}
    for metric in ["max_state_error", "final_state_error"]:
        counts = Counter()
        for learned in corrected:
            if learned["method"] != "learned_monitor":
                continue
            control = prescribed[tuple(learned["scenario"])]
            left = learned["metrics"][metric]
            right = control["metrics"][metric]
            reported_direction = (
                "learned_better"
                if left["reported"] < right["reported"]
                else "prescribed_better"
                if right["reported"] < left["reported"]
                else "reported_tie"
            )
            if left["upper"] < right["lower"]:
                corrected_direction = "learned_better"
            elif right["upper"] < left["lower"]:
                corrected_direction = "prescribed_better"
            else:
                corrected_direction = "unresolved_overlap"
            counts[f"reported|{reported_direction}"] += 1
            counts[f"corrected|{corrected_direction}"] += 1
            counts["stable_direction"] += (
                reported_direction == corrected_direction
                and corrected_direction != "unresolved_overlap"
            )
            counts["registered_pairs"] += 1
        result[metric] = dict(counts)
    return result


def distribution(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "minimum": float(np.min(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def convergence_summary(records):
    result = {}
    for system_name in ["modal", "duffing"]:
        selected = [record for record in records if record["system"] == system_name]
        result[system_name] = {
            "state_observed_order": {
                field: distribution(
                    [
                        record["state_observed_order"][field]
                        for record in selected
                        if record["state_observed_order"][field] is not None
                    ]
                )
                for field in ["maximum", "final"]
            },
            "energy_observed_order": {
                field: distribution(
                    [
                        record["energy_observed_order"][field]
                        for record in selected
                        if record["energy_observed_order"][field] is not None
                    ]
                )
                for field in ["maximum_absolute_drift", "final_absolute_drift"]
            },
            "ac13_state_reference_uncertainty": {
                field: distribution(
                    [
                        record["state_reference_uncertainty"]["ac13_dt"][field][
                            "absolute"
                        ]
                        for record in selected
                    ]
                )
                for field in ["maximum", "final"]
            },
            "finest_state_reference_uncertainty": {
                field: distribution(
                    [
                        record["state_reference_uncertainty"]["finest_dt"][field][
                            "absolute"
                        ]
                        for record in selected
                    ]
                )
                for field in ["maximum", "final"]
            },
            "finest_energy_reference_uncertainty": {
                field: distribution(
                    [
                        record["energy_reference_uncertainty_at_finest_dt"][
                            field
                        ]["absolute"]
                        for record in selected
                    ]
                )
                for field in ["maximum_absolute_drift", "final_absolute_drift"]
            },
        }
    return result


def level_uncertainties(record, field):
    order = record["state_observed_order"][field]
    effective = 0.5 if order is None or order <= 0 else min(order, 4.0)
    differences = record["state_level_differences"]
    coarse_difference = differences["0.002_vs_0.001"][field]
    coarse_uncertainty = coarse_difference / (1.0 - 2.0 ** (-effective))
    return [
        coarse_uncertainty,
        record["state_reference_uncertainty"]["ac13_dt"][field]["absolute"],
        record["state_reference_uncertainty"]["finest_dt"][field]["absolute"],
    ]


def make_figure(records):
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5))
    dt_values = np.asarray([0.002, 0.001, 0.0005])
    for axis, field in zip(axes, ["maximum", "final"]):
        for system_name, color in [("modal", "#4C72B0"), ("duffing", "#C44E52")]:
            selected = [
                level_uncertainties(record, field)
                for record in records
                if record["system"] == system_name
            ]
            medians = np.median(np.asarray(selected), axis=0)
            axis.loglog(
                dt_values,
                medians,
                marker="o",
                color=color,
                label=("Modal N=16" if system_name == "modal" else "Duffing"),
            )
        axis.set_xlabel("Velocity-Verlet dt")
        axis.set_ylabel("Median reference uncertainty")
        axis.set_title(
            "Maximum state" if field == "maximum" else "Final state"
        )
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "AC16 reference convergence (finest points are Richardson uncertainty)"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    protocol = load(PROTOCOL_PATH)
    raw = load(RAW_PATH)
    summary = load(SUMMARY_PATH)
    ac13_raw = load(AC13_RAW_PATH)
    references = {scenario_key(record): record for record in raw["records"]}
    corrected = corrected_records(ac13_raw["records"], references)
    rankings = aggregate_rankings(corrected)
    analysis = {
        "schema_version": "1.0",
        "experiment": raw["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": raw["protocol_sha256"],
        "source_sha256": {
            "protocol": sha256_file(PROTOCOL_PATH),
            "raw": sha256_file(RAW_PATH),
            "summary": sha256_file(SUMMARY_PATH),
            "ac13_raw": sha256_file(AC13_RAW_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "reference_convergence": convergence_summary(raw["records"]),
        "ac13_corrected_record_count": len(corrected),
        "aggregate_state_error_rankings": rankings,
        "learned_vs_prescribed_scenario_pairs":
            learned_prescribed_pairs(corrected),
        "ranking_stability": {
            key: value["reported_total_order_stable"]
            for key, value in rankings.items()
        },
        "artifact_gate": summary["acceptance"],
        "interpretation_limits": protocol["claim_limits"],
        "reference_display_rule":
            "The finest reference is displayed at its positive Richardson "
            "uncertainty; no reference is represented as exact zero.",
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(raw["records"])
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
