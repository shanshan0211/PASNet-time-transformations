"""Analyze corrected expAC2 controls without changing the raw artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone


def label(record):
    if record["method"] == "direct_implicit_midpoint":
        return f"direct_dt={record['dt']:g}"
    if record["method"] == "adaptive_step_doubling_midpoint":
        return f"adaptive_rtol={record['rtol']:g}"
    return f"{record['monitor']}_dtau={record['dtau']:g}"


def dominates(left, right, error_key, cost_key):
    no_worse = (
        left[error_key] <= right[error_key]
        and left[cost_key] <= right[cost_key]
    )
    strictly_better = (
        left[error_key] < right[error_key]
        or left[cost_key] < right[cost_key]
    )
    return no_worse and strictly_better


def refinement_orders(records, method, parameter, error_key, monitor=None):
    subset = [record for record in records if record["method"] == method]
    if monitor is not None:
        subset = [record for record in subset if record.get("monitor") == monitor]
    subset.sort(key=lambda record: record[parameter], reverse=True)
    results = []
    for coarse, fine in zip(subset, subset[1:]):
        if coarse[error_key] <= 0.0 or fine[error_key] <= 0.0:
            continue
        order = math.log(coarse[error_key] / fine[error_key]) / math.log(
            coarse[parameter] / fine[parameter]
        )
        results.append(
            {
                "coarse": label(coarse),
                "fine": label(fine),
                "error_metric": error_key,
                "observed_order": order,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or os.path.join(
        os.path.dirname(args.input), "corrected_100period_analysis.json"
    )

    with open(args.input, "rb") as handle:
        raw_bytes = handle.read()
    raw = json.loads(raw_bytes)
    records = raw["records"]
    labels = {label(record): record for record in records}

    dominance = {}
    for error_key in ("max_energy_drift", "final_state_error"):
        for cost_key in ("gradient_evaluations", "wall_seconds"):
            key = f"{error_key}_vs_{cost_key}"
            dominance[key] = {
                candidate: [
                    competitor
                    for competitor, competitor_record in labels.items()
                    if competitor != candidate
                    and dominates(
                        competitor_record,
                        candidate_record,
                        error_key,
                        cost_key,
                    )
                ]
                for candidate, candidate_record in labels.items()
            }

    orders = []
    for error_key in ("max_energy_drift", "final_state_error"):
        orders.extend(
            refinement_orders(
                records,
                "direct_implicit_midpoint",
                "dt",
                error_key,
            )
        )
        for monitor in ("manual_r", "learned_g"):
            orders.extend(
                refinement_orders(
                    records,
                    "extended_implicit_midpoint",
                    "dtau",
                    error_key,
                    monitor=monitor,
                )
            )

    rankings = {}
    for metric in (
        "max_energy_drift",
        "final_state_error",
        "max_angular_momentum_error",
        "gradient_evaluations",
        "wall_seconds",
    ):
        rankings[metric] = [
            {"method": method, "value": record[metric]}
            for method, record in sorted(
                labels.items(), key=lambda item: item[1][metric]
            )
        ]

    learned_labels = [
        method for method in labels if method.startswith("learned_g")
    ]
    learned_dominance = {
        criterion: {
            method: competitors
            for method, competitors in methods.items()
            if method in learned_labels
        }
        for criterion, methods in dominance.items()
    }

    artifact = {
        "schema_version": "1.0",
        "experiment": "expAC2_corrected_analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_artifact": os.path.basename(args.input),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "periods": raw["periods"],
        "acceptance": raw["acceptance"],
        "rankings": rankings,
        "observed_refinement_orders": orders,
        "pareto_dominance": dominance,
        "learned_monitor_dominance": learned_dominance,
        "interpretation_rules": [
            "Dominance requires no larger error and no larger cost, with at least one strict inequality.",
            "Wall-clock comparisons are host-specific; gradient counts are the portable cost control.",
            "A finite grid can show dominance among tested configurations, not global method dominance.",
        ],
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
