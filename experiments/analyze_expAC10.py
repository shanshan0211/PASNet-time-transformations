"""Analyze sealed AC10 completion-aware accuracy--cost results."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC10")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(ROOT, "fig_fair_accuracy_cost.pdf")
FIGURE_PNG = os.path.join(ROOT, "fig_fair_accuracy_cost.png")


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


def geometric_mean(values):
    finite = [
        value
        for value in values
        if value is not None and value > 0.0 and math.isfinite(value)
    ]
    return (
        float(math.exp(np.mean(np.log(finite)))) if finite else None
    )


def configuration_summary(records):
    complete = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    first = records[0]
    result = {
        "configuration_id": first["configuration_id"],
        "method": first["method"],
        "architecture": first.get("architecture"),
        "checkpoint_seed": first.get("checkpoint_seed"),
        "registered": len(records),
        "complete": len(complete),
        "completion_fraction": len(complete) / len(records),
        "robust_front_eligible": len(complete) == len(records),
        "configuration_peak_rss_kib":
            first["configuration_peak_rss_kib"],
        "survivor_conditioned_metrics": {},
    }
    for metric in [
        "max_energy_error",
        "final_state_closure_error",
        "max_angular_momentum_error",
        "gradient_evaluations",
        "steps",
        "wall_seconds",
    ]:
        result["survivor_conditioned_metrics"][metric] = {
            "geometric_mean": geometric_mean(
                [record[metric] for record in complete]
            ),
            "median": (
                float(np.median([record[metric] for record in complete]))
                if complete
                else None
            ),
            "n_complete": len(complete),
        }
    return result


def dominates(candidate, other, x_metric, y_metric):
    candidate_x = candidate["survivor_conditioned_metrics"][x_metric][
        "geometric_mean"
    ]
    candidate_y = candidate["survivor_conditioned_metrics"][y_metric][
        "geometric_mean"
    ]
    other_x = other["survivor_conditioned_metrics"][x_metric][
        "geometric_mean"
    ]
    other_y = other["survivor_conditioned_metrics"][y_metric][
        "geometric_mean"
    ]
    return (
        candidate_x <= other_x
        and candidate_y <= other_y
        and (candidate_x < other_x or candidate_y < other_y)
    )


def pareto_front(configurations, x_metric, y_metric):
    eligible = [
        configuration
        for configuration in configurations
        if configuration["robust_front_eligible"]
    ]
    front = []
    for candidate in eligible:
        if not any(
            dominates(other, candidate, x_metric, y_metric)
            for other in eligible
            if other is not candidate
        ):
            front.append(candidate)
    return [
        configuration["configuration_id"]
        for configuration in sorted(
            front,
            key=lambda item: item["survivor_conditioned_metrics"][
                x_metric
            ]["geometric_mean"],
        )
    ]


def architecture_dtau_summary(records, architecture, dtau):
    selected = [
        record
        for record in records
        if record.get("architecture") == architecture
        and record["configuration_id"].endswith(f"dtau_{dtau}")
    ]
    seeds = sorted(
        {record["checkpoint_seed"] for record in selected}
    )
    per_seed = {}
    for seed in seeds:
        seed_records = [
            record
            for record in selected
            if record["checkpoint_seed"] == seed
        ]
        per_seed[str(seed)] = {
            "complete": sum(
                record["completion_status"] == "complete"
                for record in seed_records
            ),
            "registered": len(seed_records),
        }
        per_seed[str(seed)]["fraction"] = (
            per_seed[str(seed)]["complete"] / len(seed_records)
        )
    values = np.array(
        [record["fraction"] for record in per_seed.values()]
    )
    rng = np.random.default_rng(
        10110 + int(round(dtau * 1000))
        + (0 if architecture == "raw" else 1000)
    )
    draws = np.empty(20000)
    for index in range(len(draws)):
        sample = rng.integers(0, len(values), size=len(values))
        draws[index] = np.mean(values[sample])
    complete = [
        record
        for record in selected
        if record["completion_status"] == "complete"
    ]
    return {
        "registered": len(selected),
        "complete": len(complete),
        "completion_fraction": len(complete) / len(selected),
        "per_checkpoint": per_seed,
        "checkpoint_mean": float(np.mean(values)),
        "checkpoint_bootstrap_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "survivor_conditioned": True,
        "geometric_means": {
            metric: geometric_mean(
                [record[metric] for record in complete]
            )
            for metric in [
                "max_energy_error",
                "final_state_closure_error",
                "gradient_evaluations",
                "wall_seconds",
            ]
        },
    }


def plot_style(configuration):
    method = configuration["method"]
    architecture = configuration.get("architecture")
    if method == "direct_implicit_midpoint":
        return "#8172B2", "s", "Direct IM"
    if method == "adaptive_implicit_midpoint":
        return "#DD8452", "^", "Adaptive IM"
    if method == "manual_sundman":
        return "#55A868", "o", "Manual $r$"
    if architecture == "raw":
        return "#C44E52", "x", "Raw learned"
    return "#4C72B0", "D", "Invariant learned"


def make_figure(configurations, fronts):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    labelled = set()
    for configuration in configurations:
        color, marker, label = plot_style(configuration)
        metrics = configuration["survivor_conditioned_metrics"]
        complete = configuration["robust_front_eligible"]
        display_label = label if label not in labelled else None
        labelled.add(label)
        face = color if complete else "none"
        axes[0].scatter(
            metrics["gradient_evaluations"]["geometric_mean"],
            metrics["max_energy_error"]["geometric_mean"],
            marker=marker,
            color=color,
            facecolors=face,
            alpha=0.8,
            label=display_label,
        )
        axes[1].scatter(
            metrics["gradient_evaluations"]["geometric_mean"],
            metrics["final_state_closure_error"]["geometric_mean"],
            marker=marker,
            color=color,
            facecolors=face,
            alpha=0.8,
        )
        axes[2].scatter(
            configuration["configuration_peak_rss_kib"] / 1024.0,
            configuration["completion_fraction"],
            marker=marker,
            color=color,
            facecolors=face,
            alpha=0.8,
        )

    by_id = {
        configuration["configuration_id"]: configuration
        for configuration in configurations
    }
    for axis, key, y_metric in [
        (axes[0], "energy_vs_evaluations", "max_energy_error"),
        (axes[1], "closure_vs_evaluations", "final_state_closure_error"),
    ]:
        front = [by_id[identifier] for identifier in fronts[key]]
        front.sort(
            key=lambda item: item["survivor_conditioned_metrics"][
                "gradient_evaluations"
            ]["geometric_mean"]
        )
        axis.plot(
            [
                item["survivor_conditioned_metrics"][
                    "gradient_evaluations"
                ]["geometric_mean"]
                for item in front
            ],
            [
                item["survivor_conditioned_metrics"][y_metric][
                    "geometric_mean"
                ]
                for item in front
            ],
            color="black",
            linewidth=1,
            linestyle=":",
            label="18/18-complete front" if axis is axes[0] else None,
        )
    axes[0].set_xlabel("Vector-field evaluations")
    axes[0].set_ylabel("Geomean max energy error")
    axes[0].set_title("(a) Energy--cost")
    axes[1].set_xlabel("Vector-field evaluations")
    axes[1].set_ylabel("Geomean closure error")
    axes[1].set_title("(b) State--cost")
    axes[2].set_xlabel("Configuration peak RSS (MiB)")
    axes[2].set_ylabel("Completion fraction")
    axes[2].set_ylim(-0.04, 1.04)
    axes[2].set_title("(c) Completion--memory")
    axes[0].legend(frameon=False, fontsize=7)
    for axis in axes[:2]:
        axis.set_xscale("log")
        axis.set_yscale("log")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def failure_classification(records):
    raw_counts = {}
    normalized_counts = {}
    for record in records:
        status = record["completion_status"]
        raw_counts[status] = raw_counts.get(status, 0) + 1
        normalized = status
        if (
            status == "unexpected_error"
            and "implicit midpoint did not converge"
            in record.get("error", "")
        ):
            normalized = "direct_fixed_point_nonconvergence"
        elif status == "solver_failure":
            normalized = "transformed_fixed_point_nonconvergence"
        normalized_counts[normalized] = (
            normalized_counts.get(normalized, 0) + 1
        )
    return {
        "raw_artifact_statuses": raw_counts,
        "analysis_normalized_reasons": normalized_counts,
        "note":
            "The runner's broad exception wrapper labels direct midpoint "
            "RuntimeError nonconvergence as unexpected_error; all 20 such "
            "records contain the explicit 100-iteration nonconvergence "
            "message and are retained unchanged in raw_results.json.",
    }


def main():
    raw = load(RAW_PATH)
    summary = load(SUMMARY_PATH)
    grouped = {}
    for record in raw["records"]:
        grouped.setdefault(record["configuration_id"], []).append(record)
    configurations = [
        configuration_summary(records)
        for _, records in sorted(grouped.items())
    ]
    fronts = {
        "energy_vs_evaluations": pareto_front(
            configurations,
            "gradient_evaluations",
            "max_energy_error",
        ),
        "closure_vs_evaluations": pareto_front(
            configurations,
            "gradient_evaluations",
            "final_state_closure_error",
        ),
        "closure_vs_wall": pareto_front(
            configurations,
            "wall_seconds",
            "final_state_closure_error",
        ),
    }
    analysis = {
        "schema_version": "1.0",
        "experiment": raw["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": raw["protocol_sha256"],
        "source_sha256": {
            "raw": sha256_file(RAW_PATH),
            "summary": sha256_file(SUMMARY_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "configuration_summaries": configurations,
        "learned_architecture_dtau": {
            f"{architecture}|{dtau}": architecture_dtau_summary(
                raw["records"], architecture, dtau
            )
            for architecture in ["raw", "invariant"]
            for dtau in [0.08, 0.04, 0.02]
        },
        "robust_pareto_fronts": fronts,
        "failure_classification": failure_classification(raw["records"]),
        "artifact_gate": summary["acceptance"],
        "interpretation": {
            "filled_markers": "18/18 complete and robust-front eligible",
            "open_markers":
                "incomplete; plotted using completed-run survivor metrics "
                "but excluded from the robust front",
            "memory":
                "configuration-level subprocess maximum RSS, not per-rollout "
                "incremental allocation",
            "wall_time": "method-only host-specific wall time",
        },
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(configurations, fronts)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
