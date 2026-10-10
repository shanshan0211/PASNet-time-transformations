"""Completion-first checkpoint-cluster analysis for sealed AC13."""

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
OUT = os.path.join(ROOT, "results", "expAC13")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(OUT, "fig_cross_hamiltonian_scale.pdf")
FIGURE_PNG = os.path.join(OUT, "fig_cross_hamiltonian_scale.png")


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


def checkpoint_completion(records, system_name, architecture):
    result = {}
    for seed in [9101, 9102, 9103]:
        selected = [
            record for record in records
            if record["method"] == "learned_monitor"
            and record["system"] == system_name
            and record["architecture"] == architecture
            and record["checkpoint_seed"] == seed
        ]
        result[str(seed)] = completion(selected)
    return result


def cluster_uncertainty(records, system_name, n_draws=20000):
    seeds = [9101, 9102, 9103]
    values = {}
    for architecture in ["raw", "reversal_even"]:
        values[architecture] = np.asarray(
            [
                checkpoint_completion(
                    records, system_name, architecture
                )[str(seed)]["fraction"]
                for seed in seeds
            ],
            dtype=float,
        )
    differences = values["reversal_even"] - values["raw"]
    rng = np.random.default_rng(1313)

    def interval(sample):
        draws = np.asarray(
            [
                np.mean(sample[rng.integers(0, len(sample), len(sample))])
                for _ in range(n_draws)
            ]
        )
        return [float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975))]

    return {
        "n_checkpoints_per_architecture": 3,
        "bootstrap_draws": n_draws,
        "bootstrap_seed": 1313,
        "architecture": {
            architecture: {
                "checkpoint_mean": float(np.mean(sample)),
                "checkpoint_standard_deviation": float(
                    np.std(sample, ddof=1)
                ),
                "checkpoint_bootstrap_95": interval(sample),
            }
            for architecture, sample in values.items()
        },
        "reversal_even_minus_raw": {
            "mean_difference": float(np.mean(differences)),
            "checkpoint_bootstrap_95": interval(differences),
            "per_seed_differences": {
                str(seed): float(value)
                for seed, value in zip(seeds, differences)
            },
        },
    }


def stratified_completion(records, field):
    result = {}
    learned = [
        record for record in records if record["method"] == "learned_monitor"
    ]
    for system_name in ["modal", "duffing"]:
        system_records = [
            record for record in learned if record["system"] == system_name
        ]
        values = sorted(
            {record.get(field) for record in system_records},
            key=lambda value: (value is None, str(value)),
        )
        for architecture in ["raw", "reversal_even"]:
            for value in values:
                selected = [
                    record for record in system_records
                    if record["architecture"] == architecture
                    and record.get(field) == value
                ]
                result[f"{system_name}|{architecture}|{value}"] = completion(
                    selected
                )
    return result


def completed_metrics(records):
    fields = [
        "max_energy_error",
        "max_state_error",
        "final_state_error",
        "steps",
        "vector_field_evaluations",
        "nonlinear_iterations",
        "wall_seconds",
    ]
    result = {}
    for system_name in ["modal", "duffing"]:
        for method, architecture in [
            ("learned_monitor", "raw"),
            ("learned_monitor", "reversal_even"),
            ("prescribed_monitor", None),
            ("resolved_velocity_verlet", None),
        ]:
            selected = [
                record for record in records
                if record["system"] == system_name
                and record["method"] == method
                and record.get("architecture") == architecture
                and record["completion_status"] == "complete"
            ]
            label = (
                architecture if method == "learned_monitor" else method
            )
            result[f"{system_name}|{label}"] = {
                "completed": len(selected),
                "survivor_conditioned": True,
                "median": {
                    field: (
                        float(np.median([record[field] for record in selected]))
                        if selected else None
                    )
                    for field in fields
                },
            }
    return result


def reversal_transfer(records):
    groups = {}
    for record in records:
        if record["method"] != "learned_monitor":
            continue
        key = (
            record["system"],
            record["architecture"],
            record["checkpoint_seed"],
            record["energy_shell"],
            record["initial_condition_seed"],
            record.get("kappa"),
        )
        groups.setdefault(key, []).append(record)
    result = {}
    for system_name in ["modal", "duffing"]:
        for architecture in ["raw", "reversal_even"]:
            selected = [
                group for key, group in groups.items()
                if key[0] == system_name and key[1] == architecture
            ]
            result[f"{system_name}|{architecture}"] = {
                "registered_pairs": len(selected),
                "completion_agreement_pairs": sum(
                    len(group) == 2
                    and len({
                        record["completion_status"] for record in group
                    }) == 1
                    for group in selected
                ),
                "both_complete_pairs": sum(
                    len(group) == 2
                    and all(record["completion_status"] == "complete"
                            for record in group)
                    for group in selected
                ),
            }
    return result


def make_figure(analysis):
    colors = {"raw": "#C44E52", "reversal_even": "#4C72B0"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    for system_index, system_name in enumerate(["modal", "duffing"]):
        for architecture_index, architecture in enumerate(
            ["raw", "reversal_even"]
        ):
            values = [
                record["fraction"]
                for record in analysis["checkpoint_completion"][
                    f"{system_name}|{architecture}"
                ].values()
            ]
            x = 2 * system_index + architecture_index
            axes[0].scatter(
                [x - 0.08, x, x + 0.08],
                values,
                color=colors[architecture],
                s=32,
            )
    axes[0].set_xticks(
        [0.5, 2.5], ["Modal (32D)", "Duffing (2D)"]
    )
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_ylabel("Per-checkpoint completion")
    axes[0].set_title("(a) Independent checkpoints")

    shells = [0.75, 1.0, 1.25]
    for architecture in ["raw", "reversal_even"]:
        values = [
            analysis["completion_by_shell"][
                f"modal|{architecture}|{shell}"
            ]["fraction"]
            for shell in shells
        ]
        axes[1].plot(
            shells,
            values,
            marker="o",
            color=colors[architecture],
            label=("Reversal-even" if architecture == "reversal_even"
                   else "Raw"),
        )
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_xlabel("Modal energy shell")
    axes[1].set_ylabel("Completion fraction")
    axes[1].set_title("(b) Modal shell transfer")
    axes[1].legend(frameon=False, fontsize=8)

    labels = ["raw", "reversal_even", "prescribed_monitor"]
    medians = [
        analysis["completed_metrics"][f"modal|{label}"]["median"][
            "final_state_error"
        ]
        for label in labels
    ]
    axes[2].bar(
        range(3),
        [value if value is not None else np.nan for value in medians],
        color=["#C44E52", "#4C72B0", "#55A868"],
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(
        range(3), ["Raw", "Rev.-even", "Prescribed"], rotation=15
    )
    axes[2].set_ylabel("Median final state error")
    axes[2].set_title("(c) Modal state accuracy")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    raw = load(RAW_PATH)
    summary = load(SUMMARY_PATH)
    training = load(TRAINING_PATH)
    records = raw["records"]
    analysis = {
        "schema_version": "1.0",
        "experiment": raw["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": raw["protocol_sha256"],
        "source_sha256": {
            "training": sha256_file(TRAINING_PATH),
            "raw": sha256_file(RAW_PATH),
            "summary": sha256_file(SUMMARY_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "completion_status_counts": dict(
            Counter(record["completion_status"] for record in records)
        ),
        "checkpoint_completion": {
            f"{system_name}|{architecture}": checkpoint_completion(
                records, system_name, architecture
            )
            for system_name in ["modal", "duffing"]
            for architecture in ["raw", "reversal_even"]
        },
        "checkpoint_cluster_uncertainty": {
            system_name: cluster_uncertainty(records, system_name)
            for system_name in ["modal", "duffing"]
        },
        "completion_by_shell": stratified_completion(
            records, "energy_shell"
        ),
        "completion_by_initial_condition": stratified_completion(
            records, "initial_condition_seed"
        ),
        "completion_by_reversal": stratified_completion(
            records, "momentum_direction"
        ),
        "completion_by_modal_kappa": stratified_completion(records, "kappa"),
        "reversal_transfer": reversal_transfer(records),
        "completed_metrics": completed_metrics(records),
        "training": {
            "checkpoint_records": len(training["records"]),
            "best_validation_state_mse": {
                f"{system_name}|{architecture}": [
                    record["best_validation_state_mse"]
                    for record in training["records"]
                    if record["system"] == system_name
                    and record["architecture"] == architecture
                    and record["status"] != "training_failure"
                ]
                for system_name in ["modal", "duffing"]
                for architecture in ["raw", "reversal_even"]
            },
        },
        "artifact_gate": summary["acceptance"],
        "cross_system_summary":
            "Descriptive side-by-side results only; no pooled universal claim.",
        "interpretation_limits": [
            "Three checkpoints per architecture give small-cluster uncertainty.",
            "Completed error and cost summaries are survivor-conditioned.",
            "Reversal-even means momentum-reversal evenness only.",
            "Exact H and fixed-point evaluation only; Newton is not included.",
        ],
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(analysis)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
