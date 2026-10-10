"""Checkpoint-level analysis for sealed AC9 generalisation artifacts."""

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
OUT = os.path.join(ROOT, "results", "expAC9")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(ROOT, "fig_checkpoint_generalisation.pdf")
FIGURE_PNG = os.path.join(ROOT, "fig_checkpoint_generalisation.png")


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


def wilson(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [centre - half, centre + half]


def completion_record(records):
    successes = sum(
        record["completion_status"] == "complete" for record in records
    )
    return {
        "complete": successes,
        "registered": len(records),
        "fraction": successes / len(records),
        "rollout_wilson_95": wilson(successes, len(records)),
    }


def checkpoint_completion(records, architecture):
    result = {}
    for seed in sorted(
        {
            record["checkpoint_seed"]
            for record in records
            if record["architecture"] == architecture
        }
    ):
        selected = [
            record
            for record in records
            if record["architecture"] == architecture
            and record["checkpoint_seed"] == seed
        ]
        result[str(seed)] = completion_record(selected)
    return result


def cluster_bootstrap(records, n_draws=20000):
    architectures = ["raw", "invariant"]
    seeds = sorted(
        {
            record["checkpoint_seed"]
            for record in records
            if record["architecture"] == "raw"
        }
    )
    fractions = {}
    for architecture in architectures:
        fractions[architecture] = {}
        for seed in seeds:
            selected = [
                record
                for record in records
                if record["architecture"] == architecture
                and record["checkpoint_seed"] == seed
            ]
            fractions[architecture][seed] = np.mean(
                [
                    record["completion_status"] == "complete"
                    for record in selected
                ]
            )
    differences = np.array(
        [
            fractions["invariant"][seed] - fractions["raw"][seed]
            for seed in seeds
        ]
    )
    rng = np.random.default_rng(9909)
    bootstrap = np.empty(n_draws)
    for draw in range(n_draws):
        indices = rng.integers(0, len(seeds), size=len(seeds))
        bootstrap[draw] = np.mean(differences[indices])
    architecture_intervals = {}
    for architecture in architectures:
        values = np.array(
            [fractions[architecture][seed] for seed in seeds]
        )
        draws = np.empty(n_draws)
        for draw in range(n_draws):
            indices = rng.integers(0, len(seeds), size=len(seeds))
            draws[draw] = np.mean(values[indices])
        architecture_intervals[architecture] = {
            "checkpoint_mean": float(np.mean(values)),
            "checkpoint_standard_deviation": float(
                np.std(values, ddof=1)
            ),
            "checkpoint_bootstrap_95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return {
        "n_checkpoints_per_architecture": len(seeds),
        "bootstrap_draws": n_draws,
        "bootstrap_seed": 9909,
        "architecture": architecture_intervals,
        "paired_invariant_minus_raw": {
            "mean_difference": float(np.mean(differences)),
            "checkpoint_bootstrap_95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "per_seed_differences": {
                str(seed): float(value)
                for seed, value in zip(seeds, differences)
            },
        },
    }


def geometric_mean(values):
    finite = [
        value
        for value in values
        if value is not None and value > 0.0 and math.isfinite(value)
    ]
    return (
        float(math.exp(np.mean(np.log(finite)))) if finite else None
    )


def completed_pair_ratios(records):
    index = {
        (
            record["architecture"],
            record["checkpoint_seed"],
            record["semi_major_axis"],
            record["eccentricity"],
            record["initial_mean_anomaly"],
            record["orientation_radians"],
            record["momentum_direction"],
        ): record
        for record in records
        if record["architecture"] in {"raw", "invariant"}
    }
    values = {
        "energy": [],
        "closure": [],
        "steps": [],
        "gradient_evaluations": [],
        "wall_seconds": [],
    }
    complete_pairs = 0
    seeds = sorted(
        {
            record["checkpoint_seed"]
            for record in records
            if record["architecture"] == "raw"
        }
    )
    scenario_keys = {
        (
            record["semi_major_axis"],
            record["eccentricity"],
            record["initial_mean_anomaly"],
            record["orientation_radians"],
            record["momentum_direction"],
        )
        for record in records
        if record["architecture"] == "raw"
    }
    metric_fields = {
        "energy": "max_energy_error",
        "closure": "final_state_closure_error",
        "steps": "steps",
        "gradient_evaluations": "gradient_evaluations",
        "wall_seconds": "wall_seconds",
    }
    for seed in seeds:
        for scenario in scenario_keys:
            raw = index[("raw", seed, *scenario)]
            invariant = index[("invariant", seed, *scenario)]
            if (
                raw["completion_status"] == "complete"
                and invariant["completion_status"] == "complete"
            ):
                complete_pairs += 1
                for metric, field in metric_fields.items():
                    values[metric].append(
                        invariant[field] / max(raw[field], 1e-300)
                    )
    return {
        "n_complete_pairs": complete_pairs,
        "registered_pairs": len(seeds) * len(scenario_keys),
        "survivor_conditioned": True,
        "invariant_over_raw_geometric_mean": {
            metric: geometric_mean(metric_values)
            for metric, metric_values in values.items()
        },
    }


def symmetry_groups(records, architecture):
    groups = {}
    for record in records:
        if record["architecture"] != architecture:
            continue
        key = (
            record["checkpoint_seed"],
            record["semi_major_axis"],
            record["eccentricity"],
            record["initial_mean_anomaly"],
        )
        groups.setdefault(key, []).append(record)
    uniform = 0
    complete_groups = []
    for group in groups.values():
        statuses = [
            record["completion_status"] == "complete" for record in group
        ]
        if all(status == statuses[0] for status in statuses):
            uniform += 1
        if all(statuses):
            energies = [record["max_energy_error"] for record in group]
            closures = [
                record["final_state_closure_error"] for record in group
            ]
            complete_groups.append(
                (
                    max(energies) / max(min(energies), 1e-300),
                    max(closures) / max(min(closures), 1e-300),
                )
            )
    return {
        "registered_groups": len(groups),
        "completion_uniform_groups": uniform,
        "all_four_complete_groups": len(complete_groups),
        "maximum_energy_max_min_ratio": max(
            (value[0] for value in complete_groups), default=None
        ),
        "maximum_closure_max_min_ratio": max(
            (value[1] for value in complete_groups), default=None
        ),
    }


def failure_reasons(records):
    result = {}
    for architecture in ["raw", "invariant", "manual_r"]:
        selected = [
            record
            for record in records
            if record["architecture"] == architecture
            and record["completion_status"] != "complete"
        ]
        reasons = Counter()
        for record in selected:
            error = record.get("error", "")
            if "fixed point did not converge" in error:
                reasons["fixed_point_iteration_cap"] += 1
            else:
                reasons[record["completion_status"]] += 1
        result[architecture] = dict(reasons)
    return result


def make_figure(analysis):
    colors = {
        "raw": "#C44E52",
        "invariant": "#4C72B0",
        "manual_r": "#55A868",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    rng = np.random.default_rng(14)
    for x, architecture in enumerate(["raw", "invariant"]):
        values = [
            record["fraction"]
            for record in analysis["checkpoint_completion"][
                architecture
            ].values()
        ]
        axes[0].scatter(
            x + rng.uniform(-0.07, 0.07, len(values)),
            values,
            color=colors[architecture],
            s=34,
            zorder=3,
        )
        stats = analysis["checkpoint_bootstrap"]["architecture"][
            architecture
        ]
        interval = stats["checkpoint_bootstrap_95"]
        axes[0].errorbar(
            x,
            stats["checkpoint_mean"],
            yerr=[
                [stats["checkpoint_mean"] - interval[0]],
                [interval[1] - stats["checkpoint_mean"]],
            ],
            fmt="_",
            markersize=18,
            capsize=5,
            color="black",
            linewidth=1.5,
            zorder=4,
        )
    axes[0].set_xticks([0, 1], ["Raw", "Invariant"])
    axes[0].set_ylabel("Per-checkpoint completion")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_title("(a) Independent checkpoints")

    shells = analysis["semi_major_axes"]
    for architecture in ["raw", "invariant", "manual_r"]:
        values = [
            analysis["completion_by_shell"][
                f"{architecture}|{shell}"
            ]["fraction"]
            for shell in shells
        ]
        axes[1].plot(
            shells,
            values,
            marker="o",
            color=colors[architecture],
            label=("Manual $r$" if architecture == "manual_r" else architecture.capitalize()),
        )
    axes[1].axvline(1.0, color="0.4", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Semi-major axis $a$")
    axes[1].set_ylabel("Completion fraction")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_title("(b) Energy-shell transfer")
    axes[1].legend(frameon=False, fontsize=8)

    eccentricities = analysis["eccentricities"]
    for architecture in ["raw", "invariant", "manual_r"]:
        values = [
            analysis["completion_by_eccentricity"][
                f"{architecture}|{eccentricity}"
            ]["fraction"]
            for eccentricity in eccentricities
        ]
        axes[2].plot(
            eccentricities,
            values,
            marker="o",
            color=colors[architecture],
            label=("Manual $r$" if architecture == "manual_r" else architecture.capitalize()),
        )
    axes[2].set_xlabel("Eccentricity")
    axes[2].set_ylabel("Completion fraction")
    axes[2].set_ylim(-0.04, 1.04)
    axes[2].set_title("(c) Eccentricity boundary")

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
    architectures = ["raw", "invariant", "manual_r"]
    semi_major_axes = sorted(
        {record["semi_major_axis"] for record in records}
    )
    eccentricities = sorted(
        {record["eccentricity"] for record in records}
    )
    completion = {
        architecture: completion_record(
            [
                record
                for record in records
                if record["architecture"] == architecture
            ]
        )
        for architecture in architectures
    }
    completion_by_shell = {}
    for architecture in architectures:
        for shell in semi_major_axes:
            completion_by_shell[f"{architecture}|{shell}"] = (
                completion_record(
                    [
                        record
                        for record in records
                        if record["architecture"] == architecture
                        and record["semi_major_axis"] == shell
                    ]
                )
            )
    completion_by_eccentricity = {}
    for architecture in architectures:
        for eccentricity in eccentricities:
            completion_by_eccentricity[
                f"{architecture}|{eccentricity}"
            ] = completion_record(
                [
                    record
                    for record in records
                    if record["architecture"] == architecture
                    and record["eccentricity"] == eccentricity
                ]
            )
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
        "semi_major_axes": semi_major_axes,
        "eccentricities": eccentricities,
        "completion": completion,
        "completion_by_shell": completion_by_shell,
        "completion_by_eccentricity": completion_by_eccentricity,
        "checkpoint_completion": {
            architecture: checkpoint_completion(records, architecture)
            for architecture in ["raw", "invariant"]
        },
        "checkpoint_bootstrap": cluster_bootstrap(records),
        "completed_pair_ratios": completed_pair_ratios(records),
        "symmetry_groups": {
            architecture: symmetry_groups(records, architecture)
            for architecture in architectures
        },
        "failure_reasons": failure_reasons(records),
        "training": {
            "n_checkpoints": len(training["records"]),
            "best_validation_state_mse": {
                architecture: [
                    record["best_validation_state_mse"]
                    for record in training["records"]
                    if record["architecture"] == architecture
                ]
                for architecture in ["raw", "invariant"]
            },
        },
        "artifact_gate": summary["acceptance"],
        "interpretation_limits": [
            "Five checkpoints per architecture give a small cluster sample.",
            "Rollout-level Wilson intervals are secondary and do not account "
            "for checkpoint clustering.",
            "Completed-pair error ratios are survivor-conditioned.",
            "Training uses one exact-H orbit and one energy shell.",
            "Evaluation uses fixed point; AC8 separately tests solver choice.",
        ],
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(analysis)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
