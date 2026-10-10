"""Analyze the sealed AC12 Hamiltonian/monitor/solver factorial."""

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
OUT = os.path.join(ROOT, "results", "expAC12")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(ROOT, "fig_learned_h_factorial.pdf")
FIGURE_PNG = os.path.join(ROOT, "fig_learned_h_factorial.png")


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


def selected(records, **factors):
    return [
        record
        for record in records
        if all(record.get(name) == value for name, value in factors.items())
    ]


def checkpoint_fractions(records):
    result = {}
    for h_mode in ["exact", "learned"]:
        for architecture in ["raw", "invariant"]:
            for solver in ["fixed_point", "monotone_residual_newton"]:
                for seed in sorted(
                    {
                        record["checkpoint_seed"]
                        for record in records
                        if record["checkpoint_seed"] is not None
                    }
                ):
                    arm = selected(
                        records,
                        h_mode=h_mode,
                        architecture=architecture,
                        solver=solver,
                        checkpoint_seed=seed,
                    )
                    result[
                        f"{h_mode}|{architecture}|{solver}|{seed}"
                    ] = completion(arm)
    return result


def cluster_estimands(records, n_draws=20000):
    seeds = [8101, 8102, 8103, 8104, 8105]
    solvers = ["fixed_point", "monotone_residual_newton"]
    architectures = ["raw", "invariant"]
    modes = ["exact", "learned"]

    def fraction(mode, architecture, solver, seed):
        arm = selected(
            records,
            h_mode=mode,
            architecture=architecture,
            solver=solver,
            checkpoint_seed=seed,
        )
        return np.mean(
            [record["completion_status"] == "complete" for record in arm]
        )

    values = {
        (mode, architecture, solver, seed):
            fraction(mode, architecture, solver, seed)
        for mode in modes
        for architecture in architectures
        for solver in solvers
        for seed in seeds
    }
    per_seed = {}
    for seed in seeds:
        architecture_contrast = {
            mode: np.mean(
                [
                    values[(mode, "invariant", solver, seed)]
                    - values[(mode, "raw", solver, seed)]
                    for solver in solvers
                ]
            )
            for mode in modes
        }
        solver_contrast = {
            mode: np.mean(
                [
                    values[
                        (
                            mode,
                            architecture,
                            "monotone_residual_newton",
                            seed,
                        )
                    ]
                    - values[(mode, architecture, "fixed_point", seed)]
                    for architecture in architectures
                ]
            )
            for mode in modes
        }
        h_contrast = np.mean(
            [
                values[("learned", architecture, solver, seed)]
                - values[("exact", architecture, solver, seed)]
                for architecture in architectures
                for solver in solvers
            ]
        )
        per_seed[seed] = {
            "H_by_architecture_interaction":
                architecture_contrast["learned"]
                - architecture_contrast["exact"],
            "H_by_solver_interaction":
                solver_contrast["learned"] - solver_contrast["exact"],
            "learned_minus_exact": h_contrast,
        }
    rng = np.random.default_rng(12012)
    result = {
        "cluster_unit": "paired monitor checkpoint seed",
        "n_clusters": len(seeds),
        "bootstrap_draws": n_draws,
        "bootstrap_seed": 12012,
        "estimands": {},
    }
    for estimand in [
        "H_by_architecture_interaction",
        "H_by_solver_interaction",
        "learned_minus_exact",
    ]:
        vector = np.array([per_seed[seed][estimand] for seed in seeds])
        draws = np.empty(n_draws)
        for draw in range(n_draws):
            indices = rng.integers(0, len(seeds), len(seeds))
            draws[draw] = np.mean(vector[indices])
        result["estimands"][estimand] = {
            "mean": float(np.mean(vector)),
            "checkpoint_standard_deviation": float(np.std(vector, ddof=1)),
            "checkpoint_bootstrap_95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "per_seed": {
                str(seed): float(value)
                for seed, value in zip(seeds, vector)
            },
        }
    return result


def symmetry_strata(records):
    result = {}
    for h_mode in ["exact", "learned"]:
        for architecture in ["raw", "invariant", "manual_r"]:
            for solver in ["fixed_point", "monotone_residual_newton"]:
                for rotation in [0.0, math.pi / 2.0]:
                    for direction in ["forward", "reversed"]:
                        key = (
                            f"{h_mode}|{architecture}|{solver}|"
                            f"{rotation}|{direction}"
                        )
                        result[key] = completion(
                            selected(
                                records,
                                h_mode=h_mode,
                                architecture=architecture,
                                solver=solver,
                                orientation_radians=rotation,
                                momentum_direction=direction,
                            )
                        )
    return result


def newton_recovery(records):
    index = {
        (
            record["h_mode"],
            record["architecture"],
            record["checkpoint_seed"],
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
            record["solver"],
        ): record
        for record in records
    }
    groups = {}
    for record in records:
        if record["solver"] != "fixed_point":
            continue
        prefix = (
            record["h_mode"],
            record["architecture"],
            record["checkpoint_seed"],
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
        )
        newton = index[prefix + ("monotone_residual_newton",)]
        key = f"{record['h_mode']}|{record['architecture']}"
        group = groups.setdefault(
            key,
            {
                "fixed_point_failures": 0,
                "recovered_by_newton": 0,
                "newton_losses_of_fixed_point_success": 0,
            },
        )
        fixed_complete = record["completion_status"] == "complete"
        newton_complete = newton["completion_status"] == "complete"
        group["fixed_point_failures"] += not fixed_complete
        group["recovered_by_newton"] += (not fixed_complete and newton_complete)
        group["newton_losses_of_fixed_point_success"] += (
            fixed_complete and not newton_complete
        )
    return groups


def survivor_errors(records):
    fields = [
        "max_exact_energy_error",
        "max_mode_energy_error",
        "max_angular_momentum_error",
        "final_state_closure_error",
        "steps",
        "wall_seconds",
    ]
    result = {}
    for h_mode in ["exact", "learned"]:
        for architecture in ["raw", "invariant", "manual_r"]:
            for solver in ["fixed_point", "monotone_residual_newton"]:
                complete = [
                    record
                    for record in selected(
                        records,
                        h_mode=h_mode,
                        architecture=architecture,
                        solver=solver,
                    )
                    if record["completion_status"] == "complete"
                ]
                result[f"{h_mode}|{architecture}|{solver}"] = {
                    "survivor_conditioned": True,
                    "n_survivors": len(complete),
                    "median": {
                        field: (
                            float(np.median([record[field] for record in complete]))
                            if complete else None
                        )
                        for field in fields
                    },
                }
    return result


def make_figure(analysis):
    modes = ["exact", "learned"]
    architectures = ["raw", "invariant", "manual_r"]
    solvers = ["fixed_point", "monotone_residual_newton"]
    colors = {
        "raw": "#C44E52",
        "invariant": "#4C72B0",
        "manual_r": "#55A868",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    for mode_index, mode in enumerate(modes):
        for architecture_index, architecture in enumerate(architectures):
            values = [
                analysis["completion"][
                    f"{mode}|{architecture}|{solver}"
                ]["fraction"]
                for solver in solvers
            ]
            x = np.arange(2) + (architecture_index - 1) * 0.08
            axes[mode_index].plot(
                x,
                values,
                marker="o",
                color=colors[architecture],
                label=("Manual $r$" if architecture == "manual_r" else architecture),
            )
        axes[mode_index].set_xticks([0, 1], ["Fixed point", "Newton"])
        axes[mode_index].set_ylim(-0.04, 1.04)
        axes[mode_index].set_ylabel("Completion fraction")
        axes[mode_index].set_title(
            f"({chr(97 + mode_index)}) {mode.capitalize()} H"
        )
    axes[0].legend(frameon=False, fontsize=8)
    estimands = analysis["checkpoint_cluster_uncertainty"]["estimands"]
    names = [
        "H×architecture",
        "H×solver",
        "Learned−exact",
    ]
    keys = [
        "H_by_architecture_interaction",
        "H_by_solver_interaction",
        "learned_minus_exact",
    ]
    means = [estimands[key]["mean"] for key in keys]
    intervals = [estimands[key]["checkpoint_bootstrap_95"] for key in keys]
    axes[2].errorbar(
        means,
        np.arange(3),
        xerr=[
            [mean - interval[0] for mean, interval in zip(means, intervals)],
            [interval[1] - mean for mean, interval in zip(means, intervals)],
        ],
        fmt="o",
        color="black",
        capsize=4,
    )
    axes[2].axvline(0.0, color="0.5", linestyle=":")
    axes[2].set_yticks(np.arange(3), names)
    axes[2].set_xlabel("Completion contrast")
    axes[2].set_title("(c) Checkpoint-cluster estimates")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    raw = load(RAW_PATH)
    summary = load(SUMMARY_PATH)
    training = load(TRAINING_PATH)
    records = raw["records"]
    completion_by_arm = {}
    for h_mode in ["exact", "learned"]:
        for architecture in ["raw", "invariant", "manual_r"]:
            for solver in ["fixed_point", "monotone_residual_newton"]:
                completion_by_arm[f"{h_mode}|{architecture}|{solver}"] = (
                    completion(
                        selected(
                            records,
                            h_mode=h_mode,
                            architecture=architecture,
                            solver=solver,
                        )
                    )
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
        "completion": completion_by_arm,
        "checkpoint_completion": checkpoint_fractions(records),
        "checkpoint_cluster_uncertainty": cluster_estimands(records),
        "symmetry_stratification": symmetry_strata(records),
        "newton_recovery": newton_recovery(records),
        "survivor_conditioned_errors": survivor_errors(records),
        "failure_reasons": dict(
            Counter(
                record["completion_status"]
                for record in records
                if record["completion_status"] != "complete"
            )
        ),
        "training": {
            "n_records": len(training["records"]),
            "n_completed_checkpoints": sum(
                record["status"] != "training_failure"
                for record in training["records"]
            ),
        },
        "artifact_gate": summary["acceptance"],
        "interpretation_limits": [
            "One learned-H checkpoint is shared across all monitor seeds.",
            "Interactions do not estimate learned-H population uncertainty.",
            "Checkpoint-cluster uncertainty uses only five paired seeds.",
            "Errors and costs are survivor-conditioned secondary estimands.",
        ],
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(analysis)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
