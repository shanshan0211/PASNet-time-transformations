"""Analyze the sealed AC8 monitor-architecture/solver factorial."""

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
OUT = os.path.join(ROOT, "results", "expAC8")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
FIGURE_PDF = os.path.join(ROOT, "fig_monitor_solver_factorial.pdf")
FIGURE_PNG = os.path.join(ROOT, "fig_monitor_solver_factorial.png")


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


def wilson_interval(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [centre - half_width, centre + half_width]


def geometric_mean(values):
    return float(math.exp(np.mean(np.log(values))))


def bootstrap_completion_difference(records, solver, n_bootstrap=10000):
    seeds = sorted(
        {
            record["seed"]
            for record in records
            if record["architecture"] in {"raw", "invariant"}
        }
    )
    by_seed = {}
    for seed in seeds:
        by_seed[seed] = {}
        for architecture in ["raw", "invariant"]:
            selected = [
                record
                for record in records
                if record["seed"] == seed
                and record["architecture"] == architecture
                and record["solver"] == solver
            ]
            by_seed[seed][architecture] = np.mean(
                [
                    record["completion_status"] == "complete"
                    for record in selected
                ]
            )
    observed = float(
        np.mean(
            [
                by_seed[seed]["invariant"] - by_seed[seed]["raw"]
                for seed in seeds
            ]
        )
    )
    rng = np.random.default_rng(8608)
    draws = []
    for _ in range(n_bootstrap):
        sample = rng.choice(seeds, size=len(seeds), replace=True)
        draws.append(
            np.mean(
                [
                    by_seed[int(seed)]["invariant"]
                    - by_seed[int(seed)]["raw"]
                    for seed in sample
                ]
            )
        )
    return {
        "estimand":
            "invariant minus raw completion fraction; checkpoint-cluster "
            "bootstrap",
        "n_checkpoint_seeds": len(seeds),
        "observed_difference": observed,
        "bootstrap_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_draws": n_bootstrap,
        "bootstrap_seed": 8608,
    }


def transition_table(records, architecture):
    index = {}
    for record in records:
        if record["architecture"] == architecture:
            key = (
                record["eccentricity"],
                record["orientation_radians"],
                record["momentum_direction"],
                record["seed"],
            )
            index.setdefault(key, {})[record["solver"]] = (
                record["completion_status"] == "complete"
            )
    counts = Counter()
    for solvers in index.values():
        fixed = solvers["fixed_point"]
        newton = solvers["damped_newton"]
        if fixed and newton:
            counts["both_complete"] += 1
        elif not fixed and newton:
            counts["newton_only"] += 1
        elif fixed and not newton:
            counts["fixed_point_only"] += 1
        else:
            counts["both_fail"] += 1
    return dict(counts)


def symmetry_groups(records, architecture, solver):
    groups = {}
    for record in records:
        if (
            record["architecture"] == architecture
            and record["solver"] == solver
        ):
            key = (record["eccentricity"], record["seed"])
            groups.setdefault(key, []).append(record)
    complete_groups = []
    completion_uniform_groups = 0
    for group in groups.values():
        statuses = [
            record["completion_status"] == "complete" for record in group
        ]
        if all(status == statuses[0] for status in statuses):
            completion_uniform_groups += 1
        if all(statuses):
            energy = [record["max_energy_error"] for record in group]
            closure = [
                record["final_state_closure_error"] for record in group
            ]
            complete_groups.append(
                {
                    "energy_max_min_ratio": max(energy) / min(energy),
                    "closure_max_min_ratio": max(closure) / min(closure),
                }
            )
    return {
        "registered_groups": len(groups),
        "completion_uniform_groups": completion_uniform_groups,
        "all_four_complete_groups": len(complete_groups),
        "maximum_energy_max_min_ratio": max(
            (
                group["energy_max_min_ratio"]
                for group in complete_groups
            ),
            default=None,
        ),
        "maximum_closure_max_min_ratio": max(
            (
                group["closure_max_min_ratio"]
                for group in complete_groups
            ),
            default=None,
        ),
    }


def failure_reasons(records):
    result = {}
    for architecture in ["raw", "invariant", "manual_r"]:
        for solver in ["fixed_point", "damped_newton"]:
            selected = [
                record
                for record in records
                if record["architecture"] == architecture
                and record["solver"] == solver
                and record["completion_status"] != "complete"
            ]
            reasons = Counter()
            for record in selected:
                error = record.get("error", "missing error")
                if "line search failed" in error:
                    reason = "newton_line_search"
                elif "Newton did not converge" in error:
                    reason = "newton_iteration_cap"
                elif "fixed point did not converge" in error:
                    reason = "fixed_point_iteration_cap"
                elif "singular Newton Jacobian" in error:
                    reason = "singular_newton_jacobian"
                else:
                    reason = record["completion_status"]
                reasons[reason] += 1
            result[f"{architecture}|{solver}"] = dict(reasons)
    return result


def make_figure(analysis):
    colors = {"raw": "#C44E52", "invariant": "#4C72B0", "manual_r": "#55A868"}
    solvers = ["fixed_point", "damped_newton"]
    solver_labels = ["Fixed point", "Damped Newton"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))

    architectures = ["raw", "invariant", "manual_r"]
    x = np.arange(len(architectures))
    width = 0.34
    for offset, (solver, label) in zip([-width / 2, width / 2], zip(solvers, solver_labels)):
        values = [
            analysis["completion"][f"{architecture}|{solver}"]["fraction"]
            for architecture in architectures
        ]
        intervals = [
            analysis["completion"][f"{architecture}|{solver}"]["wilson_95"]
            for architecture in architectures
        ]
        yerr = np.array(
            [
                [value - interval[0] for value, interval in zip(values, intervals)],
                [interval[1] - value for value, interval in zip(values, intervals)],
            ]
        )
        axes[0].bar(
            x + offset,
            values,
            width,
            yerr=yerr,
            capsize=3,
            label=label,
            color=[colors[architecture] for architecture in architectures],
            alpha=0.72 if solver == "fixed_point" else 1.0,
            edgecolor="black",
            linewidth=0.5,
        )
    axes[0].set_xticks(x, ["Raw", "Invariant", r"Manual $r$"])
    axes[0].set_ylim(0, 1.13)
    axes[0].set_ylabel("Completion fraction")
    axes[0].set_title("(a) Architecture × solver")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    eccentricities = analysis["eccentricities"]
    for architecture, marker in [("raw", "o"), ("invariant", "s")]:
        for solver, linestyle in zip(solvers, ["--", "-"]):
            values = [
                analysis["completion_by_eccentricity"][
                    f"{architecture}|{solver}|{eccentricity}"
                ]["fraction"]
                for eccentricity in eccentricities
            ]
            axes[1].plot(
                eccentricities,
                values,
                marker=marker,
                linestyle=linestyle,
                color=colors[architecture],
                label=(
                    f"{architecture.capitalize()}, "
                    f"{'FP' if solver == 'fixed_point' else 'Newton'}"
                ),
            )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Eccentricity")
    axes[1].set_ylabel("Completion fraction")
    axes[1].set_title("(b) Unseen-orbit boundary")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")

    transition_labels = [
        "Both\ncomplete",
        "Newton\nonly",
        "FP\nonly",
        "Both\nfail",
    ]
    transition_keys = [
        "both_complete",
        "newton_only",
        "fixed_point_only",
        "both_fail",
    ]
    tx = np.arange(len(transition_keys))
    for offset, architecture in zip([-width / 2, width / 2], ["raw", "invariant"]):
        transition = analysis["solver_transitions"][architecture]
        axes[2].bar(
            tx + offset,
            [transition.get(key, 0) for key in transition_keys],
            width,
            color=colors[architecture],
            label=architecture.capitalize(),
        )
    axes[2].set_xticks(tx, transition_labels)
    axes[2].set_ylabel("Paired scenarios (n=60)")
    axes[2].set_title("(c) Solver recovery")
    axes[2].legend(frameon=False, fontsize=8)

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
    completion = {}
    completion_by_eccentricity = {}
    eccentricities = sorted(
        {record["eccentricity"] for record in records}
    )
    for architecture in ["raw", "invariant", "manual_r"]:
        for solver in ["fixed_point", "damped_newton"]:
            selected = [
                record
                for record in records
                if record["architecture"] == architecture
                and record["solver"] == solver
            ]
            complete = sum(
                record["completion_status"] == "complete"
                for record in selected
            )
            completion[f"{architecture}|{solver}"] = {
                "complete": complete,
                "registered": len(selected),
                "fraction": complete / len(selected),
                "wilson_95": wilson_interval(complete, len(selected)),
            }
            for eccentricity in eccentricities:
                stratum = [
                    record
                    for record in selected
                    if record["eccentricity"] == eccentricity
                ]
                stratum_complete = sum(
                    record["completion_status"] == "complete"
                    for record in stratum
                )
                completion_by_eccentricity[
                    f"{architecture}|{solver}|{eccentricity}"
                ] = {
                    "complete": stratum_complete,
                    "registered": len(stratum),
                    "fraction": stratum_complete / len(stratum),
                    "wilson_95": wilson_interval(
                        stratum_complete, len(stratum)
                    ),
                }
    analysis = {
        "schema_version": "1.0",
        "experiment": raw["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": raw["protocol_sha256"],
        "source_sha256": {
            "raw_results": sha256_file(RAW_PATH),
            "summary": sha256_file(SUMMARY_PATH),
            "training": sha256_file(TRAINING_PATH),
            "analysis_script": sha256_file(__file__),
        },
        "eccentricities": eccentricities,
        "completion": completion,
        "completion_by_eccentricity": completion_by_eccentricity,
        "cluster_bootstrap_completion_difference": {
            solver: bootstrap_completion_difference(records, solver)
            for solver in ["fixed_point", "damped_newton"]
        },
        "solver_transitions": {
            architecture: transition_table(records, architecture)
            for architecture in ["raw", "invariant"]
        },
        "symmetry_groups": {
            f"{architecture}|{solver}": symmetry_groups(
                records, architecture, solver
            )
            for architecture in ["raw", "invariant", "manual_r"]
            for solver in ["fixed_point", "damped_newton"]
        },
        "failure_reasons": failure_reasons(records),
        "completed_pair_error_ratios": summary[
            "paired_invariant_over_raw"
        ],
        "training_fit": {
            architecture: {
                "validation_mse_range": [
                    min(
                        record["validation_mse"]
                        for record in training["records"]
                        if record["architecture"] == architecture
                    ),
                    max(
                        record["validation_mse"]
                        for record in training["records"]
                        if record["architecture"] == architecture
                    ),
                ],
                "n_checkpoints": 5,
            }
            for architecture in ["raw", "invariant"]
        },
        "interpretation_limits": [
            "Completion Wilson intervals treat registered rollouts as "
            "analysis units; they are not checkpoint-population intervals.",
            "Checkpoint-cluster bootstrap has only five optimization seeds.",
            "Error ratios condition on both members completing and are "
            "therefore survivor-conditioned.",
            "Students distill one frozen monitor on one canonical orbit; "
            "this is not end-to-end retraining of the adaptive objective.",
            "The exact-H design isolates monitor/solver behavior but does "
            "not test learned-H interaction.",
        ],
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(analysis)
    print(sha256_file(ANALYSIS_PATH))


if __name__ == "__main__":
    main()
