"""Analyse the sealed AC20 deployment-domain / Sundman-factored monitor study.

Reads the per-checkpoint part artifacts under ``results/expAC20/parts``,
the frozen AC9 baseline (``results/expAC9/raw_results.json``), and writes
``results/expAC20/analysis.json`` plus ``fig_deployment_domain_training``.
All estimands are the ones registered in ``preregistration.json``; every
additional quantity is labelled descriptive.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC20")
PARTS = os.path.join(OUT, "parts")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
AC9_RAW = os.path.join(ROOT, "results", "expAC9", "raw_results.json")
FIGURE_STEM = "fig_deployment_domain_training"
ARMS = ("invariant_e099", "sundman_e09", "sundman_e099")
SEEDS = (7101, 7102, 7103, 7104, 7105)
ECCENTRICITIES = (0.7, 0.9, 0.99)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload, attempts=60, delay_seconds=0.5):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def quantiles(values):
    values = np.asarray([float(v) for v in values], dtype=np.float64)
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "minimum": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
    }


def geometric(values):
    values = [float(v) for v in values if v > 0 and math.isfinite(v)]
    if not values:
        return None
    return {**quantiles(values),
            "geometric_mean": float(math.exp(np.mean(np.log(values))))}


def scenario_id(record):
    return (
        record["semi_major_axis"], record["eccentricity"],
        record["initial_mean_anomaly"], record["orientation_radians"],
        record["momentum_direction"],
    )


def family_id(record):
    return (
        record["semi_major_axis"], record["eccentricity"],
        record["orientation_radians"], record["momentum_direction"],
    )


def completion_block(records):
    complete = [r for r in records if r["completion_status"] == "complete"]
    block = {
        "n": len(records),
        "complete": len(complete),
        "fraction": len(complete) / len(records) if records else None,
        "wilson_95": wilson(len(complete), len(records)),
        "status_counts": {
            status: sum(r["completion_status"] == status for r in records)
            for status in sorted({r["completion_status"] for r in records})
        },
        "by_eccentricity": {},
        "by_semi_major_axis": {},
    }
    for e in ECCENTRICITIES:
        subset = [r for r in records if r["eccentricity"] == e]
        block["by_eccentricity"][str(e)] = {
            "n": len(subset),
            "complete": sum(r["completion_status"] == "complete"
                            for r in subset),
        }
    for a in sorted({r["semi_major_axis"] for r in records}):
        subset = [r for r in records if r["semi_major_axis"] == a]
        block["by_semi_major_axis"][str(a)] = {
            "n": len(subset),
            "complete": sum(r["completion_status"] == "complete"
                            for r in subset),
        }
    return block


def per_checkpoint(records, seeds, eccentricity=None):
    out = {}
    for seed in seeds:
        subset = [
            r for r in records if r["checkpoint_seed"] == seed
            and (eccentricity is None or r["eccentricity"] == eccentricity)
        ]
        out[str(seed)] = {
            "n": len(subset),
            "complete": sum(r["completion_status"] == "complete"
                            for r in subset),
        }
    return out


def paired_bootstrap(arm_by_seed, base_by_seed, seeds, draws=5000, seed=20260926):
    rng = np.random.default_rng(seed)
    diffs = np.array([
        arm_by_seed[str(s)]["complete"] / arm_by_seed[str(s)]["n"]
        - base_by_seed[str(s)]["complete"] / base_by_seed[str(s)]["n"]
        for s in seeds
    ])
    samples = np.array([
        diffs[rng.integers(0, len(seeds), len(seeds))].mean()
        for _ in range(draws)
    ])
    return {
        "paired_checkpoint_differences": diffs.tolist(),
        "mean_difference": float(diffs.mean()),
        "bootstrap_95": [float(np.quantile(samples, 0.025)),
                         float(np.quantile(samples, 0.975))],
        "draws": draws,
    }


def confusion(rows, threshold):
    tp = sum(1 for rho, failed in rows if rho >= threshold and failed)
    fn = sum(1 for rho, failed in rows if rho < threshold and failed)
    fp = sum(1 for rho, failed in rows if rho >= threshold and not failed)
    tn = sum(1 for rho, failed in rows if rho < threshold and not failed)
    total = len(rows)
    return {
        "threshold": threshold, "n": total,
        "predicted_fail_and_failed": tp,
        "predicted_complete_but_failed": fn,
        "predicted_fail_but_completed": fp,
        "predicted_complete_and_completed": tn,
        "accuracy": (tp + tn) / total if total else None,
        "accuracy_wilson_95": wilson(tp + tn, total),
        "sensitivity_for_failure": tp / (tp + fn) if tp + fn else None,
        "specificity_for_completion": tn / (tn + fp) if tn + fp else None,
    }


def ratio_block(records, manual_by_scenario):
    completed = [r for r in records if r["completion_status"] == "complete"]
    closure, energy, cost = [], [], []
    for r in completed:
        manual = manual_by_scenario[scenario_id(r)]
        if manual["completion_status"] != "complete":
            continue
        closure.append(
            r["final_state_closure_error"] / manual["final_state_closure_error"]
        )
        energy.append(r["max_energy_error"] / manual["max_energy_error"])
        cost.append(
            r["vector_field_evaluations"] / manual["vector_field_evaluations"]
        )
    return {
        "completed_pairs": len(closure),
        "final_state_closure_error_over_manual": geometric(closure),
        "max_energy_error_over_manual": geometric(energy),
        "vector_field_evaluations_over_manual": geometric(cost),
        "final_state_closure_error": quantiles(
            [r["final_state_closure_error"] for r in completed]
        ),
        "max_energy_error": quantiles(
            [r["max_energy_error"] for r in completed]
        ),
        "vector_field_evaluations": quantiles(
            [r["vector_field_evaluations"] for r in completed]
        ),
        "note": "survivor-conditioned; manual g=r denominators are the "
                "frozen AC9 dtau=0.04 rollouts of the same scenario",
    }


def main():
    protocol = load(PROTOCOL_PATH)
    ac9 = load(AC9_RAW)
    threshold = protocol["diagnostic"]["contraction_threshold"]

    # ---- gather parts ----
    training, diagnostics, rollouts, ordering_ok = {}, {}, {}, []
    part_hashes = {}
    for arm in ARMS:
        for seed in SEEDS:
            directory = os.path.join(PARTS, f"{arm}_seed{seed}")
            paths = {
                name: os.path.join(directory, f"{name}.json")
                for name in ("training", "diagnostic", "rollouts")
            }
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            training[(arm, seed)] = load(paths["training"])["record"]
            diag = load(paths["diagnostic"])
            roll = load(paths["rollouts"])
            for path in paths.values():
                part_hashes[os.path.relpath(path, ROOT).replace("\\", "/")] = \
                    sha256_file(path)
            if "completed_at_utc" not in roll:
                continue
            diagnostics[(arm, seed)] = diag["orbit_records"]
            rollouts[(arm, seed)] = roll["records"]
            ordering_ok.append(diag["completed_at_utc"] <= roll["started_at_utc"])

    arm_rollouts = {
        arm: [r for (a, s), rs in rollouts.items() if a == arm for r in rs]
        for arm in ARMS
    }
    arm_diagnostics = {
        arm: [r for (a, s), rs in diagnostics.items() if a == arm for r in rs]
        for arm in ARMS
    }
    baseline = {
        name: [r for r in ac9["records"] if r["architecture"] == name]
        for name in ("invariant", "raw", "manual_r")
    }
    manual_by_scenario = {scenario_id(r): r for r in baseline["manual_r"]}

    # ---- training summary ----
    training_summary = {}
    for arm in ARMS:
        records = [training[(arm, s)] for s in SEEDS if (arm, s) in training]
        training_summary[arm] = {
            "checkpoints": len(records),
            "best_validation_state_mse": quantiles(
                [r["best_validation_state_mse"] for r in records]
            ),
            "best_epochs": [r["best_epoch"] for r in records],
            "nonfinite_epochs_skipped_total": sum(
                r.get("nonfinite_epochs_skipped", 0) for r in records
            ),
            "nonfinite_epochs_skipped": [
                r.get("nonfinite_epochs_skipped", 0) for r in records
            ],
            "maximum_rotation_relative_error": max(
                (r["symmetry"]["maximum_rotation_relative_error"]
                 for r in records), default=None
            ),
            "maximum_momentum_reversal_relative_error": max(
                (r["symmetry"]["maximum_momentum_reversal_relative_error"]
                 for r in records), default=None
            ),
            "wall_seconds": quantiles(
                [r["wall_seconds"] for r in records if "wall_seconds" in r]
            ),
        }

    # ---- completion ----
    completion = {
        "ac9_invariant": completion_block(baseline["invariant"]),
        "ac9_raw": completion_block(baseline["raw"]),
        "ac9_manual_r": completion_block(baseline["manual_r"]),
    }
    per_ckpt = {
        "ac9_invariant": per_checkpoint(baseline["invariant"], SEEDS),
        "ac9_invariant_e099": per_checkpoint(baseline["invariant"], SEEDS, 0.99),
    }
    paired = {}
    for arm in ARMS:
        completion[arm] = completion_block(arm_rollouts[arm])
        per_ckpt[arm] = per_checkpoint(arm_rollouts[arm], SEEDS)
        per_ckpt[f"{arm}_e099"] = per_checkpoint(arm_rollouts[arm], SEEDS, 0.99)
        if len(arm_rollouts[arm]) == 360:
            paired[arm] = {
                "all_scenarios_minus_ac9_invariant": paired_bootstrap(
                    per_ckpt[arm], per_ckpt["ac9_invariant"], SEEDS
                ),
                "e099_minus_ac9_invariant": paired_bootstrap(
                    per_ckpt[f"{arm}_e099"], per_ckpt["ac9_invariant_e099"],
                    SEEDS,
                ),
            }

    # ---- monitor structure and prospective prediction ----
    structure, prediction = {}, {}
    pooled_rows = []
    for arm in ARMS:
        rho_by_family = {
            (r["checkpoint_seed"], *family_id(r)): r
            for r in arm_diagnostics[arm]
        }
        structure[arm] = {}
        for e in ECCENTRICITIES:
            subset = [r for r in arm_diagnostics[arm] if r["eccentricity"] == e]
            structure[arm][str(e)] = {
                "orbits": len(subset),
                "spectral_radius_maximum": quantiles(
                    [r["spectral_radius_maximum"] for r in subset]
                ),
                "monitor_over_radius_at_periapsis": quantiles(
                    [r["monitor_over_radius_at_periapsis"] for r in subset]
                ),
                "monitor_at_periapsis": quantiles(
                    [r["monitor_at_periapsis"] for r in subset]
                ),
                "orbits_predicted_to_fail": sum(
                    r["spectral_radius_maximum"] >= threshold for r in subset
                ),
            }
        rows = []
        for r in arm_rollouts[arm]:
            family = rho_by_family.get((r["checkpoint_seed"], *family_id(r)))
            if family is None:
                continue
            rows.append((family["spectral_radius_maximum"],
                         r["completion_status"] != "complete"))
        pooled_rows.extend(rows)
        prediction[arm] = confusion(rows, threshold)
    prediction["pooled"] = confusion(pooled_rows, threshold)
    prediction["diagnostic_completed_before_rollouts_started"] = (
        bool(ordering_ok) and all(ordering_ok)
    )

    # ---- accuracy / cost versus manual ----
    accuracy_cost = {}
    for arm in ARMS:
        accuracy_cost[arm] = {
            "all": ratio_block(arm_rollouts[arm], manual_by_scenario),
            "by_eccentricity": {
                str(e): ratio_block(
                    [r for r in arm_rollouts[arm] if r["eccentricity"] == e],
                    manual_by_scenario,
                )
                for e in ECCENTRICITIES
            },
        }
    accuracy_cost["ac9_invariant"] = {
        "all": ratio_block(baseline["invariant"], manual_by_scenario),
        "by_eccentricity": {
            str(e): ratio_block(
                [r for r in baseline["invariant"] if r["eccentricity"] == e],
                manual_by_scenario,
            )
            for e in ECCENTRICITIES
        },
    }
    accuracy_cost["ac9_manual_r_absolute"] = {
        str(e): {
            "final_state_closure_error": quantiles(
                [r["final_state_closure_error"] for r in baseline["manual_r"]
                 if r["eccentricity"] == e]
            ),
            "vector_field_evaluations": quantiles(
                [r["vector_field_evaluations"] for r in baseline["manual_r"]
                 if r["eccentricity"] == e]
            ),
        }
        for e in ECCENTRICITIES
    }

    counts = {
        "checkpoints": len(training),
        "orbit_records": sum(len(v) for v in diagnostics.values()),
        "rollouts": sum(len(v) for v in rollouts.values()),
        "rollouts_by_arm": {arm: len(arm_rollouts[arm]) for arm in ARMS},
    }
    expected = protocol["acceptance"]
    acceptance = {
        "all_checkpoints_trained": counts["checkpoints"]
        == expected["expected_checkpoints"],
        "all_orbit_diagnostics": counts["orbit_records"]
        == expected["expected_orbit_records"],
        "all_rollouts_retained": counts["rollouts"]
        == expected["expected_rollouts"],
        "diagnostic_precedes_rollouts":
            prediction["diagnostic_completed_before_rollouts_started"],
        "completed_final_time_error_below_1e-10": all(
            r["final_time_error"] < 1e-10
            for rs in rollouts.values() for r in rs
            if r["completion_status"] == "complete"
        ),
        "completed_residual_below_1e-11": all(
            r["maximum_residual"] < 1e-11
            for rs in rollouts.values() for r in rs
            if r["completion_status"] == "complete"
        ),
        "scientific_success_not_required": True,
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())

    analysis = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "ac9_raw_sha256": sha256_file(AC9_RAW),
        "part_sha256": part_hashes,
        "counts": counts,
        "training": training_summary,
        "completion": completion,
        "per_checkpoint": per_ckpt,
        "paired_checkpoint_contrasts": paired,
        "monitor_structure": structure,
        "prospective_prediction": prediction,
        "accuracy_cost": accuracy_cost,
        "claim_limits": protocol["claim_limits"],
        "acceptance": acceptance,
    }
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(analysis, arm_rollouts, arm_diagnostics, baseline, threshold)
    print(json.dumps(
        {
            "completion": {
                k: {"complete": v["complete"], "n": v["n"],
                    "e099": v["by_eccentricity"]["0.99"]}
                for k, v in completion.items()
            },
            "prediction_pooled": prediction["pooled"],
            "acceptance": acceptance,
        },
        indent=1,
    ))
    if not acceptance["artifact_gate_passed"]:
        raise SystemExit("AC20 artifact gate failed")


def make_figure(analysis, arm_rollouts, arm_diagnostics, baseline, threshold):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "legend.fontsize": 6.5})
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    groups = [
        ("ac9_raw", "AC9 raw", "#E15759"),
        ("ac9_invariant", "AC9 invariant", "#4E79A7"),
        ("invariant_e099", "invariant, trained e=0.99", "#76B7B2"),
        ("sundman_e09", "Sundman-factored, e=0.9", "#59A14F"),
        ("sundman_e099", "Sundman-factored, e=0.99", "#8CD17D"),
        ("ac9_manual_r", "manual g=r", "#F28E2B"),
    ]
    width = 0.13
    for index, (key, label, colour) in enumerate(groups):
        block = analysis["completion"].get(key)
        if block is None or block["n"] == 0:
            continue
        fractions = [
            block["by_eccentricity"][str(e)]["complete"]
            / max(block["by_eccentricity"][str(e)]["n"], 1)
            for e in ECCENTRICITIES
        ]
        axes[0].bar(
            np.arange(3) + (index - 2.5) * width, fractions, width,
            color=colour, label=label,
        )
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels([f"$e={e}$" for e in ECCENTRICITIES])
    axes[0].set(ylabel="Completion fraction", ylim=(0, 1.05))
    axes[0].set_title("(a) Completion on the AC9 grid")
    axes[0].legend(frameon=False, loc="lower left", ncol=1)

    colours = {"invariant_e099": "#76B7B2", "sundman_e09": "#59A14F",
               "sundman_e099": "#8CD17D"}
    rng = np.random.default_rng(20260926)
    plotted = set()
    for arm_index, arm in enumerate(ARMS):
        rho_by_family = {
            (r["checkpoint_seed"], *family_id(r)): r["spectral_radius_maximum"]
            for r in arm_diagnostics[arm]
        }
        for r in arm_rollouts[arm]:
            rho = rho_by_family.get((r["checkpoint_seed"], *family_id(r)))
            if rho is None:
                continue
            failed = r["completion_status"] != "complete"
            key = "failed" if failed else "complete"
            label = None
            if key not in plotted:
                label = "solver failure" if failed else "complete"
                plotted.add(key)
            position = arm_index * 4 + ECCENTRICITIES.index(r["eccentricity"])
            axes[1].scatter(
                position + rng.uniform(-0.3, 0.3), rho,
                marker="x" if failed else "o", s=12, alpha=0.6,
                color=colours[arm], linewidths=0.8, label=label,
            )
    axes[1].axhline(threshold, color="0.3", linestyle="--", linewidth=1)
    ticks = [a * 4 + i for a in range(3) for i in range(3)]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f"{e}" for _ in ARMS for e in ECCENTRICITIES],
                            fontsize=6)
    for arm_index, arm in enumerate(ARMS):
        axes[1].text(arm_index * 4 + 1, -0.14, arm.replace("_", " "),
                     ha="center", va="top", fontsize=6.5,
                     transform=axes[1].get_xaxis_transform())
    axes[1].set(yscale="log",
                ylabel=r"prospective $\rho_{\max}$ at $\Delta\tau=0.04$")
    axes[1].set_xlabel("eccentricity, grouped by arm", labelpad=20)
    axes[1].set_title("(b) Prospective contraction versus outcome")
    axes[1].legend(frameon=False, loc="upper right")

    for arm in ARMS:
        completed = [r for r in arm_rollouts[arm]
                     if r["completion_status"] == "complete"
                     and r["eccentricity"] == 0.99]
        if completed:
            axes[2].scatter(
                [r["vector_field_evaluations"] for r in completed],
                [r["final_state_closure_error"] for r in completed],
                s=12, alpha=0.7, color=colours[arm],
                label=arm.replace("_", " "),
            )
    manual = [r for r in baseline["manual_r"] if r["eccentricity"] == 0.99
              and r["completion_status"] == "complete"]
    axes[2].scatter(
        [r["vector_field_evaluations"] for r in manual],
        [r["final_state_closure_error"] for r in manual],
        s=18, marker="s", color="#F28E2B", label="manual $g=r$",
    )
    axes[2].set(xscale="log", yscale="log",
                xlabel="vector-field evaluations (one period)",
                ylabel="final-state closure error")
    axes[2].set_title("(c) Completed $e=0.99$ rollouts")
    axes[2].legend(frameon=False, loc="best")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        for directory in (ROOT, OUT):
            figure.savefig(
                os.path.join(directory, f"{FIGURE_STEM}.{extension}"),
                dpi=300, bbox_inches="tight",
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
