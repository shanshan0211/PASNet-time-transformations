"""Prospectively specified analysis and figure for sealed AC19.

Inputs: ``results/expAC19/raw_results.json`` (phase A diagnostic),
``results/expAC19/intervention.json`` (phase B), and the frozen AC9
``raw_results.json`` completion labels. Outputs ``analysis.json`` and
``fig_periapsis_mechanism.{pdf,png}``.

All classification statistics use the prespecified contraction threshold
``rho_max >= 1``. The post-hoc separating threshold and the ROC area are
reported as descriptive quantities and labelled post hoc.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from expAC19_periapsis_contraction_diagnostic import (  # noqa: E402
    radius_at_physical_time,
)


def periapsis_passages(scenario, physical_time):
    """Number of periapsis passages of the analytic reference orbit between
    the registered initial state and ``physical_time``."""
    mean_motion = scenario["semi_major_axis"] ** -1.5
    two_pi = 2.0 * math.pi
    start = scenario["initial_mean_anomaly"]
    if scenario["momentum_direction"] == "reversed":
        end = start - mean_motion * physical_time
        return int(math.ceil(start / two_pi) - math.ceil(end / two_pi))
    end = start + mean_motion * physical_time
    return int(math.floor(end / two_pi) - math.floor(start / two_pi))


ROOT = os.path.join(HERE, "..")
# AC19_OUT is only used by the pre-registration dry run; the sealed analysis
# always reads results/expAC19.
OUT = os.environ.get("AC19_OUT", os.path.join(ROOT, "results", "expAC19"))
DRY_RUN = "AC19_OUT" in os.environ
RAW_PATH = os.path.join(OUT, "raw_results.json")
INTERVENTION_PATH = os.path.join(OUT, "intervention.json")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
AC9_RAW = (
    os.path.join(OUT, "ac9_subset.json") if DRY_RUN
    else os.path.join(ROOT, "results", "expAC9", "raw_results.json")
)
FIGURE_ROOT = OUT if DRY_RUN else ROOT
FIGURE_STEM = "fig_periapsis_mechanism"


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
    """Atomic JSON write with retries against transient cloud-sync locks."""
    import time

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
        return [None, None]
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


def family_key(record):
    return (
        record["architecture"],
        record["checkpoint_seed"],
        record["semi_major_axis"],
        record["eccentricity"],
        record["orientation_radians"],
        record["momentum_direction"],
    )


def scenario_key(record):
    return (
        record["architecture"],
        record["checkpoint_seed"],
        record["semi_major_axis"],
        record["eccentricity"],
        record["initial_mean_anomaly"],
        record["orientation_radians"],
        record["momentum_direction"],
    )


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
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


def confusion(rows, threshold):
    """rows: list of (rho_max, failed)."""
    tp = sum(rho >= threshold and failed for rho, failed in rows)
    fn = sum(rho < threshold and failed for rho, failed in rows)
    fp = sum(rho >= threshold and not failed for rho, failed in rows)
    tn = sum(rho < threshold and not failed for rho, failed in rows)
    total = tp + fn + fp + tn
    return {
        "threshold": threshold,
        "n": total,
        "predicted_fail_and_failed": tp,
        "predicted_complete_but_failed": fn,
        "predicted_fail_but_completed": fp,
        "predicted_complete_and_completed": tn,
        "accuracy": (tp + tn) / total if total else None,
        "accuracy_wilson_95": wilson(tp + tn, total),
        "sensitivity_for_failure": tp / (tp + fn) if tp + fn else None,
        "specificity_for_completion": tn / (tn + fp) if tn + fp else None,
    }


def roc_area(rows):
    positives = np.array([rho for rho, failed in rows if failed])
    negatives = np.array([rho for rho, failed in rows if not failed])
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        ((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
        / (positives.size * negatives.size)
    )


def post_hoc_threshold(rows):
    candidates = sorted({rho for rho, _ in rows})
    best = None
    for index in range(len(candidates)):
        threshold = candidates[index]
        result = confusion(rows, threshold)
        if best is None or result["accuracy"] > best["accuracy"]:
            best = result
    if best is not None:
        best["label"] = "post_hoc_maximum_accuracy_threshold"
    return best


def architecture_summary(architecture, records, completed, manual_by_scenario):
    own_completed = [r for r in completed if r["architecture"] == architecture]
    ratios = [
        r["vector_field_evaluations"]
        / manual_by_scenario[scenario_key(r)[2:]]["vector_field_evaluations"]
        for r in own_completed
    ]
    return {
        "rerun": sum(r["architecture"] == architecture for r in records),
        "completed": len(own_completed),
        "completed_vector_field_evaluations_over_manual_dtau004_geometric_mean":
            float(math.exp(np.mean(np.log(ratios)))) if ratios else None,
    }


def main():
    protocol = load(PROTOCOL_PATH)
    raw = load(RAW_PATH)
    intervention = load(INTERVENTION_PATH)
    ac9 = load(AC9_RAW)
    threshold = protocol["phase_a"]["contraction_threshold"]
    training_radius_minimum = protocol["phase_a"]["training_radius_minimum"]

    rho_by_family = {
        family_key(record): record for record in raw["orbit_records"]
    }

    # ---- orbit-level summaries by architecture and eccentricity ----
    orbit_summary = defaultdict(list)
    for record in raw["orbit_records"]:
        orbit_summary[(record["architecture"], record["eccentricity"])].append(
            record
        )
    orbit_table = {}
    for (architecture, eccentricity), records in sorted(
        orbit_summary.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        orbit_table[f"{architecture}|e={eccentricity}"] = {
            "architecture": architecture,
            "eccentricity": eccentricity,
            "orbits": len(records),
            "spectral_radius_maximum": quantiles(
                [r["spectral_radius_maximum"] for r in records]
            ),
            "monitor_over_radius_at_periapsis": quantiles(
                [r["monitor_over_radius_at_periapsis"] for r in records]
            ),
            "monitor_at_periapsis": quantiles(
                [r["monitor_at_periapsis"] for r in records]
            ),
            "orbits_predicted_to_fail": sum(
                r["spectral_radius_maximum"] >= threshold for r in records
            ),
        }

    # ---- rollout-level classification against frozen AC9 labels ----
    rows_all, rows_learned, rows_by_e = [], [], defaultdict(list)
    rollout_rows = []
    for record in ac9["records"]:
        family = rho_by_family[family_key(record)]
        rho = family["spectral_radius_maximum"]
        failed = record["completion_status"] != "complete"
        rows_all.append((rho, failed))
        if record["architecture"] != "manual_r":
            rows_learned.append((rho, failed))
        rows_by_e[record["eccentricity"]].append((rho, failed))
        rollout_rows.append(
            {
                "architecture": record["architecture"],
                "checkpoint_seed": record["checkpoint_seed"],
                "semi_major_axis": record["semi_major_axis"],
                "eccentricity": record["eccentricity"],
                "initial_mean_anomaly": record["initial_mean_anomaly"],
                "orientation_radians": record["orientation_radians"],
                "momentum_direction": record["momentum_direction"],
                "completion_status": record["completion_status"],
                "spectral_radius_maximum": rho,
                "monitor_over_radius_at_periapsis":
                    family["monitor_over_radius_at_periapsis"],
            }
        )
    classification = {
        "prespecified_threshold": threshold,
        "all_792_rollouts": confusion(rows_all, threshold),
        "learned_720_rollouts": confusion(rows_learned, threshold),
        "by_eccentricity": {
            str(e): confusion(rows, threshold)
            for e, rows in sorted(rows_by_e.items())
        },
        "failed_rollouts_spectral_radius_minimum": float(
            min(rho for rho, failed in rows_all if failed)
        ),
        "completed_rollouts_spectral_radius_maximum": float(
            max(rho for rho, failed in rows_all if not failed)
        ),
        "roc_area_all_rollouts": roc_area(rows_all),
        "roc_area_learned_rollouts": roc_area(rows_learned),
        "post_hoc_learned_rollouts": post_hoc_threshold(rows_learned),
        "misclassified_rollouts": [
            row for row in rollout_rows
            if (row["spectral_radius_maximum"] >= threshold)
            != (row["completion_status"] != "complete")
        ],
    }
    # manual and learned rho ranges at e=0.99
    manual_099 = [
        r["spectral_radius_maximum"] for r in raw["orbit_records"]
        if r["architecture"] == "manual_r" and r["eccentricity"] == 0.99
    ]
    learned_099 = [
        r["spectral_radius_maximum"] for r in raw["orbit_records"]
        if r["architecture"] != "manual_r" and r["eccentricity"] == 0.99
    ]
    learned_09 = [
        r["spectral_radius_maximum"] for r in raw["orbit_records"]
        if r["architecture"] != "manual_r" and r["eccentricity"] == 0.9
    ]
    learned_ratio_099 = [
        r["monitor_over_radius_at_periapsis"] for r in raw["orbit_records"]
        if r["architecture"] != "manual_r" and r["eccentricity"] == 0.99
    ]
    manual_all = [
        r["spectral_radius_maximum"] for r in raw["orbit_records"]
        if r["architecture"] == "manual_r"
    ]
    learned_le09_age1 = [
        r["spectral_radius_maximum"] for r in raw["orbit_records"]
        if r["architecture"] != "manual_r" and r["eccentricity"] <= 0.9
        and r["semi_major_axis"] >= 1.0
    ]
    fraction_a1_e099 = sorted({
        r["orbit_time_fraction_below_training_radius"]
        for r in raw["orbit_records"]
        if r["semi_major_axis"] == 1.0 and r["eccentricity"] == 0.99
    })
    separation = {
        "manual_all_orbits_spectral_radius_maximum": quantiles(manual_all),
        "manual_e099_spectral_radius_maximum": quantiles(manual_099),
        "learned_e099_spectral_radius_maximum": quantiles(learned_099),
        "learned_e09_spectral_radius_maximum": quantiles(learned_09),
        "learned_e_at_most_09_a_at_least_1_spectral_radius_maximum":
            quantiles(learned_le09_age1),
        "orbit_time_fraction_below_training_radius_a1_e099":
            fraction_a1_e099[-1],
        "orbit_time_percent_below_training_radius_a1_e099":
            100.0 * fraction_a1_e099[-1],
        "monitor_rho_scaling_ratio_r01_to_r001": {
            "sundman_g_proportional_r_exponent_minus_half":
                float(10.0 ** 0.5),
            "plateaued_monitor_exponent_minus_three_halves":
                float(10.0 ** 1.5),
        },
        "learned_e099_monitor_over_radius_at_periapsis":
            quantiles(learned_ratio_099),
        "learned_e099_orbits_with_rho_at_least_threshold": int(
            sum(value >= threshold for value in learned_099)
        ),
        "learned_e099_orbits": len(learned_099),
        "manual_all_orbits_below_threshold": bool(
            all(
                r["spectral_radius_maximum"] < threshold
                for r in raw["orbit_records"]
                if r["architecture"] == "manual_r"
            )
        ),
    }

    # ---- failure locations ----
    locations = raw["failure_locations"]
    ratios = [loc["failure_radius_over_periapsis"] for loc in locations]
    radii = [loc["failure_radius"] for loc in locations]
    failure_location = {
        "n_failed_rollouts": len(locations),
        "failure_radius_over_periapsis": quantiles(ratios),
        "failure_radius": quantiles(radii),
        "within_2x_periapsis": int(sum(value <= 2.0 for value in ratios)),
        "within_5x_periapsis": int(sum(value <= 5.0 for value in ratios)),
        "below_training_radius_minimum": int(
            sum(value < training_radius_minimum for value in radii)
        ),
        "by_eccentricity": {
            str(e): {
                "n": sum(loc["eccentricity"] == e for loc in locations),
                "failure_radius_over_periapsis": quantiles(
                    [
                        loc["failure_radius_over_periapsis"]
                        for loc in locations if loc["eccentricity"] == e
                    ]
                ),
                "below_training_radius_minimum": int(
                    sum(
                        loc["failure_radius"] < training_radius_minimum
                        for loc in locations if loc["eccentricity"] == e
                    )
                ),
            }
            for e in sorted({loc["eccentricity"] for loc in locations})
        },
    }

    # ---- phase B intervention ----
    manual_by_scenario = {
        scenario_key(r)[2:]: r for r in ac9["records"]
        if r["architecture"] == "manual_r"
    }
    records = intervention["records"]
    changed = [r for r in records if r["rule_prescribes_change"]]
    unchanged = [r for r in records if not r["rule_prescribes_change"]]
    completed = [r for r in records if r["completion_status"] == "complete"]
    completed_changed = [
        r for r in changed if r["completion_status"] == "complete"
    ]
    cost_ratios, wall_ratios, closure_ratios = [], [], []
    for r in completed:
        manual = manual_by_scenario[scenario_key(r)[2:]]
        cost_ratios.append(
            r["vector_field_evaluations"] / manual["vector_field_evaluations"]
        )
        wall_ratios.append(r["wall_seconds"] / manual["wall_seconds"])
        closure_ratios.append(
            r["final_state_closure_error"] / manual["final_state_closure_error"]
        )
    manual_e099_closure = [
        r["final_state_closure_error"] for r in ac9["records"]
        if r["architecture"] == "manual_r" and r["eccentricity"] == 0.99
        and r["completion_status"] == "complete"
    ]
    manual_e099_energy = [
        r["max_energy_error"] for r in ac9["records"]
        if r["architecture"] == "manual_r" and r["eccentricity"] == 0.99
        and r["completion_status"] == "complete"
    ]
    by_group = defaultdict(list)
    for r in records:
        by_group[(r["architecture"], r["eccentricity"])].append(r)
    rerun_failures = [
        r for r in records if r["completion_status"] != "complete"
    ]
    rerun_failure_rows = []
    for r in rerun_failures:
        radius, _ = radius_at_physical_time(r, r["physical_time"])
        periapsis = r["semi_major_axis"] * (1.0 - r["eccentricity"])
        rerun_failure_rows.append(
            {
                "architecture": r["architecture"],
                "checkpoint_seed": r["checkpoint_seed"],
                "semi_major_axis": r["semi_major_axis"],
                "eccentricity": r["eccentricity"],
                "initial_mean_anomaly": r["initial_mean_anomaly"],
                "orientation_radians": r["orientation_radians"],
                "momentum_direction": r["momentum_direction"],
                "intervention_dtau": r["intervention_dtau"],
                "failure_physical_time": r["physical_time"],
                "failure_radius": radius,
                "failure_radius_over_periapsis": radius / periapsis,
                "periapsis_passages_before_failure":
                    periapsis_passages(r, r["physical_time"]),
                "max_energy_error": r["max_energy_error"],
                "original_steps": r["original_steps"],
                "steps": r["steps"],
            }
        )
    rerun_failure_summary = {
        "n": len(rerun_failures),
        "by_architecture": {
            architecture: sum(
                row["architecture"] == architecture
                for row in rerun_failure_rows
            )
            for architecture in ("raw", "invariant")
        },
        "failed_before_first_periapsis": sum(
            row["periapsis_passages_before_failure"] == 0
            for row in rerun_failure_rows
        ),
        "failed_after_surviving_periapsis": sum(
            row["periapsis_passages_before_failure"] >= 1
            for row in rerun_failure_rows
        ),
        "failure_radius_over_periapsis": quantiles(
            [row["failure_radius_over_periapsis"] for row in rerun_failure_rows]
        ) if rerun_failure_rows else None,
        "failure_radius_over_periapsis_before_first_periapsis": quantiles(
            [
                row["failure_radius_over_periapsis"]
                for row in rerun_failure_rows
                if row["periapsis_passages_before_failure"] == 0
            ]
        ) if any(
            row["periapsis_passages_before_failure"] == 0
            for row in rerun_failure_rows
        ) else None,
        "failure_radius_over_periapsis_after_surviving_periapsis": quantiles(
            [
                row["failure_radius_over_periapsis"]
                for row in rerun_failure_rows
                if row["periapsis_passages_before_failure"] >= 1
            ]
        ) if any(
            row["periapsis_passages_before_failure"] >= 1
            for row in rerun_failure_rows
        ) else None,
        "max_energy_error_failed_reruns": quantiles(
            [row["max_energy_error"] for row in rerun_failure_rows]
        ) if rerun_failure_rows else None,
        "max_energy_error_completed_reruns": quantiles(
            [r["max_energy_error"] for r in completed]
        ) if completed else None,
        "rows": rerun_failure_rows,
        "note":
            "Post-hoc descriptive characterisation of the rollouts that "
            "still failed under the prespecified step rule; not a "
            "registered hypothesis.",
    }
    intervention_summary = {
        "rule": intervention["rule"],
        "rho_target": intervention["rho_target"],
        "rerun_failed_rollouts": len(records),
        "rule_prescribed_smaller_step": len(changed),
        "rule_prescribed_no_change": len(unchanged),
        "completed": len(completed),
        "completed_of_prescribed_smaller_step": len(completed_changed),
        "completed_of_no_change": len(completed) - len(completed_changed),
        "completion_fraction_prescribed": (
            len(completed_changed) / len(changed) if changed else None
        ),
        "completion_wilson_95_prescribed": wilson(
            len(completed_changed), len(changed)
        ) if changed else None,
        "status_counts": {
            status: sum(r["completion_status"] == status for r in records)
            for status in sorted({r["completion_status"] for r in records})
        },
        "step_factor": quantiles([r["step_factor"] for r in changed]),
        "intervention_dtau": quantiles(
            [r["intervention_dtau"] for r in changed]
        ),
        "completed_vector_field_evaluations_over_manual_dtau004": (
            {
                **quantiles(cost_ratios),
                "geometric_mean": float(
                    math.exp(np.mean(np.log(cost_ratios)))
                ),
            } if cost_ratios else None
        ),
        "completed_wall_over_manual_dtau004": (
            {
                **quantiles(wall_ratios),
                "geometric_mean": float(
                    math.exp(np.mean(np.log(wall_ratios)))
                ),
            } if wall_ratios else None
        ),
        "completed_max_energy_error": quantiles(
            [r["max_energy_error"] for r in completed]
        ),
        "completed_final_state_closure_error": quantiles(
            [r["final_state_closure_error"] for r in completed]
        ),
        "completed_final_state_closure_error_over_manual_same_scenario": (
            {
                **quantiles(closure_ratios),
                "geometric_mean": float(
                    math.exp(np.mean(np.log(closure_ratios)))
                ),
            } if closure_ratios else None
        ),
        "manual_e099_final_state_closure_error_ac9": quantiles(
            manual_e099_closure
        ),
        "manual_e099_max_energy_error_ac9": quantiles(manual_e099_energy),
        "wall_ratio_note":
            "Wall-time ratios compare this Windows CPU run with the AC9 "
            "Linux run and are not like-for-like; vector-field evaluations "
            "are the machine-independent cost measure.",
        "by_architecture": {
            architecture: architecture_summary(
                architecture, records, completed, manual_by_scenario
            )
            for architecture in ("raw", "invariant")
        },
        "by_architecture_and_eccentricity": {
            f"{architecture}|e={eccentricity}": {
                "rerun": len(group),
                "completed": sum(
                    r["completion_status"] == "complete" for r in group
                ),
                "prescribed_smaller_step": sum(
                    r["rule_prescribes_change"] for r in group
                ),
            }
            for (architecture, eccentricity), group in sorted(
                by_group.items(), key=lambda item: (item[0][0], item[0][1])
            )
        },
        "rerun_failures": rerun_failure_summary,
        "cost_note":
            "Survivor-conditioned ratios over completed intervention "
            "rollouts; manual g=r denominators are the frozen AC9 "
            "dtau=0.04 rollouts of the same scenario.",
    }

    analysis = {
        "schema_version": "1.0",
        "experiment": raw["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "raw_results_sha256": sha256_file(RAW_PATH),
        "intervention_sha256": sha256_file(INTERVENTION_PATH),
        "ac9_raw_sha256": sha256_file(AC9_RAW),
        "counts": {
            "orbit_records": len(raw["orbit_records"]),
            "monitors": len(
                {(r["architecture"], r["checkpoint_seed"])
                 for r in raw["orbit_records"]}
            ),
            "orbit_families": len(
                {family_key(r)[2:] for r in raw["orbit_records"]}
            ),
            "ac9_rollouts_classified": len(rollout_rows),
            "failure_locations": len(locations),
            "intervention_rollouts": len(records),
        },
        "orbit_table": orbit_table,
        "separation": separation,
        "classification": classification,
        "failure_location": failure_location,
        "intervention": intervention_summary,
        "claim_limits": {
            "exact_hamiltonian_only": True,
            "linearised_fixed_point_criterion_on_analytic_orbit": True,
            "threshold_is_sufficient_not_necessary_condition": True,
            "kepler_ac9_grid_only": True,
            "step_rule_tested_only_on_failed_ac9_rollouts": True,
            "no_learned_hamiltonian": True,
        },
    }
    acceptance = {
        "all_396_orbit_records": analysis["counts"]["orbit_records"]
        == protocol["acceptance"]["expected_orbit_records"],
        "all_792_ac9_rollouts_classified":
            analysis["counts"]["ac9_rollouts_classified"]
            == protocol["acceptance"].get("expected_ac9_rollouts", 792),
        "all_failed_ac9_rollouts_located":
            len(locations) == protocol["acceptance"]["expected_failed_rollouts"],
        "all_failed_learned_rollouts_rerun":
            len(records) == protocol["acceptance"]["expected_failed_rollouts"],
        "all_intervention_outcomes_retained": all(
            "completion_status" in r for r in records
        ),
        "scientific_success_not_required": True,
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    analysis["acceptance"] = acceptance
    atomic_json(ANALYSIS_PATH, analysis)
    make_figure(raw, analysis, rollout_rows, threshold, training_radius_minimum)
    print(json.dumps(
        {
            "classification_learned": classification["learned_720_rollouts"],
            "intervention": {
                k: intervention_summary[k]
                for k in ("rerun_failed_rollouts", "completed",
                          "completed_of_prescribed_smaller_step")
            },
            "acceptance": acceptance,
        },
        indent=1,
    ))
    if not acceptance["artifact_gate_passed"]:
        raise SystemExit("AC19 artifact gate failed")


def make_figure(raw, analysis, rollout_rows, threshold,
                training_radius_minimum):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    colors = {"manual_r": "#F28E2B", "raw": "#E15759", "invariant": "#4E79A7"}
    labels = {"manual_r": "Manual $g=r$", "raw": "Raw-state",
              "invariant": "Invariant-feature"}

    # (a) monitor versus radius on the a=1, e=0.99 reference orbit
    used = set()
    for record in raw["orbit_records"]:
        if (
            record["semi_major_axis"] != 1.0
            or record["eccentricity"] != 0.99
            or "profile" not in record
        ):
            continue
        profile = record["profile"]
        architecture = record["architecture"]
        anomaly = np.asarray(profile["eccentric_anomaly"])
        radius = np.asarray(profile["radius"])
        monitor = np.asarray(profile["monitor"])
        # draw the outbound (E < pi) and inbound (E >= pi) legs separately so
        # that momentum-dependent raw-state monitors do not zig-zag
        for leg in (anomaly < math.pi, anomaly >= math.pi):
            order = np.argsort(radius[leg])
            axes[0].plot(
                radius[leg][order],
                monitor[leg][order],
                color=colors[architecture],
                linewidth=1.6 if architecture == "manual_r" else 0.8,
                alpha=1.0 if architecture == "manual_r" else 0.7,
                label=(
                    labels[architecture] if architecture not in used else None
                ),
            )
            used.add(architecture)
    axes[0].axvspan(
        0.0, training_radius_minimum, color="0.85", zorder=0,
        label="Below AC9 training $r_{\\min}$",
    )
    axes[0].set(
        xscale="log", yscale="log", xlabel="Radius $r$ on $a=1$, $e=0.99$ orbit",
        ylabel="Monitor $g(z)$",
    )
    axes[0].set_title("(a) Monitor extrapolation below training radius")
    axes[0].legend(frameon=False, loc="lower right")

    # (b) rho_max per rollout versus completion
    positions = {}
    x_labels = []
    group_centres = []
    short = {"manual_r": "man.", "raw": "raw", "invariant": "inv."}
    index = 0
    for eccentricity in (0.7, 0.9, 0.99):
        for architecture in ("manual_r", "raw", "invariant"):
            positions[(architecture, eccentricity)] = index
            x_labels.append(short[architecture])
            index += 1
        group_centres.append((index - 2, eccentricity))
        if eccentricity != 0.99:
            axes[1].axvline(index - 0.5 + 0.3, color="0.85", linewidth=0.8)
        index += 0.6
    for centre, eccentricity in group_centres:
        axes[1].text(
            centre, -0.13, f"$e={eccentricity}$", ha="center", va="top",
            fontsize=7, transform=axes[1].get_xaxis_transform(),
        )
    rng = np.random.default_rng(20260926)
    plotted = set()
    for row in rollout_rows:
        key = (row["architecture"], row["eccentricity"])
        failed = row["completion_status"] != "complete"
        marker = "x" if failed else "o"
        label = None
        if ("failed" if failed else "complete") not in plotted:
            label = "AC9 solver failure" if failed else "AC9 complete"
            plotted.add("failed" if failed else "complete")
        axes[1].scatter(
            positions[key] + rng.uniform(-0.22, 0.22),
            row["spectral_radius_maximum"],
            marker=marker,
            s=14,
            color=colors[row["architecture"]],
            alpha=0.7,
            linewidths=0.8,
            label=label,
        )
    axes[1].axhline(threshold, color="0.3", linestyle="--", linewidth=1)
    axes[1].text(
        0.05, threshold * 1.15, r"linearised threshold $\rho_{\max}=1$",
        fontsize=6.5, transform=axes[1].get_yaxis_transform(),
    )
    axes[1].set_xticks([positions[k] for k in positions])
    axes[1].set_xticklabels(x_labels, fontsize=6.5)
    axes[1].set(
        yscale="log",
        ylabel=r"$\rho_{\max}$ of $(\Delta\tau/2)\,J\,D^2K$ on reference orbit",
    )
    axes[1].set_xlabel("Monitor, grouped by eccentricity", labelpad=22)
    axes[1].set_title(r"(b) Prospective contraction versus AC9 outcome")
    axes[1].legend(frameon=False, loc="upper left")

    # (c) failure location and intervention completion
    ratios = [
        loc["failure_radius_over_periapsis"]
        for loc in raw["failure_locations"]
    ]
    bins = np.logspace(0, np.log10(max(max(ratios), 1.5) * 1.05), 24)
    axes[2].hist(ratios, bins=bins, color="#B07AA1", alpha=0.85)
    axes[2].set(
        xscale="log", xlabel=r"Failure radius / periapsis radius",
        ylabel="Failed AC9 rollouts",
    )
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axes[2].set_xticks([1, 2, 5, 10, 20, 50, 100, 200][: 1 + int(
        np.searchsorted([1, 2, 5, 10, 20, 50, 100, 200], bins[-1])
    )])
    axes[2].set_title("(c) Where failures occur; step-rule rerun")
    intervention = analysis["intervention"]
    by_arch = intervention["by_architecture"]
    text = (
        f"rule $\\Delta\\tau^*=\\Delta\\tau\\min(1,0.5/\\rho_{{\\max}})$\n"
        f"rerun {intervention['rerun_failed_rollouts']} failed rollouts, "
        f"completed {intervention['completed']}\n"
        f"raw {by_arch['raw']['completed']}/{by_arch['raw']['rerun']}, "
        f"invariant {by_arch['invariant']['completed']}/"
        f"{by_arch['invariant']['rerun']}"
    )
    axes[2].text(
        0.97, 0.95, text, transform=axes[2].transAxes, ha="right", va="top",
        fontsize=6.8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9,
              "edgecolor": "0.7"},
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        for directory in {FIGURE_ROOT, OUT}:
            figure.savefig(
                os.path.join(directory, f"{FIGURE_STEM}.{extension}"),
                dpi=300,
                bbox_inches="tight",
            )
    plt.close(figure)


if __name__ == "__main__":
    main()
