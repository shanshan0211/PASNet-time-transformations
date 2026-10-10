"""Audit saved subsection 5.2 numbers; do not rerun training or dynamics.

Only standard-library arithmetic, joins, source hashes and analytic radius
projection are checked. Saved monitor values/Hessians are not recomputed.
This is a partial claim trace, not a full manuscript or originality audit.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAMILY = ("architecture", "checkpoint_seed", "semi_major_axis", "eccentricity",
          "orientation_radians", "momentum_direction")
SCENARIO = ("semi_major_axis", "eccentricity", "initial_mean_anomaly",
            "orientation_radians", "momentum_direction")
RUN = ("architecture", "checkpoint_seed") + SCENARIO


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def key(record, fields):
    return tuple(record.get(field) for field in fields)


def index(records, fields):
    result = {key(r, fields): r for r in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate records: {fields}")
    return result


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    check(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
          f"Mismatch: {actual} != {expected}")


def summary(values):
    return {"n": len(values), "minimum": min(values),
            "median": median(values), "maximum": max(values)}


def geom(values):
    check(all(v > 0 for v in values), "Nonpositive geometric-mean input")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def reference_radius(record):
    # Analytic reference orbit at the recorded failure clock, not failed iterate.
    a, e = record["semi_major_axis"], record["eccentricity"]
    sign = -1 if record["momentum_direction"] == "reversed" else 1
    anomaly = (record["initial_mean_anomaly"] + sign * a ** -1.5
               * record["physical_time"]) % (2 * math.pi)
    lo, hi = anomaly - 1, anomaly + 1
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid - e * math.sin(mid) > anomaly:
            hi = mid
        else:
            lo = mid
    return a * (1 - e * math.cos((lo + hi) / 2))


def main():
    raw_path = "results/expAC19/raw_results.json"
    intervention_path = "results/expAC19/intervention.json"
    baseline_path = "results/expAC9/raw_results.json"
    raw, intervention, baseline = map(read, (raw_path, intervention_path, baseline_path))
    saved = read("results/expAC19/analysis.json")
    for field, path in (("raw_results_sha256", raw_path),
                        ("intervention_sha256", intervention_path),
                        ("ac9_raw_sha256", baseline_path)):
        check(digest(path) == saved[field], f"Changed saved-analysis source: {path}")
    protocol_path = "results/expAC19/preregistration.json"
    protocol = read(protocol_path)
    payload = json.loads(json.dumps(protocol))
    payload["integrity"].pop("protocol_sha256", None)
    canonical_hash = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    check(canonical_hash == saved["protocol_sha256"], "Changed canonical protocol")
    for data in (raw, intervention):
        check(data["protocol_sha256"] == saved["protocol_sha256"], "Protocol mismatch")
    source_checks = []
    for frozen in protocol["frozen_sources"]:
        match = digest(frozen["path"]) == frozen["sha256"]
        source_checks.append({"path": frozen["path"], "matches_frozen_hash": match})
        # Saved analyzer was subsequently extended with post-hoc descriptions.
        if frozen["path"] != "experiments/analyze_expAC19.py":
            check(match, f"Frozen source changed: {frozen['path']}")

    orbits = index(raw["orbit_records"], FAMILY)
    runs = index(baseline["records"], RUN)
    check(len(orbits) == 396 and len(runs) == 792, "Unexpected grid size")
    check(set(Counter(key(r, FAMILY) for r in runs.values()).values()) == {2},
          "Each reference diagnostic must serve two initial anomalies")
    for orbit in orbits.values():
        check(orbit["sampled_states"] == 720, "Unexpected sampling")
        check(orbit["predicted_fixed_point_failure"] ==
              (orbit["spectral_radius_maximum"] >= 1), "Wrong stored classifier")
    confusion = Counter()
    missed = []
    for r in runs.values():
        predicted = orbits[key(r, FAMILY)]["spectral_radius_maximum"] >= 1
        failed = r["completion_status"] != "complete"
        cell = "TP" if predicted and failed else "FN" if failed else "FP" if predicted else "TN"
        confusion[cell] += 1
        if cell == "FN":
            missed.append(r)
    check(dict(confusion) == {"TN": 540, "FN": 12, "TP": 240}, "Confusion mismatch")
    check(all((r["architecture"], r["semi_major_axis"], r["eccentricity"])
              == ("raw", .75, .9) for r in missed), "Unexpected misses")
    confusion["FP"] = 0
    high_e = {}
    for architecture in ("raw", "invariant", "manual_r"):
        selected = [r for r in orbits.values() if r["architecture"] == architecture
                    and r["eccentricity"] == .99]
        high_e[architecture] = {field: summary([r[field] for r in selected])
                               for field in ("monitor_at_periapsis",
                                             "monitor_over_radius_at_periapsis",
                                             "spectral_radius_maximum")}
    manual_max = max(r["spectral_radius_maximum"] for r in orbits.values()
                     if r["architecture"] == "manual_r")
    close(manual_max, saved["separation"]["manual_all_orbits_spectral_radius_maximum"]["maximum"])

    failed_runs = {k: r for k, r in runs.items() if r["completion_status"] != "complete"}
    locations = index(raw["failure_locations"], RUN)
    reruns = index(intervention["records"], RUN)
    check(set(failed_runs) == set(locations) == set(reruns), "Failure/rerun coverage mismatch")
    for k, r in locations.items():
        close(r["physical_time"], failed_runs[k]["physical_time"])
        radius = reference_radius(r)
        close(radius, r["failure_radius"])
        close(radius / (r["semi_major_axis"] * (1 - r["eccentricity"])),
              r["failure_radius_over_periapsis"])
    ratios = [r["failure_radius_over_periapsis"] for r in locations.values()]
    location_counts = {"n": len(ratios), "within_2x": sum(x <= 2 for x in ratios),
                       "within_5x": sum(x <= 5 for x in ratios),
                       "below_training_radius": sum(r["failure_radius"] < .1 for r in locations.values()),
                       "reference_radius_over_periapsis": summary(ratios)}
    check((location_counts["n"], location_counts["within_2x"], location_counts["within_5x"],
           location_counts["below_training_radius"]) == (252, 60, 231, 250), "Location counts differ")
    for r in reruns.values():
        rho = orbits[key(r, FAMILY)]["spectral_radius_maximum"]
        close(r["spectral_radius_maximum_at_original_dtau"], rho)
        close(r["step_factor"], min(1, .5 / rho))
        close(r["intervention_dtau"], .04 * r["step_factor"])
        check(r["step_factor"] < 1, "Unchanged step")
    statuses = Counter(r["completion_status"] for r in reruns.values())
    check(dict(statuses) == {"complete": 151, "solver_failure": 101}, "Rerun status mismatch")
    strata = {}
    for architecture, e in (("raw", .9), ("raw", .99), ("invariant", .99)):
        selected = [r for r in reruns.values() if r["architecture"] == architecture and r["eccentricity"] == e]
        strata[f"{architecture}|{e}"] = {"n": len(selected),
            "complete": sum(r["completion_status"] == "complete" for r in selected)}
    manual = index([r for r in runs.values() if r["architecture"] == "manual_r"], SCENARIO)
    completed = [r for r in reruns.values() if r["completion_status"] == "complete"]
    cost = geom([r["vector_field_evaluations"] / manual[key(r, SCENARIO)]["vector_field_evaluations"]
                 for r in completed])
    error = geom([r["final_state_closure_error"] / manual[key(r, SCENARIO)]["final_state_closure_error"]
                  for r in completed])
    close(cost, saved["intervention"]["completed_vector_field_evaluations_over_manual_dtau004"]["geometric_mean"])
    close(error, saved["intervention"]["completed_final_state_closure_error_over_manual_same_scenario"]["geometric_mean"])

    distilled = read("results/expAC8/raw_results.json")["records"]
    dist_analysis = read("results/expAC8/analysis.json")
    transitions, solver_counts = {}, {}
    for architecture in ("raw", "invariant", "manual_r"):
        selected = [r for r in distilled if r["architecture"] == architecture]
        paired = {}
        for r in selected:
            k = key(r, ("seed", "eccentricity", "orientation_radians", "momentum_direction"))
            pair = paired.setdefault(k, {})
            check(r["solver"] not in pair, "Duplicate solver case")
            pair[r["solver"]] = r["completion_status"] == "complete"
        counter = Counter()
        for pair in paired.values():
            check(set(pair) == {"fixed_point", "damped_newton"}, "Incomplete solver pair")
            fp, newton = pair["fixed_point"], pair["damped_newton"]
            counter["both_complete" if fp and newton else "fixed_point_only" if fp
                    else "newton_only" if newton else "both_fail"] += 1
        transitions[architecture] = dict(counter)
        if architecture != "manual_r":
            check(dict(counter) == dist_analysis["solver_transitions"][architecture], "Solver transition mismatch")
        solver_counts[architecture] = {solver: {"n": len(paired),
            "complete": sum(p[solver] for p in paired.values())}
            for solver in ("fixed_point", "damped_newton")}
    paths = (raw_path, intervention_path, baseline_path, protocol_path,
             "results/expAC8/raw_results.json", "results/expAC8/analysis.json",
             "experiments/analyze_expAC19.py", "new/paper_g2_jcp_new.tex",
             "new/audit_section_5_2.py")
    result = {"scope": __doc__, "source_sha256": {p: digest(p) for p in paths},
              "registered_source_checks": source_checks,
              "reference_orbits": len(orbits), "high_e_diagnostics": high_e,
              "manual_all_orbit_rho_max": manual_max, "confusion": dict(confusion),
              "classification_accuracy": (confusion["TP"] + confusion["TN"]) / len(runs),
              "failure_sensitivity": confusion["TP"] / len(failed_runs),
              "reference_failure_locations": location_counts,
              "intervention": {"statuses": dict(statuses), "strata": strata,
                  "step_factor": summary([r["step_factor"] for r in reruns.values()]),
                  "completed_n": len(completed), "completed_field_ratio_geom": cost,
                  "completed_closure_ratio_geom": error},
              "distilled_solver_transitions": transitions, "distilled_solver_counts": solver_counts,
              "limits": ["No new dynamics, training, Hessian or checkpoint inference.",
                         "Reference failure radius is not an actual failed-iterate radius.",
                         "Known-label retrospective classification; correlated evaluations.",
                         "Work ratios compare field calls, not identical arithmetic or pipeline costs.",
                         "Cross-environment wall ratios excluded; accuracy/cost ratios condition on completion.",
                         "Saved analyzer was extended after sealing; current hash separately recorded."]}
    successor_path = "results/returned_endpoint_review/verified/analysis.json"
    pairing_path = "results/returned_endpoint_review/paired_analysis.json"
    if (ROOT / successor_path).is_file():
        successor = read(successor_path)
        paired = read(pairing_path)
        restored = sum(successor["by_arm"][arm]["complete"]
                       for arm in ("AC19_raw", "AC19_invariant"))
        check(restored == 150, "Verified intervention count differs")
        result["intervention"]["historical_update_stop_scope"] = True
        result["verified_intervention"] = {
            "complete": restored, "failed": 252 - restored,
            "strata": paired["intervention_strata"],
            "completed_closure_ratio_geom": paired["pooled_intervention"]["closure_ratio_geom"],
            "completed_field_ratio_with_verification_geom": paired["pooled_intervention"]["field_ratio_geom_with_acceptance_verification"],
            "scope": "current main-text counts; fresh replay acceptance; historical manual cost denominator"}
        result["source_sha256"].update({p: digest(p) for p in (successor_path, pairing_path)})
    output = HERE / "SECTION_5_2_CLAIMS.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"reference_orbits": len(orbits), "confusion": dict(confusion),
                      "intervention_statuses": dict(statuses), "solver_transitions": transitions,
                      "field_ratio_geom": cost, "closure_ratio_geom": error,
                      "trace": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
