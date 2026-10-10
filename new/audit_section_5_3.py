"""Trace subsection 5.3 to saved training/diagnostic/rollout artifacts.

Standard-library checks only. No training, network inference, Hessian,
rollout or bootstrap is rerun. This is a partial numerical claim audit.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ("invariant_e099", "sundman_e09", "sundman_e099")
SCENARIO = ("semi_major_axis", "eccentricity", "initial_mean_anomaly",
            "orientation_radians", "momentum_direction")
FAMILY = ("semi_major_axis", "eccentricity", "orientation_radians", "momentum_direction")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def key(record, fields):
    return tuple(record[field] for field in fields)


def stats(values):
    return {"n": len(values), "minimum": min(values), "median": median(values), "maximum": max(values)}


def confusion(records, orbits):
    counts = Counter({k: 0 for k in ("TP", "FN", "FP", "TN")})
    for r in records:
        predicted = orbits[(r["checkpoint_seed"], *key(r, FAMILY))]["spectral_radius_maximum"] >= 1
        failed = r["completion_status"] != "complete"
        counts["TP" if predicted and failed else "FN" if failed else "FP" if predicted else "TN"] += 1
    return dict(counts)


def main():
    protocol_path = "results/expAC20/preregistration.json"
    analysis_path = "results/expAC20/analysis.json"
    baseline_path = "results/expAC9/raw_results.json"
    protocol, saved, baseline = map(read, (protocol_path, analysis_path, baseline_path))
    payload = json.loads(json.dumps(protocol))
    payload["integrity"].pop("protocol_sha256", None)
    protocol_hash = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    check(protocol_hash == protocol["integrity"]["protocol_sha256"] == saved["protocol_sha256"],
          "Canonical protocol hash mismatch")
    check(digest(baseline_path) == saved["ac9_raw_sha256"], "Baseline hash mismatch")
    hashes = {p: digest(p) for p in (protocol_path, analysis_path, baseline_path)}
    for r in protocol["frozen_sources"]:
        hashes[r["path"]] = digest(r["path"])
        check(hashes[r["path"]] == r["sha256"], f"Changed frozen source: {r['path']}")
    for p, expected in saved["part_sha256"].items():
        hashes[p] = digest(p)
        check(hashes[p] == expected, f"Part hash mismatch: {p}")
    check(len(saved["part_sha256"]) == 45, "Unexpected part manifest")
    seeds = [r["checkpoint_seed"] for r in protocol["training"]["seed_manifest"]]
    baseline_invariant = [r for r in baseline["records"] if r["architecture"] == "invariant"]
    scenario_set = {key(r, SCENARIO) for r in baseline_invariant}
    check(len(scenario_set) == 72, "Unexpected baseline grid")
    outcomes, timing, checkpoints, all_confusion = {}, [], [], Counter()
    for arm in ARMS:
        records, diagnostic, training = [], [], []
        for seed in seeds:
            stem = f"results/expAC20/parts/{arm}_seed{seed}"
            t, d, r = [read(f"{stem}/{name}.json") for name in ("training", "diagnostic", "rollouts")]
            check(all(data["protocol_sha256"] == protocol_hash for data in (t, d, r)), "Part protocol mismatch")
            times = [datetime.fromisoformat(data[field]) for data, field in
                     ((t, "completed_at_utc"), (d, "started_at_utc"), (d, "completed_at_utc"),
                      (r, "started_at_utc"), (r, "completed_at_utc"))]
            check(all(a <= b for a, b in zip(times, times[1:])), "Diagnostic did not precede rollouts")
            timing.append({"arm": arm, "seed": seed,
                           "diagnostic_completed": d["completed_at_utc"], "rollouts_started": r["started_at_utc"]})
            tr = t["record"]
            check(tr["arm"] == arm and tr["checkpoint_seed"] == seed and tr["status"] == "trained", "Training identity mismatch")
            hashes[tr["checkpoint_path"]] = digest(tr["checkpoint_path"])
            check(hashes[tr["checkpoint_path"]] == tr["checkpoint_sha256"], "Checkpoint hash mismatch")
            check(len(d["orbit_records"]) == 36 and len(r["records"]) == 72, "Part count mismatch")
            check({key(x, SCENARIO) for x in r["records"]} == scenario_set, "Scenario grid mismatch")
            check(len({key(x, FAMILY) for x in d["orbit_records"]}) == 36, "Duplicate reference families")
            for x in d["orbit_records"] + r["records"]:
                check(x["arm"] == arm and x["checkpoint_seed"] == seed, "Record identity mismatch")
            check(all(x["sampled_states"] == 720 for x in d["orbit_records"]), "Diagnostic sample mismatch")
            check(all(x["predicted_fixed_point_failure"] == (x["spectral_radius_maximum"] >= 1)
                      for x in d["orbit_records"]), "Stored prediction mismatch")
            records.extend(r["records"])
            diagnostic.extend(d["orbit_records"])
            training.append(tr)
            complete = sum(x["completion_status"] == "complete" for x in r["records"])
            high_complete = sum(x["completion_status"] == "complete" for x in r["records"] if x["eccentricity"] == .99)
            checkpoints.append({"arm": arm, "seed": seed, "complete": complete,
                                "high_e_complete": high_complete, "validation_mse": tr["best_validation_state_mse"],
                                "skipped_epochs": tr["nonfinite_epochs_skipped"]})
            check(complete == saved["per_checkpoint"][arm][str(seed)]["complete"], "Checkpoint completion mismatch")
        orbit_index = {(x["checkpoint_seed"], *key(x, FAMILY)): x for x in diagnostic}
        check(len(orbit_index) == 180, "Reference identity mismatch")
        cell = confusion(records, orbit_index)
        all_confusion.update(cell)
        mapping = {"TP": "predicted_fail_and_failed", "FN": "predicted_complete_but_failed",
                   "FP": "predicted_fail_but_completed", "TN": "predicted_complete_and_completed"}
        check(all(cell[k] == saved["prospective_prediction"][arm][v] for k, v in mapping.items()), "Confusion mismatch")
        by_e = {}
        for e in (.7, .9, .99):
            subset = [x for x in records if x["eccentricity"] == e]
            block = {"n": len(subset), "complete": sum(x["completion_status"] == "complete" for x in subset)}
            check(block == saved["completion"][arm]["by_eccentricity"][str(e)], "Stratum count mismatch")
            by_e[str(e)] = block
        high_orbits = [x for x in diagnostic if x["eccentricity"] == .99]
        diag = {field: stats([x[field] for x in high_orbits]) for field in
                ("monitor_over_radius_at_periapsis", "spectral_radius_maximum")}
        for field, block in diag.items():
            compare = saved["monitor_structure"][arm]["0.99"][field]
            check(all(abs(block[k] - compare[k]) <= 1e-12 for k in block), "Diagnostic summary mismatch")
        fit = stats([t["best_validation_state_mse"] for t in training])
        check(all(abs(fit[k] - saved["training"][arm]["best_validation_state_mse"][k]) <= 1e-12
                  for k in fit), "Validation summary mismatch")
        paired = []
        for seed in seeds:
            new = [x for x in records if x["checkpoint_seed"] == seed]
            old = [x for x in baseline_invariant if x["checkpoint_seed"] == seed]
            paired.append((sum(x["completion_status"] == "complete" for x in new)
                           - sum(x["completion_status"] == "complete" for x in old)) / 72)
        saved_diff = saved["paired_checkpoint_contrasts"][arm]["all_scenarios_minus_ac9_invariant"]
        check(all(abs(a - b) < 1e-12 for a, b in zip(paired, saved_diff["paired_checkpoint_differences"])), "Paired contrast mismatch")
        outcomes[arm] = {"completion_by_e": by_e, "confusion": cell,
                         "high_e_diagnostic": diag, "all_family_rho_max": max(x["spectral_radius_maximum"] for x in diagnostic),
                         "validation_mse": fit, "skipped_epochs": sum(t["nonfinite_epochs_skipped"] for t in training),
                         "paired_all_scenario_difference": paired,
                         "saved_bootstrap_read_only": saved_diff}
    check(dict(all_confusion) == {"TP": 120, "FN": 0, "FP": 0, "TN": 960}, "Pooled confusion mismatch")
    for p in ("new/paper_g2_jcp_new.tex", "new/audit_section_5_3.py"):
        hashes[p] = digest(p)
    output = {"scope": __doc__, "source_sha256": hashes, "outcomes": outcomes,
              "pooled_confusion": dict(all_confusion), "checkpoint_details": checkpoints,
              "time_order_checks": timing,
              "limits": ["Only saved timestamp fields verified; no independent execution reconstruction.",
                         "Checkpoint hashes checked without loading networks or rerunning Hessians.",
                         "Bootstrap fields read, not rerun; equal checkpoint differences give degenerate intervals.",
                         "Factored output bias zero with random weights does not initialise exactly g=r.",
                         "Changed training periapsis .01 does not cover deployment .0075 or all phase states.",
                         "No residual replay coverage or global convergence/efficiency guarantee."]}
    (HERE / "SECTION_5_3_CLAIMS.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoints": len(checkpoints), "rollouts": 1080,
                      "pooled_confusion": dict(all_confusion),
                      "completion": {a: o["completion_by_e"] for a, o in outcomes.items()},
                      "skipped_epochs": {a: o["skipped_epochs"] for a, o in outcomes.items()}}, indent=2))


if __name__ == "__main__":
    main()
