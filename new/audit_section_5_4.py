"""Audit saved accuracy/cost and scope evidence without new experiments.

Recompute counts, positive geometric means, discrete Pareto sets, fidelity
decisions, and reference-uncertainty arithmetic. Do not rerun dynamics,
networks, reference trajectories, training or bootstrap inference.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KEPLER = ("semi_major_axis", "eccentricity", "initial_mean_anomaly",
          "orientation_radians", "momentum_direction")
CROSS = ("system", "energy_shell", "initial_condition_seed", "kappa", "momentum_direction")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    check(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-13), f"Mismatch {a} != {b}")


def key(r, fields):
    return tuple(r.get(k) for k in fields)


def geom(values):
    check(values and all(v > 0 and math.isfinite(v) for v in values), "Invalid geometric mean input")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def front(configs, x, y):
    eligible = [r for r in configs if r["complete"] == r["n"]]
    result = [r for r in eligible if not any(
        s[x] <= r[x] and s[y] <= r[y] and (s[x] < r[x] or s[y] < r[y])
        for s in eligible if s is not r)]
    return [r["id"] for r in sorted(result, key=lambda r: r[x])]


def main():
    hashes = {}
    for study in ("expAC10", "expAC13", "expAC16", "expAC18"):
        a = read(f"results/{study}/analysis.json")
        names = {"raw": f"results/{study}/raw_results.json",
                 "summary": f"results/{study}/summary.json",
                 "training": f"results/{study}/training.json",
                 "protocol": f"results/{study}/preregistration.json",
                 "analysis_script": f"experiments/analyze_{study}.py",
                 "ac13_raw": "results/expAC13/raw_results.json",
                 "hamiltonian_audit": "results/expAC18/hamiltonian_audit.json"}
        if study == "expAC18":
            # AC18's wrapper delegates to AC17; __file__ in that analyzer
            # is the source identified by the saved analysis hash.
            names["analysis_script"] = "experiments/analyze_expAC17.py"
        for label, expected in a["source_sha256"].items():
            path = names[label]
            hashes[path] = digest(path)
            check(hashes[path] == expected, f"Changed analysis source {path}")
        hashes[f"results/{study}/analysis.json"] = digest(f"results/{study}/analysis.json")
    raw = read("results/expAC10/raw_results.json")["records"]
    saved = read("results/expAC10/analysis.json")
    grouped = defaultdict(list)
    for r in raw:
        grouped[r["configuration_id"]].append(r)
    check(len(raw) == 684 and len(grouped) == 38, "Unexpected cost grid")
    saved_configs = {r["configuration_id"]: r for r in saved["configuration_summaries"]}
    configs, architecture_steps = [], {}
    metric_mapping = {"field": "gradient_evaluations", "energy": "max_energy_error",
                      "closure": "final_state_closure_error", "wall": "wall_seconds"}
    for name, records in grouped.items():
        check(len(records) == 18 and len({key(r, KEPLER) for r in records}) == 18, "Duplicate/missing cost scenario")
        completed = [r for r in records if r["completion_status"] == "complete"]
        s = saved_configs[name]
        check(len(completed) == s["complete"], "Completion mismatch")
        c = {"id": name, "n": len(records), "complete": len(completed),
             "method": records[0]["method"], "architecture": records[0].get("architecture")}
        for label, field in metric_mapping.items():
            c[label] = geom([r[field] for r in completed])
            close(c[label], s["survivor_conditioned_metrics"][field]["geometric_mean"])
        configs.append(c)
    fronts = {name: front(configs, x, y) for name, x, y in
              (("energy_vs_evaluations", "field", "energy"),
               ("closure_vs_evaluations", "field", "closure"),
               ("closure_vs_wall", "wall", "closure"))}
    check(fronts == saved["robust_pareto_fronts"], "Pareto set mismatch")
    for architecture in ("raw", "invariant"):
        for step in (.08, .04, .02):
            selected = [r for r in raw if r["architecture"] == architecture and r["configuration_id"].endswith(f"dtau_{step}")]
            block = {"n": len(selected), "complete": sum(r["completion_status"] == "complete" for r in selected),
                     "high_e_complete": sum(r["completion_status"] == "complete" for r in selected if r["eccentricity"] == .99)}
            check(block["n"] == 90, "Architecture/step coverage mismatch")
            check(block["complete"] == saved["learned_architecture_dtau"][f"{architecture}|{step}"]["complete"], "Step completion mismatch")
            architecture_steps[f"{architecture}|{step}"] = block
    replay_path = "results/expAC15/raw_results.json"
    replay = [r for r in read(replay_path)["records"] if r["parent_experiment"] == "AC10"]
    check(len(replay) == 684 and len({r["parent_record_index"] for r in replay}) == 684, "Replay coverage mismatch")
    for r in replay:
        parent = raw[r["parent_record_index"]]
        check(r["configuration_id"] == parent["configuration_id"], "Replay identity mismatch")
        check((r["completion_status"] == "complete") == (parent["completion_status"] == "complete"), "Replay changed completion")
    hashes[replay_path] = digest(replay_path)

    baseline_path = "results/expAC9/raw_results.json"
    manual = {key(r, KEPLER): r for r in read(baseline_path)["records"] if r["architecture"] == "manual_r"}
    structural = read("results/expAC20/analysis.json")
    check(digest(baseline_path) == structural["ac9_raw_sha256"], "Factored baseline changed")
    factored = {}
    for arm in ("sundman_e09", "sundman_e099"):
        records = []
        for seed in range(7101, 7106):
            path = f"results/expAC20/parts/{arm}_seed{seed}/rollouts.json"
            check(digest(path) == structural["part_sha256"][path], "Factored part changed")
            hashes[path] = digest(path)
            records.extend(read(path)["records"])
        check(len(records) == 360 and all(r["completion_status"] == "complete" for r in records), "Factored scope mismatch")
        factored[arm] = {}
        for label in ("all", "0.99"):
            selected = records if label == "all" else [r for r in records if r["eccentricity"] == .99]
            result = {"n": len(selected)}
            saved_block = structural["accuracy_cost"][arm]["all"] if label == "all" else structural["accuracy_cost"][arm]["by_eccentricity"][label]
            for metric in ("final_state_closure_error", "max_energy_error", "vector_field_evaluations"):
                value = geom([r[metric] / manual[key(r, KEPLER)][metric] for r in selected])
                close(value, saved_block[metric + "_over_manual"]["geometric_mean"])
                result[metric + "_ratio_geom"] = value
            factored[arm][label] = result

    cross = read("results/expAC13/raw_results.json")["records"]
    references = read("results/expAC16/raw_results.json")["records"]
    check(len(cross) == 288 and all(r["completion_status"] == "complete" for r in cross), "Cross-system completion mismatch")
    reference_index = {key(r, CROSS): r for r in references}
    check(len(reference_index) == 36 and all(r["completion_status"] == "complete" for r in references), "Reference coverage mismatch")
    cross_groups = {}
    ranked_saved = read("results/expAC16/analysis.json")["aggregate_state_error_rankings"]
    for system in ("modal", "duffing"):
        groups = {}
        for label in ("raw", "reversal_even", "prescribed_monitor"):
            selected = [r for r in cross if r["system"] == system and
                        (r.get("architecture") == label if label != "prescribed_monitor" else r["method"] == label)]
            groups[label] = {"n": len(selected), "energy_median": median(r["max_energy_error"] for r in selected),
                             "field_median": median(r["vector_field_evaluations"] for r in selected)}
            for metric, ref_field in (("max_state_error", "maximum"), ("final_state_error", "final")):
                estimates = [reference_index[key(r, CROSS)]["state_reference_uncertainty"]["ac13_dt"][ref_field]["absolute"] for r in selected]
                lower = median(max(0, r[metric] - err) for r, err in zip(selected, estimates))
                upper = median(r[metric] + err for r, err in zip(selected, estimates))
                block = {"median": median(r[metric] for r in selected), "reference_sensitivity_interval": [lower, upper]}
                saved_label = label if label == "prescribed_monitor" else "learned_" + label
                saved_block = ranked_saved[f"{system}|{metric}"]["groups"][saved_label]
                close(block["median"], saved_block["median_reported"])
                for a, b in zip(block["reference_sensitivity_interval"], saved_block["conservative_median_interval"]):
                    close(a, b)
                groups[label][metric] = block
        for metric in ("max_state_error", "final_state_error"):
            order = sorted(groups, key=lambda label: groups[label][metric]["median"])
            stable = all(groups[a][metric]["reference_sensitivity_interval"][1] < groups[b][metric]["reference_sensitivity_interval"][0]
                         for a, b in zip(order, order[1:]))
            check(stable == ranked_saved[f"{system}|{metric}"]["reported_total_order_stable"], "Reference ranking mismatch")
        cross_groups[system] = groups
    observed_orders = {}
    for system in ("modal", "duffing"):
        selected = [r for r in references if r["system"] == system]
        orders = [math.log2(r["state_level_differences"]["0.002_vs_0.001"]["maximum"] /
                           r["state_level_differences"]["0.001_vs_0.0005"]["maximum"]) for r in selected]
        for r, order in zip(selected, orders):
            close(order, r["state_observed_order"]["maximum"])
        observed_orders[system] = {"median": median(orders), "minimum": min(orders), "maximum": max(orders)}

    audit_path = "results/expAC18/hamiltonian_audit.json"
    audit = read(audit_path)
    gate = read("results/expAC18/preregistration.json")["hamiltonian_fidelity_gate"]
    decisions = []
    for r in audit["records"]:
        results = []
        for kind in ("hamiltonian_gradient", "vector_field"):
            m = r[kind]
            passed = m["all_finite"] and m["normalized_rmse"] <= gate["maximum_normalized_rmse"] and m["relative_error_p95"] <= gate["maximum_relative_error_p95"] and m["relative_error_max"] <= gate["maximum_relative_error_max"] and all(v <= gate["maximum_stratum_normalized_rmse"] for v in m["by_eccentricity_normalized_rmse"].values())
            results.append(passed)
        check(all(results) == r["accepted"], "Gate decision mismatch")
        decisions.append({"seed": r["h_seed"], "accepted": r["accepted"],
                          "overall_nrmse": r["hamiltonian_gradient"]["normalized_rmse"],
                          "high_e_nrmse": r["hamiltonian_gradient"]["by_eccentricity_normalized_rmse"]["0.99"]})
    check(len(decisions) == 5 and not any(r["accepted"] for r in decisions), "Unexpected gate outcome")
    check(read("results/expAC18/raw_results.json")["records"] == [], "Downstream deployment unexpectedly exists")
    for path in (baseline_path, "results/expAC20/analysis.json", "results/expAC18/preregistration.json",
                 "new/paper_g2_jcp_new.tex", "new/audit_section_5_4.py",
                 "new/make_section_5_4_figure.py", "new/fig_accuracy_cost_new.tex",
                 "new/fig_accuracy_cost_new.pdf", "new/fig_accuracy_cost_new.png"):
        hashes[path] = digest(path)
    output = {"scope": __doc__, "source_sha256": hashes, "cost_configurations": configs,
              "cost_fronts_original_counts": fronts, "learned_completion_by_step": architecture_steps,
              "residual_replay_completion_preserved": len(replay), "factored_same_step_ratios": factored,
              "cross_system_groups": cross_groups, "reference_max_state_observed_orders": observed_orders,
              "learned_h_gate": decisions,
              "limits": ["Historical fronts use original counts/timings; residual replay adds work and is not substituted.",
                         "Incomplete configuration metrics condition on completion and use different scenario subsets.",
                         "Factored same-step comparisons are not matched-cost or pipeline-cost comparisons.",
                         "Reference-sensitivity intervals are arithmetic using Richardson estimates, not rigorous or sampling confidence bounds.",
                         "No reference is an exact zero-error comparator; no general coupled-system evidence."]}
    (HERE / "SECTION_5_4_CLAIMS.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"fronts": fronts, "factored_ratios": factored, "learned_h_gate": decisions,
                      "cross_system_records": len(cross)}, indent=2))


if __name__ == "__main__":
    main()
