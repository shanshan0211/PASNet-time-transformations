"""Pair endpoint reviews with immutable parent records and manual controls."""
from collections import Counter
import json
import math
from statistics import median

import torch
import review_returned_endpoints as review
from replay_verified_endpoints import verified_step


def key(r):
    return tuple(r[k] for k in review.SCENARIO)


def geom(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def main():
    base = review.OUT
    originals = {g["id"]: g for g in review.groups()}
    manual = {key(r): r for r in review.read(review.ROOT / "results/expAC9/raw_results.json")["records"]
              if r["architecture"] == "manual_r"}
    result = {"paired_metrics": {}, "intervention_strata": {}, "status_transitions": [],
              "source_hashes": {}, "observational_scope": "all original solver returns, including terminal search candidates",
              "verified_scope": "update trigger AND returned residual <1e-12 within 100 iterations"}
    for group_id, group in originals.items():
        path = base / "verified/parts" / (group_id + ".json")
        result["source_hashes"][str(path.relative_to(review.ROOT))] = review.digest(path)
        rs = review.read(path)["records"]
        for r in rs:
            old = group["records"][r["parent_index"]]
            if r["historical_status"] != r["completion_status"]:
                result["status_transitions"].append(r)
            if group["study"] == "AC19":
                name = f"{r['arm']}_e{r['eccentricity']}"
                cell = result["intervention_strata"].setdefault(name, {"n": 0, "complete": 0, "failed": 0})
                cell["n"] += 1
                cell["complete"] += r["completion_status"] == "complete"
                cell["failed"] += r["completion_status"] != "complete"
            name = f"{r['study']}_{r['arm']}"
            cell = result["paired_metrics"].setdefault(name, {"closure_changes": [], "energy_changes": [],
                    "field_ratios_to_manual_including_verification": [], "closure_ratios_to_manual": [],
                    "energy_ratios_to_manual": [], "n_complete": 0, "maximum_terminal_discrepancy": 0.0})
            if r["completion_status"] == "complete":
                cell["n_complete"] += 1
                cell["closure_changes"].append(r["final_state_closure_error"] - old["final_state_closure_error"])
                cell["energy_changes"].append(r["max_energy_error"] - old["max_energy_error"])
                m = manual[key(r)]
                cell["field_ratios_to_manual_including_verification"].append(r["original_algorithm_field_evaluations"] / m["vector_field_evaluations"])
                cell["closure_ratios_to_manual"].append(r["final_state_closure_error"] / m["final_state_closure_error"])
                cell["energy_ratios_to_manual"].append(r["max_energy_error"] / m["max_energy_error"])
                cell["maximum_terminal_discrepancy"] = max(cell["maximum_terminal_discrepancy"], abs(r["terminal_candidate_clock_discrepancy"]))
    for cell in result["paired_metrics"].values():
        for name in ("closure_changes", "energy_changes"):
            values = cell.pop(name)
            cell[name + "_max_abs"] = max(map(abs, values))
            cell[name + "_median"] = median(values)
        for name in ("field_ratios_to_manual_including_verification", "closure_ratios_to_manual", "energy_ratios_to_manual"):
            cell[name + "_geom"] = geom(cell.pop(name))
    interventions = [r for g in originals.values() if g["study"] == "AC19"
                    for r in review.read(base / "verified/parts" / (g["id"] + ".json"))["records"]
                    if r["completion_status"] == "complete"]
    result["pooled_intervention"] = {"n_complete": len(interventions),
        "closure_ratio_geom": geom([r["final_state_closure_error"] / manual[key(r)]["final_state_closure_error"] for r in interventions]),
        "field_ratio_geom_with_acceptance_verification": geom([r["original_algorithm_field_evaluations"] / manual[key(r)]["vector_field_evaluations"] for r in interventions]),
        "manual_denominator": "historical update-stopping manual counts, NOT matched-cost verified controls"}
    changed = result["status_transitions"][0]
    g = originals[f"AC19_raw_{changed['checkpoint_seed']}"]
    evaluation = review.read(review.ROOT / "results/expAC9/preregistration.json")["evaluation"]
    evaluation.update(dtau=g["records"][changed["parent_index"]]["intervention_dtau"], maximum_steps=2000000)

    def scalar_step(z, h, field, config):
        out, ok, bad, its, calls, update, residual = verified_step(z, torch.tensor([h]), lambda value, ids: field(value),
                                         config["tolerance"], config["maximum_iterations"])
        if not bool(ok[0]):
            raise RuntimeError("scalar residual-verification iteration limit")
        return out, float(residual[0]), int(its[0]), int(calls[0])
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    review.ac9.transformed_step = scalar_step
    scalar = review.ac9.run_rollout(review.load_model(g), "raw", changed["checkpoint_seed"],
                                    {k: changed[k] for k in review.SCENARIO}, evaluation)
    result["changed_case_scalar_confirmation"] = scalar
    assert scalar["completion_status"] == changed["completion_status"]
    assert scalar["steps"] == changed["steps"]
    review.atomic(base / "paired_analysis.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
