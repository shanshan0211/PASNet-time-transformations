"""Trace the reliability subsection to saved data, without rerunning dynamics.

Uses only the Python standard library. The generated record covers this
subsection's numerical statements; it is not an experiment reproduction,
a bibliography audit, or a complete manuscript claim manifest.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def counts(records):
    return {
        "complete": sum(r["completion_status"] == "complete" for r in records),
        "registered": len(records),
    }


def check_counts(actual, saved, context):
    for key in ("complete", "registered"):
        if actual[key] != saved[key]:
            raise ValueError(f"{context}: {key} differs from saved analysis")


def main():
    raw_path = "results/expAC9/raw_results.json"
    analysis_path = "results/expAC9/analysis.json"
    raw = read(raw_path)
    analysis = read(analysis_path)
    records = raw["records"]
    source_paths = {
        "raw": raw_path,
        "training": "results/expAC9/training.json",
        "summary": "results/expAC9/summary.json",
        "analysis_script": "experiments/analyze_expAC9.py",
    }
    for key, relative in source_paths.items():
        if sha256(relative) != analysis["source_sha256"][key]:
            raise ValueError(f"Saved analysis source has changed: {relative}")

    if len(records) != 792:
        raise ValueError("Unexpected reliability evaluation count")
    scenario_fields = (
        "architecture", "checkpoint_seed", "semi_major_axis", "eccentricity",
        "initial_mean_anomaly", "orientation_radians", "momentum_direction",
    )
    keys = {tuple(r.get(k) for k in scenario_fields) for r in records}
    if len(keys) != len(records):
        raise ValueError("Duplicate checkpoint/scenario evaluation")

    totals, eccentricities, checkpoints, grid, symmetries, failures = {}, {}, {}, [], {}, {}
    for architecture in ("raw", "invariant", "manual_r"):
        selected = [r for r in records if r["architecture"] == architecture]
        totals[architecture] = counts(selected)
        check_counts(totals[architecture], analysis["completion"][architecture], architecture)
        eccentricities[architecture] = {}
        for e in (0.7, 0.9, 0.99):
            subset = [r for r in selected if r["eccentricity"] == e]
            eccentricities[architecture][str(e)] = counts(subset)
            check_counts(counts(subset), analysis["completion_by_eccentricity"][f"{architecture}|{e}"], f"{architecture},e={e}")
        for a in (0.75, 1.0, 1.25):
            subset = [r for r in selected if r["semi_major_axis"] == a]
            check_counts(counts(subset), analysis["completion_by_shell"][f"{architecture}|{a}"], f"{architecture},a={a}")
            for e in (0.7, 0.9, 0.99):
                grid.append({"architecture": architecture, "a": a, "e": e,
                             **counts([r for r in subset if r["eccentricity"] == e])})
        if architecture != "manual_r":
            checkpoints[architecture] = {}
            for seed in sorted({r["checkpoint_seed"] for r in selected}):
                subset = [r for r in selected if r["checkpoint_seed"] == seed]
                checkpoints[architecture][str(seed)] = counts(subset)
                check_counts(counts(subset), analysis["checkpoint_completion"][architecture][str(seed)], f"{architecture},seed={seed}")
            if len(checkpoints[architecture]) != 5:
                raise ValueError("Unexpected checkpoint count")
        groups = defaultdict(list)
        for r in selected:
            groups[(r.get("checkpoint_seed"), r["semi_major_axis"],
                    r["eccentricity"], r["initial_mean_anomaly"])].append(r)
        if any(len(group) != 4 for group in groups.values()):
            raise ValueError("Incomplete symmetry group")
        uniform = sum(len({r["completion_status"] == "complete" for r in group}) == 1
                      for group in groups.values())
        symmetries[architecture] = {"groups": len(groups), "uniform_completion": uniform}
        saved = analysis["symmetry_groups"][architecture]
        if (len(groups), uniform) != (saved["registered_groups"], saved["completion_uniform_groups"]):
            raise ValueError("Symmetry-group count differs")
        failures[architecture] = dict(Counter(r["completion_status"] for r in selected
                                             if r["completion_status"] != "complete"))
        if any("fixed point did not converge" not in r.get("error", "")
               for r in selected if r["completion_status"] != "complete"):
            raise ValueError("Failure is not the documented fixed-point iteration limit")

    extra_failures = [r for r in records if r["architecture"] == "raw"
                      and r["completion_status"] != "complete" and r["eccentricity"] != 0.99]
    if len(extra_failures) != 12 or any((r["semi_major_axis"], r["eccentricity"]) != (0.75, 0.9)
                                      for r in extra_failures):
        raise ValueError("Unexpected additional raw failure region")
    replay = read("results/expAC15/analysis.json")["by_parent"]["AC9"]
    if replay["registered"] != len(records) or replay["parent_statuses"] != replay["corrected_statuses"]:
        raise ValueError("Returned-residual replay aggregate differs")
    if any(key.split(" -> ")[0] != key.split(" -> ")[1] for key in replay["status_transitions"]):
        raise ValueError("Replay includes a completion-status transition")

    figure_paths = ["fig_checkpoint_generalisation.pdf", "fig_checkpoint_generalisation.png"]
    paths = [*source_paths.values(), analysis_path, "results/expAC15/analysis.json", *figure_paths,
             "new/paper_g2_jcp_new.tex", "new/audit_section_5_1.py"]
    claims = [
        {"id": "R1-overall", "tex_anchor": "sec:kepler-reliability", "value": totals, "verification": "recounted from saved raw records"},
        {"id": "R1-eccentricity", "tex_anchor": "fig:kepler-reliability", "value": eccentricities, "verification": "recounted from saved raw records"},
        {"id": "R1-checkpoints", "value": checkpoints, "verification": "recounted from saved raw records"},
        {"id": "R1-grid", "value": grid, "verification": "recounted from saved raw records"},
        {"id": "R1-symmetry", "value": symmetries, "verification": "regrouped saved raw records"},
        {"id": "R1-failure-status", "value": failures, "verification": "saved status and error fields checked"},
        {"id": "R1-bootstrap", "value": analysis["checkpoint_bootstrap"]["paired_invariant_minus_raw"], "verification": "saved bootstrap read; bootstrap draws not rerun"},
        {"id": "R1-residual-replay", "value": replay["status_transitions"], "verification": "saved replay analysis read; replay not rerun"},
    ]
    audit = {
        "scope": "partial claim trace for subsection 5.1 only",
        "tex_source": "paper_g2_jcp_new.tex",
        "experiments_rerun": False,
        "figure": "existing Matplotlib artifact inspected against saved data; not regenerated",
        "figure_source": "experiments/analyze_expAC9.py:make_figure",
        "source_sha256": {p: sha256(p) for p in paths},
        "claims": claims,
        "limits": ["No new training or rollout", "Not a complete manuscript audit",
                   "No population inference from correlated rollout totals",
                   "Figure hash records current artifact; it is not an independent regeneration"],
    }
    destination = HERE / "SECTION_5_1_CLAIMS.json"
    destination.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} saved records and {len(claims)} claim groups; wrote {destination.name}")


if __name__ == "__main__":
    main()
