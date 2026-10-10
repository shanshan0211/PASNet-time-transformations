"""Residual-enforced successor to the observational endpoint review."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json

import torch
import review_returned_endpoints as review

OUT = review.ROOT / "results/returned_endpoint_review/verified"


def verified_step(state, step, field, tolerance=1e-12, maximum=100):
    n = len(state)
    trial = state + step[:, None] * field(state, torch.arange(n))
    running = torch.ones(n, dtype=torch.bool)
    iterations = torch.zeros(n, dtype=torch.int64)
    evaluations = torch.ones(n, dtype=torch.int64)
    updates = torch.zeros(n)
    nonfinite = torch.zeros(n, dtype=torch.bool)
    residual = torch.full((n,), float("inf"))
    for k in range(1, maximum + 1):
        ids = torch.where(running)[0]
        if not len(ids):
            break
        proposal = state[ids] + step[ids, None] * field((state[ids] + trial[ids]) * 0.5, ids)
        updates[ids] = (proposal - trial[ids]).abs().amax(dim=1)
        trial[ids] = proposal
        iterations[ids] = k
        evaluations[ids] += 1
        bad = ~torch.isfinite(proposal).all(dim=1)
        nonfinite[ids] = bad
        running[ids[bad]] = False
        trigger = ids[(updates[ids] < tolerance) & ~bad]
        if len(trigger):
            r = trial[trigger] - state[trigger] - step[trigger, None] * field(
                (state[trigger] + trial[trigger]) * 0.5, trigger)
            residual[trigger] = r.abs().amax(dim=1)
            evaluations[trigger] += 1
            running[trigger[residual[trigger] < tolerance]] = False
    remaining = torch.where(running)[0]
    if len(remaining):
        r = trial[remaining] - state[remaining] - step[remaining, None] * field(
            (state[remaining] + trial[remaining]) * 0.5, remaining)
        residual[remaining] = r.abs().amax(dim=1)
        evaluations[remaining] += 1
    return trial, (~running & ~nonfinite), nonfinite, iterations, evaluations, updates, residual


def worker(group):
    review.batch_step = verified_step
    result = review.replay(group)
    result["solver"] = "update trigger plus verified returned residual acceptance"
    result["cost_note"] = "field count includes acceptance verification; additional count in base replay is bookkeeping only"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    parent = review.read(review.OUT / "manifest.json")
    assert all(review.digest(review.ROOT / s["path"]) == s["sha256"] for s in parent["sources"])
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "parent_manifest_sha256": review.digest(review.OUT / "manifest.json"),
                "sources": [{"path": "new/replay_verified_endpoints.py",
                             "sha256": review.digest(__file__)}],
                "purpose": "full residual-enforced replay; same checkpoints, grid, step and budgets",
                "tolerance": 1e-12, "maximum_iterations": 100, "expected_records": 1332}
    path = OUT / "manifest.json"
    if path.exists():
        existing = review.read(path)
        assert existing["sources"] == manifest["sources"]
        assert existing["parent_manifest_sha256"] == manifest["parent_manifest_sha256"]
    else:
        review.atomic(path, manifest)
    pending = [g for g in review.groups() if not (OUT / "parts" / (g["id"] + ".json")).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, g): g for g in pending}
        for future in as_completed(futures):
            g = futures[future]
            value = future.result()
            review.atomic(OUT / "parts" / (g["id"] + ".json"), value)
            print(f"Verified {g['id']}: {value['wall_seconds']:.1f}s", flush=True)
    review.OUT = OUT
    # The original source manifest remains authoritative for the grouped analysis.
    review.atomic(OUT / "analysis_inputs.json", parent)
    records = [r for g in review.groups() for r in review.read(OUT / "parts" / (g["id"] + ".json"))["records"]]
    assert len(records) == 1332
    summary = {"n": len(records), "by_arm": {}}
    for study, arm in sorted({(r["study"], r["arm"]) for r in records}):
        rs = [r for r in records if (r["study"], r["arm"]) == (study, arm)]
        summary["by_arm"][f"{study}_{arm}"] = {
            "n": len(rs), "complete": sum(r["completion_status"] == "complete" for r in rs),
            "status_changes": sum(r["completion_status"] != r["historical_status"] for r in rs),
            "step_changes": sum(r["steps"] != r["historical_steps"] for r in rs),
            "returned_solves": sum(r["returned_solve_count"] for r in rs),
            "violations": sum(r["returned_residual_violations"] for r in rs),
            "maximum_returned_residual": max(r["maximum_all_returned_residual_inf"] for r in rs),
            "maximum_terminal_clock_discrepancy": max(abs(r["terminal_candidate_clock_discrepancy"]) for r in rs)}
    review.atomic(OUT / "analysis.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
