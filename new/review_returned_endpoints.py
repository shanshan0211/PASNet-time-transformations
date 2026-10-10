"""Observe returned residuals under frozen AC9 stopping semantics.

Independent trajectories share batched field evaluations. This is a fresh
float64 replay, not reconstruction of unsaved historical endpoints. No
training or Hessian diagnosis is repeated; historical files are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import expAC9_end_to_end_monitor_generalisation as ac9
import expAC20_deployment_domain_training as ac20
from systems import Kepler2D

OUT = ROOT / "results" / "returned_endpoint_review"
SCENARIO = ("semi_major_axis", "eccentricity", "initial_mean_anomaly",
            "orientation_radians", "momentum_direction")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def groups():
    result = []
    records = read(ROOT / "results/expAC19/intervention.json")["records"]
    for architecture in ("raw", "invariant"):
        for seed in ac20.SEEDS:
            selected = [r for r in records if r["architecture"] == architecture
                        and r["checkpoint_seed"] == seed]
            result.append({"id": f"AC19_{architecture}_{seed}", "study": "AC19",
                           "arm": architecture, "seed": seed, "records": selected,
                           "checkpoint": f"results/expAC9/monitor_{architecture}_seed{seed}.pt"})
    for arm in ac20.ARMS:
        for seed in ac20.SEEDS:
            path = ROOT / f"results/expAC20/parts/{arm}_seed{seed}/rollouts.json"
            result.append({"id": f"AC20_{arm}_{seed}", "study": "AC20",
                           "arm": arm, "seed": seed, "records": read(path)["records"],
                           "checkpoint": f"results/expAC20/monitor_{arm}_seed{seed}.pt"})
    assert sum(len(g["records"]) for g in result) == 1332
    return result


def load_model(group):
    torch.set_default_dtype(torch.float64)
    model = (ac9.Monitor(group["arm"]) if group["study"] == "AC19"
             else ac20.build_monitor(group["arm"]))
    model.load_state_dict(torch.load(ROOT / group["checkpoint"],
                                     map_location="cpu", weights_only=True)["state_dict"])
    return model.eval()


def batch_step(state, step, field, tolerance=1e-12, maximum=100):
    """Each row returns at its own first update-based stopping iteration."""
    n = len(state)
    trial = state + step[:, None] * field(state, torch.arange(n))
    running = torch.ones(n, dtype=torch.bool)
    iterations = torch.zeros(n, dtype=torch.int64)
    evaluations = torch.ones(n, dtype=torch.int64)
    updates = torch.zeros(n)
    nonfinite = ~torch.isfinite(trial).all(dim=1)
    for k in range(1, maximum + 1):
        ids = torch.where(running)[0]
        if not len(ids):
            break
        proposal = state[ids] + step[ids, None] * field(
            (state[ids] + trial[ids]) * 0.5, ids)
        updates[ids] = (proposal - trial[ids]).abs().amax(dim=1)
        trial[ids] = proposal
        iterations[ids] = k
        evaluations[ids] += 1
        bad = ~torch.isfinite(proposal).all(dim=1)
        nonfinite[ids] = bad
        stopped = (updates[ids] < tolerance) | bad
        running[ids[stopped]] = False
    ok = (~running) & (~nonfinite)
    residual = torch.full((n,), float("inf"))
    finite = torch.where(torch.isfinite(trial).all(dim=1))[0]
    if len(finite):
        value = trial[finite] - state[finite] - step[finite, None] * field(
            (state[finite] + trial[finite]) * 0.5, finite)
        residual[finite] = value.abs().amax(dim=1)
    return trial, ok, nonfinite, iterations, evaluations, updates, residual


def replay(group, limit_steps=None):
    torch.set_num_threads(1)
    model = load_model(group)
    originals = group["records"]
    n = len(originals)
    system = Kepler2D()
    initial = []
    for r in originals:
        z = ac9.rotate_state(ac9.kepler_state(r["semi_major_axis"],
                            r["eccentricity"], r["initial_mean_anomaly"]),
                            r["orientation_radians"])
        if r["momentum_direction"] == "reversed":
            z[2:] *= -1
        initial.append(z)
    initial_np = np.stack(initial)
    state = torch.tensor(initial_np)
    energy0 = torch.tensor(system.H(initial_np[:, :2], initial_np[:, 2:]))
    momentum0 = state[:, 0] * state[:, 3] - state[:, 1] * state[:, 2]
    target = torch.tensor([2 * math.pi * r["semi_major_axis"]**1.5 for r in originals])
    stepsize = torch.tensor([r["intervention_dtau"] if group["study"] == "AC19"
                            else 0.04 for r in originals])
    cap = 2000000 if group["study"] == "AC19" else 100000
    clock = torch.zeros(n)
    clock_correction = torch.zeros(n)
    steps = torch.zeros(n, dtype=torch.int64)
    counts = torch.zeros(n, dtype=torch.int64)
    verification_calls = torch.zeros(n, dtype=torch.int64)
    iterations = torch.zeros(n, dtype=torch.int64)
    max_residual = torch.zeros(n)
    accepted_residual = torch.zeros(n)
    max_update = torch.zeros(n)
    max_energy = torch.zeros(n)
    max_momentum = torch.zeros(n)
    violations = torch.zeros(n, dtype=torch.int64)
    solves = torch.zeros(n, dtype=torch.int64)
    status = ["running"] * n
    failure_residual = [None] * n
    started = time.perf_counter()

    def solve(ids, z, h):
        def field(value, local):
            with torch.enable_grad():
                differentiable = value.detach().requires_grad_(True)
                extended = model(differentiable) * (
                    system.H_torch(differentiable) - energy0[ids[local]])
                grad = torch.autograd.grad(extended.sum(), differentiable)[0]
            return torch.cat([grad[:, 2:], -grad[:, :2]], dim=1).detach()
        value = batch_step(z, h, field)
        trial, ok, bad, its, calls, update, residual = value
        counts[ids] += calls
        verification_calls[ids] += torch.isfinite(trial).all(dim=1).to(torch.int64)
        solves[ids] += ok.to(torch.int64)
        max_update[ids] = torch.maximum(max_update[ids], torch.where(ok, update, 0.0))
        max_residual[ids] = torch.maximum(max_residual[ids], torch.where(ok, residual, 0.0))
        violations[ids] += (ok & ~(residual < 1e-12)).to(torch.int64)
        for local in torch.where(~ok)[0].tolist():
            i = int(ids[local])
            status[i] = "nonfinite" if bool(bad[local]) else "solver_failure"
            failure_residual[i] = (float(residual[local])
                                   if torch.isfinite(residual[local]) else None)
        return value

    while True:
        active = [i for i in range(n) if status[i] == "running"]
        if not active:
            break
        for i in active:
            if clock[i] >= target[i]:
                status[i] = "complete"
            elif steps[i] >= cap:
                status[i] = "step_cap"
            elif limit_steps is not None and steps[i] >= limit_steps:
                status[i] = "smoke_step_limit"
        ids = torch.tensor([i for i in active if status[i] == "running"], dtype=torch.int64)
        if not len(ids):
            continue
        trial, ok, _, its, _, _, residual = solve(ids, state[ids], stepsize[ids])
        good = torch.where(ok)[0]
        if not len(good):
            continue
        ids, trial, its, residual = ids[good], trial[good], its[good], residual[good]
        with torch.no_grad():
            dt = stepsize[ids] * model((state[ids] + trial) * 0.5)
        valid = torch.isfinite(dt) & (dt > 0)
        for i in ids[~valid].tolist():
            status[i] = "nonfinite"
        ids, trial, its, residual, dt = (v[valid] for v in (ids, trial, its, residual, dt))
        over = torch.where(clock[ids] + dt > target[ids])[0]
        if len(over):
            terminal_ids = ids[over]
            remaining = target[terminal_ids] - clock[terminal_ids]
            low = torch.zeros(len(over))
            high = stepsize[terminal_ids].clone()
            terminal_ok = torch.ones(len(over), dtype=torch.bool)
            candidate_dt = torch.zeros(len(over))
            for _ in range(40):
                local = torch.where(terminal_ok)[0]
                if not len(local):
                    break
                h = (low[local] + high[local]) * 0.5
                c, cok, _, ci, _, _, cr = solve(terminal_ids[local], state[terminal_ids[local]], h)
                terminal_ok[local[~cok]] = False
                local, c, ci, cr, h = (v[cok] for v in (local, c, ci, cr, h))
                with torch.no_grad():
                    cd = h * model((state[terminal_ids[local]] + c) * 0.5)
                candidate_dt[local] = cd
                less = cd < remaining[local]
                low[local[less]] = h[less]
                high[local[~less]] = h[~less]
                trial[over[local]], its[over[local]], residual[over[local]] = c, ci, cr
            terminal_good = over[terminal_ok]
            clock_correction[ids[terminal_good]] = (candidate_dt[terminal_ok]
                                                   - remaining[terminal_ok])
            dt[terminal_good] = remaining[terminal_ok]
        keep = torch.tensor([status[i] == "running" for i in ids.tolist()], dtype=torch.bool)
        ids, trial, its, residual, dt = (v[keep] for v in (ids, trial, its, residual, dt))
        state[ids] = trial
        clock[ids] += dt
        steps[ids] += 1
        iterations[ids] += its
        accepted_residual[ids] = torch.maximum(accepted_residual[ids], residual)
        max_energy[ids] = torch.maximum(max_energy[ids], (system.H_torch(trial) - energy0[ids]).abs())
        momentum = trial[:, 0] * trial[:, 3] - trial[:, 1] * trial[:, 2]
        max_momentum[ids] = torch.maximum(max_momentum[ids], (momentum - momentum0[ids]).abs())
    records = []
    for i, original in enumerate(originals):
        records.append({"parent_index": i, "study": group["study"], "arm": group["arm"],
                        "checkpoint_seed": group["seed"],
                        **{k: original[k] for k in SCENARIO},
                        "historical_status": original["completion_status"],
                        "completion_status": status[i], "steps": int(steps[i]),
                        "historical_steps": original["steps"],
                        "returned_solve_count": int(solves[i]),
                        "returned_residual_violations": int(violations[i]),
                        "maximum_all_returned_residual_inf": float(max_residual[i]),
                        "maximum_accepted_endpoint_residual_inf": float(accepted_residual[i]),
                        "failure_endpoint_residual_inf": failure_residual[i],
                        "maximum_stopping_update_inf": float(max_update[i]),
                        "nonlinear_iterations_accepted_steps": int(iterations[i]),
                        "original_algorithm_field_evaluations": int(counts[i]),
                        "additional_residual_field_evaluations": int(verification_calls[i]),
                        "max_energy_error": float(max_energy[i]),
                        "max_angular_momentum_error": float(max_momentum[i]),
                        "physical_time": float(clock[i]),
                        "terminal_candidate_clock_discrepancy": float(clock_correction[i]),
                        "recorded_final_time_error": float(abs(clock[i] - target[i])),
                        "final_state_closure_error": float(torch.linalg.vector_norm(state[i] - torch.tensor(initial_np[i]))),
                        "final_state": state[i].tolist()})
    return {"group": group["id"], "wall_seconds": time.perf_counter() - started,
            "limit_steps": limit_steps, "records": records}


def seal():
    sources = ["new/review_returned_endpoints.py", "src/systems.py", "src/models/pasnet.py",
               "experiments/expAC9_end_to_end_monitor_generalisation.py",
               "experiments/expAC20_deployment_domain_training.py",
               "results/expAC9/preregistration.json", "results/expAC19/preregistration.json",
               "results/expAC19/intervention.json", "results/expAC20/preregistration.json"]
    gs = groups()
    sources += [g["checkpoint"] for g in gs]
    sources += [f"results/expAC20/parts/{g['arm']}_seed{g['seed']}/rollouts.json"
                for g in gs if g["study"] == "AC20"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "observational returned-residual replay at original update stop",
                "solver_tolerance": 1e-12, "maximum_iterations": 100,
                "terminal_bisections": 40, "expected_records": 1332,
                "batching": "independent rows; no residual enforcement or training",
                "sources": [{"path": p, "sha256": digest(ROOT / p)} for p in sources],
                "environment": {"python": sys.version, "torch": torch.__version__,
                                "numpy": np.__version__, "platform": platform.platform()},
                "groups": [{k: v for k, v in g.items() if k != "records"} for g in gs]}
    path = OUT / "manifest.json"
    if path.exists():
        old = read(path)
        if old["sources"] != manifest["sources"] or old["environment"] != manifest["environment"]:
            raise ValueError("manifest drift: do not silently resume changed inputs/runtime")
        return old
    atomic(path, manifest)
    return manifest


def analyze():
    manifest = read(OUT / "manifest.json")
    records = []
    for group in manifest["groups"]:
        part = read(OUT / "parts" / (group["id"] + ".json"))
        assert part["limit_steps"] is None
        records.extend(part["records"])
    assert len(records) == 1332
    summary = {"records": len(records), "by_study_arm": {},
               "all_inputs_unchanged": all(digest(ROOT / s["path"]) == s["sha256"]
                                           for s in manifest["sources"])}
    for study, arm in sorted({(r["study"], r["arm"]) for r in records}):
        rs = [r for r in records if (r["study"], r["arm"]) == (study, arm)]
        summary["by_study_arm"][f"{study}_{arm}"] = {
            "n": len(rs), "statuses": dict(Counter(r["completion_status"] for r in rs)),
            "status_changes": sum(r["completion_status"] != r["historical_status"] for r in rs),
            "step_count_changes": sum(r["steps"] != r["historical_steps"] for r in rs),
            "returned_residual_violations": sum(r["returned_residual_violations"] for r in rs),
            "returned_solves": sum(r["returned_solve_count"] for r in rs),
            "maximum_accepted_residual": max(r["maximum_accepted_endpoint_residual_inf"] for r in rs),
            "maximum_all_returned_residual": max(r["maximum_all_returned_residual_inf"] for r in rs),
            "maximum_terminal_clock_discrepancy": max(abs(r["terminal_candidate_clock_discrepancy"]) for r in rs)}
    atomic(OUT / "analysis.json", summary)
    print(json.dumps(summary, indent=2))


def parity():
    comparisons = []
    evaluation0 = read(ROOT / "results/expAC9/preregistration.json")["evaluation"]
    for g in (groups()[0], groups()[10], groups()[15], groups()[20]):
        for index in (0, len(g["records"]) - 1):
            original = g["records"][index]
            small = {**g, "records": [original]}
            observed = replay(small)["records"][0]
            evaluation = json.loads(json.dumps(evaluation0))
            if g["study"] == "AC19":
                evaluation.update(dtau=original["intervention_dtau"], maximum_steps=2000000)
            reference = ac9.run_rollout(load_model(g), original["architecture"],
                                       g["seed"], {k: original[k] for k in SCENARIO}, evaluation)
            comparisons.append({"group": g["id"], "parent_index": index,
                                "status_equal": observed["completion_status"] == reference["completion_status"],
                                "steps_equal": observed["steps"] == reference["steps"],
                                "energy_error_difference": observed["max_energy_error"] - reference["max_energy_error"],
                                "closure_difference": observed["final_state_closure_error"] - reference["final_state_closure_error"]})
    atomic(OUT / "scalar_parity.json", comparisons)
    print(json.dumps(comparisons, indent=2))
    if not all(r["status_equal"] and r["steps_equal"] and
               abs(r["energy_error_difference"]) < 1e-9 and abs(r["closure_difference"]) < 1e-8
               for r in comparisons):
        raise ValueError("scalar parity failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    seal()
    if args.parity:
        parity()
        return
    gs = groups()
    if args.smoke:
        for group in (gs[0], gs[10], gs[15], gs[20]):
            atomic(OUT / "smoke" / (group["id"] + ".json"), replay(group, 3))
        print("Four checkpoint groups: three-step smoke complete")
        return
    pending = [g for g in gs if not (OUT / "parts" / (g["id"] + ".json")).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(replay, g): g for g in pending}
        for future in as_completed(futures):
            group = futures[future]
            value = future.result()
            atomic(OUT / "parts" / (group["id"] + ".json"), value)
            print(f"Saved {group['id']}: {len(value['records'])} records, "
                  f"{value['wall_seconds']:.1f}s", flush=True)
    analyze()


if __name__ == "__main__":
    main()
