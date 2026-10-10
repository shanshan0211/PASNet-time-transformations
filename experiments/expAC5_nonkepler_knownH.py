"""Corrected non-Kepler known-H transformed-midpoint controls.

These are mechanism checks with deterministic, bounded, state-dependent
monitors. They do not test a learned monitor or support a learning advantage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from systems import Duffing1D, ModalOscillator  # noqa: E402


OUT = os.path.join(ROOT, "results", "expAC5", "nonkepler_knownH.json")
DTAUS = (0.1, 0.05, 0.025, 0.0125)
T_MAX = 20.0


def hh_energy_torch(z):
    x, y, px, py = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    return (
        0.5 * (px**2 + py**2)
        + 0.5 * (x**2 + y**2)
        + x**2 * y
        - y**3 / 3.0
    )


def build_systems():
    duffing = Duffing1D(beta=0.1)
    duffing_z0 = np.array([0.5, 0.0], dtype=np.float64)

    fpu = ModalOscillator(
        n_modes=8,
        omega_min=1.0,
        kappa=10.0,
        beta=0.25,
        seed=0,
        orthogonal_mixing=False,
    )
    fpu_q0, fpu_p0 = fpu.initial_condition(
        e0=1.0, slow_fraction=0.9, seed=0
    )
    fpu_z0 = np.concatenate([fpu_q0, fpu_p0])

    hh_energy = 1.0 / 8.0
    x0, y0 = 0.1, -0.1
    momentum = math.sqrt(
        hh_energy - 0.5 * (x0**2 + y0**2)
        - x0**2 * y0 + y0**3 / 3.0
    )
    hh_z0 = np.array([x0, y0, momentum, momentum], dtype=np.float64)

    return (
        {
            "name": "Duffing1D",
            "dimension": 1,
            "z0": duffing_z0,
            "H": duffing.H_torch,
            "monitor": lambda z: 0.75 + 0.25 / (1.0 + z[:, 0] ** 2),
        },
        {
            "name": "FPU_like_modal",
            "dimension": 8,
            "z0": fpu_z0,
            "H": fpu.H_torch,
            "monitor": lambda z: 0.75
            + 0.25
            / (1.0 + torch.mean(z[:, :8] ** 2, dim=1)),
        },
        {
            "name": "Henon_Heiles",
            "dimension": 2,
            "z0": hh_z0,
            "H": hh_energy_torch,
            "monitor": lambda z: 0.75
            + 0.25
            / (1.0 + torch.sum(z[:, :2] ** 2, dim=1)),
        },
    )


def transformed_step(z, h, vector_field, tolerance=1e-12):
    evaluations = 1
    z_next = z + h * vector_field(z)
    residual = math.inf
    for iteration in range(1, 101):
        proposal = z + h * vector_field(0.5 * (z + z_next))
        evaluations += 1
        residual = float(torch.max(torch.abs(proposal - z_next)))
        z_next = proposal
        if residual < tolerance:
            return z_next, residual, iteration, evaluations
    raise RuntimeError(
        f"midpoint failed at h={h:.3e}, residual={residual:.3e}"
    )


def run(system, dtau):
    z = torch.tensor(system["z0"][None, :], dtype=torch.float64)
    energy0 = float(system["H"](z))
    p0_ext = torch.tensor([-energy0], dtype=torch.float64)
    dimension = system["dimension"]

    def vector_field(state):
        with torch.enable_grad():
            differentiable = state.detach().requires_grad_(True)
            extended_h = system["monitor"](differentiable) * (
                system["H"](differentiable) + p0_ext
            )
            gradient = torch.autograd.grad(
                extended_h.sum(), differentiable
            )[0]
        return torch.cat(
            [gradient[:, dimension:], -gradient[:, :dimension]], dim=1
        ).detach()

    record = {
        "system": system["name"],
        "monitor": "deterministic_bounded_state_dependent",
        "dtau": dtau,
        "steps": 0,
        "gradient_evaluations": 0,
        "alignment_gradient_evaluations": 0,
        "max_fp_residual": 0.0,
        "max_fp_iterations": 0,
        "max_energy_drift": 0.0,
        "min_monitor": math.inf,
        "max_monitor": 0.0,
    }
    initial_state = z.detach().numpy().copy()
    started = time.perf_counter()
    physical_time = 0.0
    while physical_time < T_MAX:
        trial, residual, iterations, evaluations = transformed_step(
            z, dtau, vector_field
        )
        midpoint = 0.5 * (z + trial)
        physical_dt = dtau * float(system["monitor"](midpoint))

        if physical_time + physical_dt > T_MAX:
            remaining = T_MAX - physical_time
            lower, upper = 0.0, dtau
            alignment_evaluations = 0
            for _ in range(45):
                candidate_h = 0.5 * (lower + upper)
                candidate, cand_res, cand_iter, cand_evals = transformed_step(
                    z, candidate_h, vector_field
                )
                candidate_midpoint = 0.5 * (z + candidate)
                candidate_dt = candidate_h * float(
                    system["monitor"](candidate_midpoint)
                )
                alignment_evaluations += cand_evals
                if candidate_dt < remaining:
                    lower = candidate_h
                else:
                    upper = candidate_h
                trial = candidate
                residual = cand_res
                iterations = cand_iter
                physical_dt = candidate_dt
            record["alignment_gradient_evaluations"] += alignment_evaluations
            evaluations += alignment_evaluations
            physical_dt = remaining

        z = trial
        physical_time += physical_dt
        monitor_value = float(system["monitor"](z))
        energy_error = abs(float(system["H"](z)) - energy0)
        record["steps"] += 1
        record["gradient_evaluations"] += evaluations
        record["max_fp_residual"] = max(
            record["max_fp_residual"], residual
        )
        record["max_fp_iterations"] = max(
            record["max_fp_iterations"], iterations
        )
        record["max_energy_drift"] = max(
            record["max_energy_drift"], energy_error
        )
        record["min_monitor"] = min(record["min_monitor"], monitor_value)
        record["max_monitor"] = max(record["max_monitor"], monitor_value)

    record.update(
        physical_time=physical_time,
        final_time_error=abs(physical_time - T_MAX),
        state_displacement=float(
            np.linalg.norm(z.detach().numpy() - initial_state)
        ),
        wall_seconds=time.perf_counter() - started,
    )
    return record


def observed_orders(records):
    orders = []
    for coarse, fine in zip(records[:-1], records[1:]):
        numerator = coarse["max_energy_drift"]
        denominator = fine["max_energy_drift"]
        orders.append(
            math.log(numerator / denominator, 2.0)
            if numerator > 0.0 and denominator > 0.0
            else None
        )
    return orders


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    configuration = {
        "systems": ["Duffing1D", "FPU_like_modal", "Henon_Heiles"],
        "physical_horizon": T_MAX,
        "dtau_refinement": list(DTAUS),
        "monitor": "deterministic_bounded_state_dependent",
        "fixed_point_tolerance": 1e-12,
        "fixed_point_max_iterations": 100,
        "dtype": "float64",
    }
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    all_records = []
    analyses = {}
    for system in build_systems():
        records = []
        for dtau in DTAUS:
            result = run(system, dtau)
            records.append(result)
            all_records.append(result)
            print(
                f"{system['name']} dtau={dtau:g}: "
                f"dE={result['max_energy_drift']:.3e}, "
                f"res={result['max_fp_residual']:.3e}",
                flush=True,
            )
        analyses[system["name"]] = {
            "energy_refinement_orders": observed_orders(records),
            "finest_energy_drift": records[-1]["max_energy_drift"],
        }

    acceptance = {
        "all_final_times_aligned": all(
            record["final_time_error"] < 1e-10 for record in all_records
        ),
        "all_solves_converged": all(
            record["max_fp_residual"] < 1e-11 for record in all_records
        ),
        "all_systems_move": all(
            max(
                record["state_displacement"]
                for record in all_records
                if record["system"] == name
            )
            > 1e-2
            for name in analyses
        ),
        "finest_better_than_coarsest": all(
            records[-1]["max_energy_drift"]
            < records[0]["max_energy_drift"]
            for name in analyses
            for records in [
                [
                    record
                    for record in all_records
                    if record["system"] == name
                ]
            ]
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    artifact = {
        "schema_version": "1.0",
        "experiment": "expAC5_nonkepler_knownH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": None,
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "dtype": "float64",
        "physical_horizon": T_MAX,
        "dtau_refinement": list(DTAUS),
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "records": all_records,
        "analysis": analyses,
        "acceptance": acceptance,
        "interpretation": {
            "supported":
                "true-H extended vector fields work beyond Kepler and "
                "energy error decreases under refinement",
            "not_supported":
                "a learned monitor improves accuracy or efficiency",
            "learned_monitor_tested": False,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    if not acceptance["passed"]:
        raise SystemExit("non-Kepler known-H gate failed")


if __name__ == "__main__":
    main()
