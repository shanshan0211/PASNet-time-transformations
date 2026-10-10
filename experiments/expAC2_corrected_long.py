"""Corrected long-horizon, matched-final-time Kepler controls.

All methods use the attractive Kepler Hamiltonian, float64, the same initial
condition, and exactly the same final physical time. Every implicit solve
records its residual and iteration count. The adaptive physical-time method
is an accuracy control; state-dependent step selection is not described as
globally symplectic.
"""

from __future__ import annotations

import argparse
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
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from kepler_integrators import implicit_midpoint_step, kepler_energy  # noqa: E402
from models.pasnet import PASNet  # noqa: E402
from systems import Kepler2D  # noqa: E402


PERIOD = 2.0 * math.pi
SCHEMA_VERSION = "2.0-corrected"


def angular_momentum(q, p):
    return float(q[0] * p[1] - q[1] * p[0])


def state_error(q, p, q0, p0):
    return float(np.linalg.norm(np.concatenate([q - q0, p - p0])))


def update_invariants(record, q, p, energy0, momentum0):
    record["max_energy_drift"] = max(
        record["max_energy_drift"], abs(kepler_energy(q, p) - energy0)
    )
    record["max_angular_momentum_error"] = max(
        record["max_angular_momentum_error"],
        abs(angular_momentum(q, p) - momentum0),
    )
    radius = float(np.linalg.norm(q))
    record["min_radius"] = min(record["min_radius"], radius)
    record["max_radius"] = max(record["max_radius"], radius)


def run_direct(system, q0, p0, dt, t_max):
    q, p = q0.copy(), p0.copy()
    energy0 = kepler_energy(q, p)
    momentum0 = angular_momentum(q, p)
    record = {
        "method": "direct_implicit_midpoint",
        "dt": dt,
        "steps": 0,
        "gradient_evaluations": 0,
        "rejected_steps": 0,
        "nonconverged_steps": 0,
        "max_fp_residual": 0.0,
        "max_fp_iterations": 0,
        "max_energy_drift": 0.0,
        "max_angular_momentum_error": 0.0,
        "min_radius": float(np.linalg.norm(q)),
        "max_radius": float(np.linalg.norm(q)),
    }
    started = time.perf_counter()
    t = 0.0
    while t < t_max:
        h = min(dt, t_max - t)
        q, p, info = implicit_midpoint_step(
            q, p, h, max_iter=100, tol=1e-13, require_convergence=True
        )
        t += h
        record["steps"] += 1
        record["gradient_evaluations"] += info.iterations + 1
        record["max_fp_residual"] = max(
            record["max_fp_residual"], info.residual
        )
        record["max_fp_iterations"] = max(
            record["max_fp_iterations"], info.iterations
        )
        update_invariants(record, q, p, energy0, momentum0)
    record.update(
        physical_time=t,
        final_time_error=abs(t - t_max),
        final_state_error=state_error(q, p, q0, p0),
        wall_seconds=time.perf_counter() - started,
    )
    return record


def run_adaptive(system, q0, p0, rtol, t_max):
    del system
    q, p = q0.copy(), p0.copy()
    energy0 = kepler_energy(q, p)
    momentum0 = angular_momentum(q, p)
    atol = 1e-12
    h = 0.005
    record = {
        "method": "adaptive_step_doubling_midpoint",
        "rtol": rtol,
        "atol": atol,
        "steps": 0,
        "gradient_evaluations": 0,
        "rejected_steps": 0,
        "nonconverged_steps": 0,
        "max_fp_residual": 0.0,
        "max_fp_iterations": 0,
        "max_energy_drift": 0.0,
        "max_angular_momentum_error": 0.0,
        "min_radius": float(np.linalg.norm(q)),
        "max_radius": float(np.linalg.norm(q)),
        "min_accepted_dt": math.inf,
        "max_accepted_dt": 0.0,
    }
    started = time.perf_counter()
    t = 0.0
    while t < t_max:
        h = min(h, t_max - t)
        q1, p1, info1 = implicit_midpoint_step(
            q, p, h, max_iter=100, tol=1e-13, require_convergence=True
        )
        qa, pa, info2 = implicit_midpoint_step(
            q, p, h / 2, max_iter=100, tol=1e-13, require_convergence=True
        )
        q2, p2, info3 = implicit_midpoint_step(
            qa, pa, h / 2, max_iter=100, tol=1e-13, require_convergence=True
        )
        infos = (info1, info2, info3)
        record["gradient_evaluations"] += sum(
            info.iterations + 1 for info in infos
        )
        record["max_fp_residual"] = max(
            record["max_fp_residual"], *(info.residual for info in infos)
        )
        record["max_fp_iterations"] = max(
            record["max_fp_iterations"], *(info.iterations for info in infos)
        )

        scale = atol + rtol * np.maximum(
            np.abs(np.concatenate([q, p])),
            np.abs(np.concatenate([q2, p2])),
        )
        diff = np.concatenate([q1 - q2, p1 - p2]) / 3.0
        error = float(np.sqrt(np.mean((diff / scale) ** 2)))
        factor = min(5.0, max(0.2, 0.9 * max(error, 1e-16) ** (-1.0 / 3.0)))
        if error <= 1.0:
            q, p = q2, p2
            t += h
            record["steps"] += 1
            record["min_accepted_dt"] = min(record["min_accepted_dt"], h)
            record["max_accepted_dt"] = max(record["max_accepted_dt"], h)
            update_invariants(record, q, p, energy0, momentum0)
        else:
            record["rejected_steps"] += 1
        h = min(0.05, max(1e-7, h * factor))

    record.update(
        physical_time=t,
        final_time_error=abs(t - t_max),
        final_state_error=state_error(q, p, q0, p0),
        wall_seconds=time.perf_counter() - started,
    )
    return record


def h_features(z):
    q = z[:, :2]
    r = torch.linalg.vector_norm(q, dim=1, keepdim=True)
    return torch.cat([r, 1.0 / r], dim=1)


def load_model():
    checkpoint = os.path.join(RESULTS, "expE2", "pasnet_weights.pt")
    model = PASNet(
        dim=2,
        hidden_H=128,
        n_hidden_H=3,
        hidden_g=64,
        n_hidden_g=2,
        g_max=2.0,
        h_features=h_features,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.H_net.load_state_dict(state["H_net"])
    model.g_net.load_state_dict(state["g_net"])
    model = model.double().eval()
    with open(checkpoint, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return model, checkpoint, digest


def transformed_step(z, h, vector_field, tol=1e-12):
    evaluations = 1
    z_next = z + h * vector_field(z)
    residual = math.inf
    for iteration in range(1, 101):
        proposal = z + h * vector_field(0.5 * (z + z_next))
        evaluations += 1
        residual = float(torch.max(torch.abs(proposal - z_next)))
        z_next = proposal
        if residual < tol:
            return z_next, residual, iteration, evaluations
    raise RuntimeError(
        f"transformed midpoint failed at h={h:.3e}, residual={residual:.3e}"
    )


def run_transformed(system, q0, p0, model, monitor, dtau, t_max):
    z = torch.tensor(
        np.concatenate([q0, p0])[None, :], dtype=torch.float64
    )
    energy0 = float(system.H(q0, p0))
    momentum0 = angular_momentum(q0, p0)
    p0_ext = torch.tensor([-energy0], dtype=torch.float64)

    def g_fn(zz):
        if monitor == "manual_r":
            return torch.linalg.vector_norm(zz[:, :2], dim=1)
        return model.g(zz)

    def vector_field(zz):
        with torch.enable_grad():
            zr = zz.detach().requires_grad_(True)
            k = g_fn(zr) * (system.H_torch(zr) + p0_ext)
            grad_k = torch.autograd.grad(k.sum(), zr)[0]
        return torch.cat([grad_k[:, 2:], -grad_k[:, :2]], dim=1).detach()

    record = {
        "method": "extended_implicit_midpoint",
        "monitor": monitor,
        "dtau": dtau,
        "steps": 0,
        "gradient_evaluations": 0,
        "alignment_gradient_evaluations": 0,
        "rejected_steps": 0,
        "nonconverged_steps": 0,
        "max_fp_residual": 0.0,
        "max_fp_iterations": 0,
        "max_K_residual": 0.0,
        "max_energy_drift": 0.0,
        "max_angular_momentum_error": 0.0,
        "min_radius": float(np.linalg.norm(q0)),
        "max_radius": float(np.linalg.norm(q0)),
        "min_physical_dt": math.inf,
        "max_physical_dt": 0.0,
    }
    started = time.perf_counter()
    t = 0.0
    while t < t_max:
        z_trial, residual, iterations, evaluations = transformed_step(
            z, dtau, vector_field
        )
        midpoint = 0.5 * (z + z_trial)
        physical_dt = dtau * float(g_fn(midpoint).detach())

        if t + physical_dt > t_max:
            remaining = t_max - t
            lower, upper = 0.0, dtau
            for _ in range(45):
                h_mid = 0.5 * (lower + upper)
                candidate, cand_res, cand_iter, cand_evals = transformed_step(
                    z, h_mid, vector_field
                )
                candidate_midpoint = 0.5 * (z + candidate)
                candidate_dt = h_mid * float(g_fn(candidate_midpoint).detach())
                record["alignment_gradient_evaluations"] += cand_evals
                if candidate_dt < remaining:
                    lower = h_mid
                else:
                    upper = h_mid
                z_trial, residual, iterations = candidate, cand_res, cand_iter
                physical_dt = candidate_dt
            evaluations += record["alignment_gradient_evaluations"]
            physical_dt = remaining

        z = z_trial
        t += physical_dt
        record["steps"] += 1
        record["gradient_evaluations"] += evaluations
        record["max_fp_residual"] = max(record["max_fp_residual"], residual)
        record["max_fp_iterations"] = max(
            record["max_fp_iterations"], iterations
        )
        record["min_physical_dt"] = min(
            record["min_physical_dt"], physical_dt
        )
        record["max_physical_dt"] = max(
            record["max_physical_dt"], physical_dt
        )

        q = z.detach().numpy()[0, :2]
        p = z.detach().numpy()[0, 2:]
        update_invariants(record, q, p, energy0, momentum0)
        k_residual = abs(
            float(g_fn(z).detach()) * (kepler_energy(q, p) - energy0)
        )
        record["max_K_residual"] = max(
            record["max_K_residual"], k_residual
        )

    q = z.detach().numpy()[0, :2]
    p = z.detach().numpy()[0, 2:]
    record.update(
        physical_time=t,
        final_time_error=abs(t - t_max),
        final_state_error=state_error(q, p, q0, p0),
        wall_seconds=time.perf_counter() - started,
    )
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=float, default=100.0)
    parser.add_argument("--direct-dt", type=float, nargs="+", default=[0.005])
    parser.add_argument(
        "--adaptive-rtol", type=float, nargs="+", default=[1e-7]
    )
    parser.add_argument(
        "--dtau", type=float, nargs="+", default=[0.04, 0.02]
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS, "expAC2", "corrected_long.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    system = Kepler2D(0.9)
    q0, p0 = system.initial_condition()
    model, checkpoint, checkpoint_sha256 = load_model()
    t_max = args.periods * PERIOD
    records = []

    for dt in args.direct_dt:
        print(f"direct midpoint dt={dt:g}", flush=True)
        records.append(run_direct(system, q0, p0, dt, t_max))
    for rtol in args.adaptive_rtol:
        print(f"adaptive midpoint rtol={rtol:g}", flush=True)
        records.append(run_adaptive(system, q0, p0, rtol, t_max))
    for monitor in ("manual_r", "learned_g"):
        for dtau in args.dtau:
            print(f"transformed {monitor} dtau={dtau:g}", flush=True)
            records.append(
                run_transformed(
                    system, q0, p0, model, monitor, dtau, t_max
                )
            )

    acceptance = {
        "all_finite": all(
            np.isfinite(record["max_energy_drift"]) for record in records
        ),
        "all_solvers_converged": all(
            record["nonconverged_steps"] == 0 for record in records
        ),
        "matched_final_time": all(
            record["final_time_error"] < 1e-10 for record in records
        ),
        "residual_lt_1e-11": all(
            record["max_fp_residual"] < 1e-11 for record in records
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "expAC2_corrected_long",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": "Kepler2D(e=0.9,a=1)",
        "dtype": "float64",
        "physical_horizon": t_max,
        "periods": args.periods,
        "initial_condition": {
            "q": q0.tolist(),
            "p": p0.tolist(),
            "energy": float(system.H(q0, p0)),
            "angular_momentum": angular_momentum(q0, p0),
        },
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "checkpoint_sha256": checkpoint_sha256,
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "configuration": {
            "direct_dt": args.direct_dt,
            "adaptive_rtol": args.adaptive_rtol,
            "dtau": args.dtau,
            "fp_tolerance": 1e-12,
            "fp_max_iterations": 100,
        },
        "records": records,
        "acceptance": acceptance,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(json.dumps(artifact, indent=2), flush=True)
    if not acceptance["passed"]:
        raise SystemExit("corrected long-horizon gate failed")


if __name__ == "__main__":
    main()
