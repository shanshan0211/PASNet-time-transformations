"""Corrected modal-oscillator control across stiffness ratios.

For the quadratic modal Hamiltonian, implicit midpoint has a closed-form
Cayley step and preserves the exact quadratic energy up to roundoff at any
step size. This experiment records state/phase error as well as energy error,
showing why energy drift alone cannot establish useful accuracy at high
stiffness.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from systems import ModalOscillator  # noqa: E402


KAPPAS = (10.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
N_MODES = 4
E0 = 1.0
T_MAX = 20.0 * 2.0 * math.pi
OUT = os.path.join(ROOT, "results", "expAC4", "modal_correctness.json")


def energy(q, p, omega2):
    return float(0.5 * np.sum(p**2 + omega2 * q**2))


def exact_state(q0, p0, omegas, t):
    phase = omegas * t
    q = q0 * np.cos(phase) + (p0 / omegas) * np.sin(phase)
    p = p0 * np.cos(phase) - (omegas * q0) * np.sin(phase)
    return q, p


def midpoint_step(q, p, omega2, h):
    a = 0.25 * h**2 * omega2
    denominator = 1.0 + a
    q_next = ((1.0 - a) * q + h * p) / denominator
    p_next = (-h * omega2 * q + (1.0 - a) * p) / denominator
    return q_next, p_next


def verlet_step(q, p, omega2, h):
    p_half = p - 0.5 * h * omega2 * q
    q_next = q + h * p_half
    p_next = p_half - 0.5 * h * omega2 * q_next
    return q_next, p_next


def integrate(method, q0, p0, omega2, h_nominal, t_max):
    q, p = q0.copy(), p0.copy()
    initial_energy = energy(q, p, omega2)
    max_drift = 0.0
    steps = int(math.ceil(t_max / h_nominal))
    sample_every = max(1, steps // 10000)
    started = time.perf_counter()
    t = 0.0
    for index in range(steps):
        h = min(h_nominal, t_max - t)
        if method == "implicit_midpoint":
            q, p = midpoint_step(q, p, omega2, h)
        else:
            q, p = verlet_step(q, p, omega2, h)
        t += h
        if index % sample_every == 0 or index == steps - 1:
            max_drift = max(
                max_drift, abs(energy(q, p, omega2) - initial_energy)
            )
    return q, p, {
        "method": method,
        "dt": h_nominal,
        "steps": steps,
        "energy_samples": 1 + steps // sample_every,
        "max_energy_drift": max_drift,
        "wall_seconds": time.perf_counter() - started,
        "final_time_error": abs(t - t_max),
    }


def main():
    configuration = {
        "kappas": list(KAPPAS),
        "n_modes": N_MODES,
        "energy": E0,
        "physical_horizon": T_MAX,
        "initial_condition_seed": 1,
        "fixed_dt": 0.01,
        "resolved_dt_rule": "0.1/kappa",
        "dtype": "float64",
    }
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    records = []
    for kappa in KAPPAS:
        system = ModalOscillator(
            n_modes=N_MODES,
            omega_min=1.0,
            kappa=kappa,
            beta=0.0,
            orthogonal_mixing=False,
            seed=0,
        )
        q0, p0 = system.initial_condition(
            e0=E0, slow_fraction=0.9, seed=1
        )
        exact_q, exact_p = exact_state(q0, p0, system.omegas, T_MAX)
        exact_vector = np.concatenate([exact_q, exact_p])
        scales = np.concatenate(
            [
                np.maximum(np.abs(exact_q), 1.0 / system.omegas),
                np.maximum(np.abs(exact_p), 1.0),
            ]
        )

        configurations = (
            ("implicit_midpoint", 0.01, "fixed_cost"),
            ("implicit_midpoint", 0.1 / kappa, "resolved"),
            ("verlet", 0.1 / kappa, "resolved"),
        )
        for method, dt, regime in configurations:
            q, p, result = integrate(
                method, q0, p0, system.omegas**2, dt, T_MAX
            )
            error = np.concatenate([q, p]) - exact_vector
            result.update(
                kappa=kappa,
                regime=regime,
                h_omega_max=dt * float(np.max(system.omegas)),
                final_state_error=float(np.linalg.norm(error)),
                scaled_final_state_error=float(np.linalg.norm(error / scales)),
            )
            records.append(result)
            print(
                f"kappa={kappa:g} {method} {regime}: "
                f"dE={result['max_energy_drift']:.3e}, "
                f"state={result['final_state_error']:.3e}",
                flush=True,
            )

    fixed_midpoint = [
        record
        for record in records
        if record["method"] == "implicit_midpoint"
        and record["regime"] == "fixed_cost"
    ]
    acceptance = {
        "all_final_times_aligned": all(
            record["final_time_error"] < 1e-10 for record in records
        ),
        "midpoint_energy_roundoff": all(
            record["max_energy_drift"] < 1e-10
            for record in records
            if record["method"] == "implicit_midpoint"
        ),
        "fixed_cost_state_error_detected": max(
            record["final_state_error"] for record in fixed_midpoint
        )
        > 1e-2,
    }
    acceptance["passed"] = all(acceptance.values())
    artifact = {
        "schema_version": "1.0",
        "experiment": "expAC4_modal_correctness",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": None,
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "system": {
            "name": "ModalOscillator",
            "n_modes": N_MODES,
            "energy": E0,
            "kappas": list(KAPPAS),
            "physical_horizon": T_MAX,
            "initial_condition_seed": 1,
            "orthogonal_mixing": False,
        },
        "dtype": "float64",
        "nonlinear_solver": {
            "type": "closed_form_Cayley",
            "residual": "not_applicable",
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "records": records,
        "acceptance": acceptance,
        "interpretation": {
            "supported":
                "quadratic implicit midpoint energy remains at roundoff",
            "not_supported":
                "small energy drift implies small state or phase error",
            "learned_monitor_tested": False,
            "reason":
                "historical modal learned-monitor checkpoints were not saved",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    if not acceptance["passed"]:
        raise SystemExit("modal correctness gate failed")


if __name__ == "__main__":
    main()
