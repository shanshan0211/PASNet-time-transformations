"""Fast correctness gate for Kepler controls used by the PASNet paper.

This is intentionally a one-period, float64 test.  It must pass before any
100-period performance experiment is trusted.  The script writes a compact
JSON artifact with state errors, energy errors, nonlinear-solver residuals,
and observed convergence orders.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from kepler_integrators import implicit_midpoint_step, kepler_energy  # noqa: E402
from systems import Kepler2D  # noqa: E402


SYSTEM = Kepler2D(0.9)
Q0, P0 = SYSTEM.initial_condition()
E0 = kepler_energy(Q0, P0)
PERIOD = 2.0 * math.pi
OUT = os.path.join(ROOT, "results", "expAC0")


def state_error(q: np.ndarray, p: np.ndarray) -> float:
    return float(np.linalg.norm(np.concatenate([q - Q0, p - P0])))


def run_direct(dt: float) -> dict:
    q, p = Q0.copy(), P0.copy()
    t = 0.0
    max_drift = 0.0
    max_residual = 0.0
    max_iterations = 0
    steps = 0
    while t < PERIOD:
        h = min(dt, PERIOD - t)
        q, p, info = implicit_midpoint_step(
            q, p, h, max_iter=100, tol=1e-13, require_convergence=True
        )
        t += h
        steps += 1
        max_drift = max(max_drift, abs(kepler_energy(q, p) - E0))
        max_residual = max(max_residual, info.residual)
        max_iterations = max(max_iterations, info.iterations)
    return {
        "dt": dt,
        "steps": steps,
        "state_error": state_error(q, p),
        "max_energy_drift": max_drift,
        "max_fp_residual": max_residual,
        "max_fp_iterations": max_iterations,
        "final_radius": float(np.linalg.norm(q)),
    }


def convergence_orders(records: list[dict], key: str) -> list[float]:
    orders = []
    for coarse, fine in zip(records, records[1:]):
        ratio_h = coarse["dt"] / fine["dt"]
        ratio_e = coarse[key] / fine[key]
        orders.append(float(math.log(ratio_e) / math.log(ratio_h)))
    return orders


def main() -> None:
    # Fixed-point midpoint is only contractive once h resolves the periapsis
    # timescale. These values exercise the converged asymptotic regime.
    records = [run_direct(dt) for dt in (0.01, 0.005, 0.0025)]
    orders = convergence_orders(records, "state_error")
    passed = (
        all(record["max_fp_residual"] < 1e-12 for record in records)
        and all(order > 1.7 for order in orders)
        and records[-1]["final_radius"] < 2.0
    )
    artifact = {
        "experiment": "expAC0_correctness_smoke",
        "system": "Kepler2D(e=0.9, a=1)",
        "dtype": "float64",
        "physical_horizon": PERIOD,
        "initial_energy": E0,
        "direct_implicit_midpoint": records,
        "observed_state_orders": orders,
        "acceptance": {
            "residual_lt": 1e-12,
            "observed_order_gt": 1.7,
            "final_radius_lt": 2.0,
            "passed": passed,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(OUT, exist_ok=True)
    output = os.path.join(OUT, "correctness_smoke.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise SystemExit("correctness gate failed")


if __name__ == "__main__":
    main()
