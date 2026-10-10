"""Low-cost corrected known-H PASNet rollout gate.

This gate verifies that the learned time transformation is coupled to the
actual Kepler vector field, that the trajectory moves through periapsis, and
that energy error decreases under transformed-step refinement. It does not
replace the 100-period matched-cost experiment.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.pasnet import PASNet  # noqa: E402
from systems import Kepler2D  # noqa: E402


def h_features(z):
    q = z[:, :2]
    r = torch.linalg.vector_norm(q, dim=1, keepdim=True)
    return torch.cat([r, 1.0 / r], dim=1)


def load_model():
    model = PASNet(
        dim=2,
        hidden_H=128,
        n_hidden_H=3,
        hidden_g=64,
        n_hidden_g=2,
        g_max=2.0,
        h_features=h_features,
    )
    checkpoint = os.path.join(RESULTS, "expE2", "pasnet_weights.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.H_net.load_state_dict(state["H_net"])
    model.g_net.load_state_dict(state["g_net"])
    return model.double().eval(), checkpoint


def run(model, system, q0, p0, energy0, dtau, t_max):
    z = torch.tensor(
        np.concatenate([q0, p0])[None, :], dtype=torch.float64
    )
    p0_ext = torch.tensor([-energy0], dtype=torch.float64)
    max_drift = 0.0
    min_radius = float(np.linalg.norm(q0))
    max_radius = min_radius
    max_residual = 0.0
    max_iterations = 0
    t = 0.0
    steps = 0

    def vector_field(zz):
        with torch.enable_grad():
            zr = zz.detach().requires_grad_(True)
            k = model.g(zr) * (system.H_torch(zr) + p0_ext)
            grad_k = torch.autograd.grad(k.sum(), zr)[0]
        return torch.cat([grad_k[:, 2:], -grad_k[:, :2]], dim=1).detach()

    while t < t_max:
        z_next = z + dtau * vector_field(z)
        residual = math.inf
        for iteration in range(1, 101):
            proposal = z + dtau * vector_field(0.5 * (z + z_next))
            residual = float(torch.max(torch.abs(proposal - z_next)))
            z_next = proposal
            if residual < 1e-12:
                break
        else:
            raise RuntimeError(
                f"known-H midpoint failed: dtau={dtau}, "
                f"residual={residual:.3e}"
            )

        midpoint = 0.5 * (z + z_next)
        t += dtau * float(model.g(midpoint).detach())
        z = z_next
        steps += 1
        max_residual = max(max_residual, residual)
        max_iterations = max(max_iterations, iteration)
        energy = float(system.H_torch(z))
        radius = float(torch.linalg.vector_norm(z[:, :2]))
        max_drift = max(max_drift, abs(energy - energy0))
        min_radius = min(min_radius, radius)
        max_radius = max(max_radius, radius)

    return {
        "dtau": dtau,
        "physical_time": t,
        "steps": steps,
        "max_energy_drift": max_drift,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "max_fp_residual": max_residual,
        "max_fp_iterations": max_iterations,
    }


def main():
    torch.set_default_dtype(torch.float64)
    system = Kepler2D(0.9)
    q0, p0 = system.initial_condition()
    energy0 = float(system.H(q0, p0))
    model, checkpoint = load_model()
    records = [
        run(model, system, q0, p0, energy0, dtau, 2.0 * math.pi)
        for dtau in (0.04, 0.02, 0.01)
    ]
    drifts = [record["max_energy_drift"] for record in records]
    passed = (
        all(np.isfinite(drifts))
        and drifts[1] < drifts[0]
        and drifts[2] < drifts[1]
        and all(record["max_fp_residual"] < 1e-11 for record in records)
        and records[-1]["min_radius"] < 0.2
        and records[-1]["max_radius"] > 1.8
    )
    artifact = {
        "experiment": "expAC1_knownH_corrected",
        "system": "Kepler2D(e=0.9, a=1)",
        "dtype": "float64",
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "records": records,
        "acceptance": {
            "strict_drift_refinement": True,
            "max_fp_residual_lt": 1e-11,
            "periapsis_lt": 0.2,
            "apoapsis_gt": 1.8,
            "passed": passed,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = os.path.join(RESULTS, "expAC1")
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "knownH_corrected.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(artifact, handle, indent=2)
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise SystemExit("corrected known-H gate failed")


if __name__ == "__main__":
    main()
