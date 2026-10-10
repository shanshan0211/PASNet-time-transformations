"""Audit physical symmetries of the learned Kepler time monitor.

For an isotropic Kepler problem, a transferable monitor should normally be
invariant when q and p are rotated together and should be even under momentum
reversal. PASNet does not enforce these properties, so this script measures
their violation at fixed physical states without making a performance claim.
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


def rotated_states(z, n_angles=360):
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    states = []
    q, p = z[:2], z[2:]
    for angle in angles:
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        states.append(np.concatenate([rotation @ q, rotation @ p]))
    return np.asarray(states)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    return {
        "mean": mean,
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "relative_range": float((np.max(values) - np.min(values)) / mean),
        "coefficient_of_variation": float(np.std(values) / mean),
    }


def main():
    torch.set_default_dtype(torch.float64)
    model, checkpoint = load_model()
    system = Kepler2D(0.9)
    q0, p0 = system.initial_condition()
    reference_states = {
        "apoapsis": np.concatenate([q0, p0]),
        "periapsis": np.array(
            [-0.1, 0.0, 0.0, -math.sqrt((1.0 + 0.9) / (1.0 - 0.9))]
        ),
        "generic": np.array([0.6, 0.4, -0.5, 0.8]),
    }

    rotations = {}
    reversals = {}
    with torch.no_grad():
        for name, state in reference_states.items():
            rotated = torch.tensor(rotated_states(state), dtype=torch.float64)
            rotations[name] = summarize(model.g(rotated).numpy())

            forward = torch.tensor(state[None, :], dtype=torch.float64)
            reversed_state = state.copy()
            reversed_state[2:] *= -1.0
            backward = torch.tensor(reversed_state[None, :], dtype=torch.float64)
            g_forward = float(model.g(forward))
            g_backward = float(model.g(backward))
            reversals[name] = {
                "g_forward": g_forward,
                "g_reversed": g_backward,
                "relative_difference": abs(g_forward - g_backward)
                / (0.5 * (g_forward + g_backward)),
            }

    max_rotation_range = max(
        result["relative_range"] for result in rotations.values()
    )
    max_reversal_difference = max(
        result["relative_difference"] for result in reversals.values()
    )
    artifact = {
        "schema_version": "1.0",
        "experiment": "expAC3_monitor_symmetry",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "dtype": "float64",
        "n_rotation_angles": 360,
        "rotation_audit": rotations,
        "momentum_reversal_audit": reversals,
        "summary": {
            "max_rotation_relative_range": max_rotation_range,
            "max_momentum_reversal_relative_difference":
                max_reversal_difference,
            "rotation_invariance_enforced_by_architecture": False,
            "momentum_evenness_enforced_by_architecture": False,
        },
    }
    output_dir = os.path.join(RESULTS, "expAC3")
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, "monitor_symmetry.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
