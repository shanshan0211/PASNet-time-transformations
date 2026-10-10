"""Create the immutable Exp AC6 monitor-generalisation audit protocol."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
OUTPUT = os.path.join(RESULTS, "expAC6", "preregistration.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")

    checkpoint = os.path.join(RESULTS, "expE2", "pasnet_weights.pt")
    ac2 = os.path.join(RESULTS, "expAC2", "corrected_100period.json")
    ac3 = os.path.join(RESULTS, "expAC3", "monitor_symmetry.json")
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC6_monitor_orbit_generalisation",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_expAC6_rollout",
        "hypothesis": {
            "primary":
                "the frozen learned monitor has orbit-conditioned error and "
                "cost that vary under rotations and momentum reversal",
            "positive_control":
                "the radial manual Sundman monitor is invariant to rotation "
                "and momentum reversal up to numerical tolerance",
            "baseline":
                "expAC3 measured pointwise symmetry violations but did not "
                "measure their rollout-level consequences",
        },
        "frozen_checkpoint": {
            "path": "results/expE2/pasnet_weights.pt",
            "sha256": sha256_file(checkpoint),
        },
        "frozen_sources": [
            {
                "path": "results/expAC2/corrected_100period.json",
                "sha256": sha256_file(ac2),
            },
            {
                "path": "results/expAC3/monitor_symmetry.json",
                "sha256": sha256_file(ac3),
            },
        ],
        "design": {
            "system": "Kepler2D(a=1,mu=1)",
            "eccentricities": [0.5, 0.7, 0.9, 0.95, 0.99],
            "reference_eccentricity": 0.9,
            "unseen_eccentricities": [0.5, 0.7, 0.95, 0.99],
            "orientation_radians": [
                0.0,
                0.7853981633974483,
                1.5707963267948966,
                2.356194490192345,
            ],
            "momentum_directions": ["forward", "reversed"],
            "methods": ["manual_r", "learned_g"],
            "true_hamiltonian": True,
            "periods": 10,
            "physical_horizon": 62.83185307179586,
            "dtau": 0.04,
            "dtype": "float64",
            "torch_threads": 1,
            "fixed_point_tolerance": 1e-12,
            "fixed_point_max_iterations": 100,
            "maximum_steps_per_rollout": 250000,
            "matched_final_physical_time": True,
            "total_registered_rollouts": 80,
        },
        "frozen_outcomes": {
            "per_rollout": [
                "completion_status",
                "max_energy_drift",
                "final_state_closure_error",
                "max_angular_momentum_error",
                "steps",
                "gradient_evaluations",
                "wall_seconds",
                "max_fixed_point_residual",
                "monitor_range",
                "final_time_error",
            ],
            "primary_estimands": [
                "within-eccentricity rotation max/min error ratio",
                "forward/reversed paired error ratio",
                "learned-g/manual-r paired error and cost ratios",
                "failure counts with no exclusion",
            ],
        },
        "acceptance": {
            "artifact_gate": [
                "all 80 registered rollouts attempted and retained",
                "no failed rollout silently excluded from an estimand",
                "every completed rollout final-time error below 1e-10",
                "every completed rollout fixed-point residual below 1e-11",
                "checkpoint and source hashes unchanged",
            ],
            "scientific_success_not_required": True,
            "claim_rule":
                "report a failure/applicability result regardless of whether "
                "the learned monitor is better or worse",
        },
        "censoring": {
            "nonconverged_solver": "retain as solver_failure",
            "maximum_steps_reached": "retain as step_cap",
            "nonfinite_state_or_metric": "retain as nonfinite",
            "ratio_rule":
                "ratios use only paired finite completed rollouts and always "
                "report their denominator plus all failure counts",
        },
        "hardware_registration": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "integrity": {"protocol_sha256": None},
    }
    protocol["integrity"]["protocol_sha256"] = canonical_hash(protocol)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=False)
    with open(OUTPUT, "x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
        handle.write("\n")
    print(protocol["integrity"]["protocol_sha256"])


if __name__ == "__main__":
    main()
