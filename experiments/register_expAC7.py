"""Register the post-hoc AC7 refinement of AC6 failed scenarios."""

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
AC6_RAW = os.path.join(RESULTS, "expAC6", "raw_results.json")
AC6_SUMMARY = os.path.join(RESULTS, "expAC6", "summary.json")
OUTPUT = os.path.join(RESULTS, "expAC7", "preregistration.json")


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
    with open(AC6_RAW, encoding="utf-8") as handle:
        ac6 = json.load(handle)
    failed_scenarios = sorted(
        [
            {
                "eccentricity": record["eccentricity"],
                "orientation_radians": record["orientation_radians"],
                "momentum_direction": record["momentum_direction"],
                "ac6_status": record["completion_status"],
            }
            for record in ac6["records"]
            if record["method"] == "learned_g"
            and record["completion_status"] != "complete"
        ],
        key=lambda item: (
            item["eccentricity"],
            item["orientation_radians"],
            item["momentum_direction"],
        ),
    )
    if len(failed_scenarios) != 15:
        raise RuntimeError(
            f"expected 15 AC6 learned-g failures, found {len(failed_scenarios)}"
        )

    checkpoint = os.path.join(RESULTS, "expE2", "pasnet_weights.pt")
    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC7_failed_scenario_refinement",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_but_sealed_before_any_expAC7_rollout",
        "relationship_to_ac6":
            "AC7 is a declared post-hoc diagnostic restricted to all and "
            "only AC6 learned-g failures; it does not alter AC6 estimands",
        "hypotheses": {
            "solver_resolution":
                "if failure is only coarse fictitious-time resolution, "
                "halving dtau should restore convergence",
            "monitor_conditioning":
                "if learned-monitor derivatives create poor conditioning, "
                "learned g may require more refinement than radial manual g",
            "high_eccentricity_accuracy":
                "manual-r energy error at e=0.99 should decrease under "
                "refinement even though AC6 dtau=0.04 was inaccurate",
        },
        "frozen_checkpoint": {
            "path": "results/expE2/pasnet_weights.pt",
            "sha256": sha256_file(checkpoint),
        },
        "frozen_sources": [
            {
                "path": "results/expAC6/raw_results.json",
                "sha256": sha256_file(AC6_RAW),
            },
            {
                "path": "results/expAC6/summary.json",
                "sha256": sha256_file(AC6_SUMMARY),
            },
        ],
        "failed_scenarios": failed_scenarios,
        "design": {
            "methods": ["manual_r", "learned_g"],
            "dtau_values": [0.02, 0.01],
            "physical_horizon": 62.83185307179586,
            "periods": 10,
            "dtype": "float64",
            "torch_threads": 1,
            "fixed_point_tolerance": 1e-12,
            "fixed_point_max_iterations": 100,
            "maximum_steps_per_rollout": 500000,
            "matched_final_physical_time": True,
            "total_registered_rollouts": 60,
        },
        "outcomes": {
            "primary": [
                "completion rate by method and dtau",
                "energy and closure refinement on completed pairs",
                "learned/manual step and wall-time ratios",
            ],
            "censoring":
                "retain every failure; finite-pair metrics report denominator",
        },
        "acceptance": {
            "all_60_attempted_and_retained": True,
            "completed_final_time_error_below": 1e-10,
            "completed_solver_residual_below": 1e-11,
            "scientific_success_not_required": True,
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
