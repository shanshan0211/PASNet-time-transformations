"""Unit and sealed-artifact tests for the AC16 reference correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def import_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AC16 = import_file(
    "expAC16", "experiments/expAC16_reference_convergence.py"
)
ANALYSIS = import_file("analyze_expAC16", "experiments/analyze_expAC16.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReferenceConvergenceAC16Tests(unittest.TestCase):
    def test_short_hierarchical_references_have_nonzero_uncertainty(self):
        system, state = AC16.initial_state("duffing", 1.0, 0)
        trajectories = {
            dt: AC16.velocity_verlet_trajectory(
                "duffing", system, state, 0.2, dt
            )
            for dt in [0.004, 0.002, 0.001]
        }
        coarse = AC16.interpolated_state_discrepancy(
            *trajectories[0.004], *trajectories[0.002]
        )
        fine = AC16.interpolated_state_discrepancy(
            *trajectories[0.002], *trajectories[0.001]
        )
        order = AC16.observed_order(coarse["maximum"], fine["maximum"])
        uncertainty = AC16.richardson_uncertainty(
            fine["maximum"], order, fine_side=True
        )
        self.assertGreater(coarse["maximum"], 0.0)
        self.assertGreater(fine["maximum"], 0.0)
        self.assertGreater(uncertainty["absolute"], 0.0)
        self.assertTrue(math.isfinite(uncertainty["absolute"]))
        self.assertGreater(order, 1.5)
        self.assertLess(order, 2.5)

    def test_all_ac13_scenarios_are_registered(self):
        protocol = {
            "evaluation": {
                "modal": {
                    "energy_shells": [0.75, 1.0, 1.25],
                    "initial_condition_seeds": [0, 1],
                    "kappas": [10.0, 20.0],
                    "momentum_directions": ["forward", "reversed"],
                },
                "duffing": {
                    "energy_shells": [0.75, 1.0, 1.25],
                    "initial_condition_seeds": [0, 1],
                    "momentum_directions": ["forward", "reversed"],
                },
            }
        }
        self.assertEqual(len(AC16.scenarios(protocol, "modal")), 24)
        self.assertEqual(len(AC16.scenarios(protocol, "duffing")), 12)

    def test_triangle_intervals_never_treat_reference_as_exact(self):
        scenario = {
            "system": "duffing",
            "energy_shell": 1.0,
            "initial_condition_seed": 0,
            "kappa": None,
            "momentum_direction": "forward",
        }
        reference = {
            **scenario,
            "state_reference_uncertainty": {
                "ac13_dt": {
                    "maximum": {"absolute": 0.02},
                    "final": {"absolute": 0.01},
                }
            },
        }
        ac13 = {
            **scenario,
            "method": "prescribed_monitor",
            "architecture": None,
            "checkpoint_seed": None,
            "completion_status": "complete",
            "max_state_error": 0.1,
            "final_state_error": 0.05,
        }
        corrected = ANALYSIS.corrected_records(
            [ac13], {AC16.scenario_key(reference): reference}
        )
        maximum = corrected[0]["metrics"]["max_state_error"]
        self.assertAlmostEqual(maximum["lower"], 0.08)
        self.assertAlmostEqual(maximum["upper"], 0.12)
        self.assertGreater(maximum["reference_uncertainty"], 0.0)

    def test_sealed_protocol_when_present(self):
        path = ROOT / "results" / "expAC16" / "preregistration.json"
        if not path.exists():
            self.skipTest("AC16 protocol has not yet been sealed")
        with path.open(encoding="utf-8") as handle:
            protocol = json.load(handle)
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(protocol["reference"]["dt_values"], [0.002, 0.001, 0.0005])
        self.assertEqual(protocol["evaluation"]["total_scenarios"], 36)
        self.assertTrue(protocol["reference"]["zero_reference_points_forbidden"])
        for source in protocol["frozen_sources"]:
            self.assertEqual(
                sha256_file(ROOT / source["path"]),
                source["sha256"],
                source["path"],
            )
        for artifact in protocol["frozen_ac13_artifacts"]:
            self.assertEqual(
                sha256_file(ROOT / artifact["path"]),
                artifact["sha256"],
                artifact["path"],
            )

    def test_runtime_artifacts_when_present(self):
        summary_path = ROOT / "results" / "expAC16" / "summary.json"
        if not summary_path.exists():
            self.skipTest("full AC16 experiment intentionally not run")
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with (summary_path.parent / "raw_results.json").open(
            encoding="utf-8"
        ) as handle:
            raw = json.load(handle)
        self.assertEqual(len(raw["records"]), 36)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        for record in raw["records"]:
            for field in ["maximum", "final"]:
                self.assertGreater(
                    record["state_reference_uncertainty"]["finest_dt"][field][
                        "absolute"
                    ],
                    0.0,
                )


if __name__ == "__main__":
    unittest.main()
