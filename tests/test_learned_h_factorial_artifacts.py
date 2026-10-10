"""Unit, protocol, and artifact tests for the sealed AC12 factorial."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

from tests.source_integrity import assert_frozen_or_registered_remediation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from implicit_midpoint_solvers import (  # noqa: E402
    endpoint_residual,
    fixed_point_endpoint,
    monotone_residual_newton_endpoint,
)


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def load_runner():
    path = (
        ROOT
        / "experiments"
        / "expAC12_learned_h_monitor_solver_factorial.py"
    )
    spec = importlib.util.spec_from_file_location("expAC12_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AC12UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.runner = load_runner()

    def test_monitor_features_exactly_match_ac9_definition(self):
        state = torch.tensor([[2.0, 0.0, 3.0, 4.0]])
        features = self.runner.invariant_features(state)
        expected = torch.tensor([[2.0, 0.5, 25.0, 36.0, 64.0]])
        self.assertTrue(torch.equal(features, expected))
        self.assertEqual(self.runner.Monitor("raw").network[0].in_features, 4)
        self.assertEqual(
            self.runner.Monitor("invariant").network[0].in_features, 5
        )

    def test_frozen_learned_h_is_float64_and_h_net_only(self):
        checkpoint = ROOT / "results" / "expE2" / "pasnet_weights.pt"
        model = self.runner.HamiltonianMode("learned", checkpoint)
        self.assertIsNotNone(model.H_net)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.dtype == torch.float64 for p in model.parameters()))
        state = torch.tensor([[1.3, 0.2, -0.1, 0.8]], requires_grad=True)
        value = model(state)
        self.assertEqual(value.dtype, torch.float64)
        value.sum().backward()
        self.assertIsNotNone(state.grad)
        self.assertTrue(torch.isfinite(state.grad).all())
        with self.assertRaises(TypeError):
            model(state.detach().float())

    def test_corrected_solvers_accept_true_returned_residual(self):
        state = torch.tensor([[0.7, -0.2]])

        def field(value):
            return torch.tanh(value)

        for solver in [
            fixed_point_endpoint,
            monotone_residual_newton_endpoint,
        ]:
            endpoint, diagnostics = solver(
                state,
                0.3,
                field,
                tolerance=1e-12,
            )
            explicit = float(
                torch.max(
                    torch.abs(endpoint_residual(state, endpoint, 0.3, field))
                ).detach()
            )
            self.assertTrue(diagnostics.converged)
            self.assertLess(explicit, 1e-12)
            self.assertAlmostEqual(
                explicit,
                diagnostics.returned_endpoint_residual_inf,
                places=15,
            )

    def test_scenario_grid_has_twelve_unique_symmetry_cases(self):
        protocol = {
            "evaluation": {
                "eccentricities": [0.7, 0.9, 0.99],
                "orientations_radians": [0.0, 1.5707963267948966],
                "momentum_directions": ["forward", "reversed"],
            }
        }
        scenarios = self.runner.scenarios_from_protocol(protocol)
        self.assertEqual(len(scenarios), 12)
        self.assertEqual(
            len(
                {
                    (
                        item["eccentricity"],
                        item["orientation_radians"],
                        item["momentum_direction"],
                    )
                    for item in scenarios
                }
            ),
            12,
        )


class AC12ProtocolAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = ROOT / "results" / "expAC12"
        cls.protocol_path = cls.out / "preregistration.json"
        cls.training_path = cls.out / "training.json"
        cls.raw_path = cls.out / "raw_results.json"
        cls.summary_path = cls.out / "summary.json"
        cls.analysis_path = cls.out / "analysis.json"

    def test_protocol_integrity_if_registered(self):
        if not self.protocol_path.exists():
            self.skipTest("AC12 preregistration has not been generated")
        protocol = load(self.protocol_path)
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(protocol["training"]["checkpoint_count"], 20)
        self.assertEqual(
            [entry["checkpoint_seed"] for entry in protocol["training"]["seed_manifest"]],
            [8101, 8102, 8103, 8104, 8105],
        )
        self.assertEqual(
            protocol["evaluation"]["total_registered_rollouts"], 528
        )
        self.assertEqual(protocol["evaluation"]["learned_rollouts"], 480)
        self.assertEqual(
            protocol["evaluation"]["manual_control_rollouts"], 48
        )
        self.assertEqual(
            protocol["evaluation"]["acceptance_tolerance"], 1e-12
        )
        self.assertIn(
            "monotone_residual_newton", protocol["evaluation"]["solvers"]
        )
        learned_sources = [
            source
            for source in protocol["frozen_sources"]
            if source["role"] == "single_frozen_learned_h_checkpoint"
        ]
        self.assertEqual(len(learned_sources), 1)
        self.assertEqual(
            learned_sources[0]["sha256"],
            protocol["hamiltonians"]["learned"]["checkpoint_sha256"],
        )
        for source in protocol["frozen_sources"]:
            assert_frozen_or_registered_remediation(self, source)

    def test_result_artifacts_if_generated(self):
        if not self.raw_path.exists():
            self.skipTest("AC12 long training/evaluation has not been run")
        for path in [
            self.protocol_path,
            self.training_path,
            self.raw_path,
            self.summary_path,
            self.analysis_path,
        ]:
            self.assertTrue(path.is_file(), str(path))
        protocol = load(self.protocol_path)
        training = load(self.training_path)
        raw = load(self.raw_path)
        summary = load(self.summary_path)
        self.assertEqual(len(training["records"]), 20)
        self.assertEqual(len(raw["records"]), 528)
        self.assertEqual(
            len({record["rollout_key"] for record in raw["records"]}), 528
        )
        self.assertEqual(
            Counter(record["architecture"] for record in raw["records"]),
            {"raw": 240, "invariant": 240, "manual_r": 48},
        )
        self.assertEqual(
            Counter(record["h_mode"] for record in raw["records"]),
            {"exact": 264, "learned": 264},
        )
        self.assertEqual(
            Counter(record["solver"] for record in raw["records"]),
            {"fixed_point": 264, "monotone_residual_newton": 264},
        )
        for record in training["records"]:
            self.assertNotEqual(record["status"], "training_failure")
            self.assertEqual(
                sha256_file(ROOT / record["checkpoint_path"]),
                record["checkpoint_sha256"],
            )
        for record in raw["records"]:
            self.assertNotEqual(record["completion_status"], "running")
            if record["completion_status"] == "complete":
                self.assertLess(
                    record["maximum_returned_endpoint_residual_inf"],
                    protocol["evaluation"]["acceptance_tolerance"],
                )
        self.assertEqual(summary["counts"]["attempted"], 528)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        self.assertTrue(
            (ROOT / "fig_learned_h_factorial.pdf").is_file()
        )
        self.assertTrue(
            (ROOT / "fig_learned_h_factorial.png").is_file()
        )


if __name__ == "__main__":
    unittest.main()
