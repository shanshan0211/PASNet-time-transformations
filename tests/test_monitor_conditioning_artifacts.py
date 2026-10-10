"""Unit and artifact tests for AC11 solver-conditioning audit."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import torch

from tests.source_integrity import assert_frozen_or_registered_remediation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from implicit_midpoint_solvers import (  # noqa: E402
    endpoint_residual,
    fixed_point_endpoint,
)
from monitor_conditioning import (  # noqa: E402
    conditioning_diagnostics,
    kepler_shell_scaling,
    scaled_step_size,
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


class CorrectedSolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_returned_residual_is_explicit_and_below_tolerance(self):
        state = torch.tensor([[0.7, -0.2]], requires_grad=True)

        def field(value):
            return torch.tanh(value)

        endpoint, diagnostics = fixed_point_endpoint(
            state,
            0.3,
            field,
            tolerance=1e-11,
            maximum_iterations=100,
        )
        explicit = float(
            torch.max(
                torch.abs(endpoint_residual(state, endpoint, 0.3, field))
            ).detach()
        )
        self.assertTrue(diagnostics.converged)
        self.assertLess(explicit, 1e-11)
        self.assertAlmostEqual(
            diagnostics.returned_endpoint_residual_inf, explicit, places=15
        )
        self.assertIsNotNone(diagnostics.stopping_update_inf)
        endpoint.sum().backward()
        self.assertIsNotNone(state.grad)
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_shell_scaling(self):
        scaling, inverse = kepler_shell_scaling(4.0)
        self.assertTrue(
            torch.allclose(
                torch.diag(scaling),
                torch.tensor([4.0, 4.0, 0.5, 0.5]),
            )
        )
        self.assertTrue(torch.allclose(inverse @ scaling, torch.eye(4)))
        self.assertEqual(scaled_step_size(0.08, 4.0), 0.01)

    def test_quadratic_conditioning_formula(self):
        midpoint = torch.tensor([[0.3, -0.4, 0.2, 0.1]])

        def monitor(value):
            return 2.0 + 0.0 * value[..., 0]

        def hamiltonian(value):
            return 0.5 * torch.sum(value * value, dim=-1)

        diagnostics = conditioning_diagnostics(
            midpoint,
            monitor,
            hamiltonian,
            p0=-0.15,
            step_size=0.1,
            semi_major_axis=1.0,
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(diagnostics["D2zzK"]),
                2.0 * torch.eye(4),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertAlmostEqual(
            diagnostics["scaled_J_D2zzK_norm_2"], 2.0, places=12
        )
        self.assertAlmostEqual(diagnostics["contraction_q"], 0.1, places=12)
        expected_singular = math.sqrt(1.0 + 0.1**2)
        for value in diagnostics[
            "scaled_endpoint_jacobian_singular_values"
        ]:
            self.assertAlmostEqual(value, expected_singular, places=12)
        self.assertAlmostEqual(
            diagnostics["scaled_endpoint_jacobian_condition_2"], 1.0, places=12
        )


class AC11ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = (
            ROOT / "results" / "expAC11" / "preregistration.json"
        )
        cls.raw_path = ROOT / "results" / "expAC11" / "raw_results.json"
        cls.analysis_path = ROOT / "results" / "expAC11" / "analysis.json"

    def test_protocol_integrity_if_registered(self):
        if not self.protocol_path.exists():
            self.skipTest("AC11 preregistration has not been generated")
        protocol = load(self.protocol_path)
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(
            protocol["evaluation"]["total_registered_rollouts"], 1056
        )
        self.assertEqual(protocol["evaluation"]["AC8_arm_rollouts"], 264)
        self.assertEqual(protocol["evaluation"]["AC9_arm_rollouts"], 792)
        self.assertEqual(
            protocol["sampling"]["accepted_physical_step_stride"], 10
        )
        checkpoint_sources = [
            source
            for source in protocol["frozen_sources"]
            if source["role"] == "parent_checkpoint"
        ]
        self.assertEqual(len(checkpoint_sources), 20)
        for source in protocol["frozen_sources"]:
            assert_frozen_or_registered_remediation(self, source)

    def test_replay_artifacts_if_generated(self):
        if not self.raw_path.exists():
            self.skipTest("AC11 replay has not been run")
        protocol = load(self.protocol_path)
        raw = load(self.raw_path)
        self.assertEqual(
            raw["protocol_sha256"], protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(len(raw["records"]), 1056)
        self.assertEqual(
            len({record["rollout_key"] for record in raw["records"]}), 1056
        )
        self.assertEqual(
            sum(record["arm"] == "AC8" for record in raw["records"]), 264
        )
        self.assertEqual(
            sum(record["arm"] == "AC9" for record in raw["records"]), 792
        )
        for record in raw["records"]:
            self.assertNotEqual(record["replay_completion_status"], "running")
            for sample in record["samples"]:
                if sample["sample_kind"] == "accepted_stride":
                    self.assertEqual(sample["accepted_physical_step"] % 10, 0)
                solver = sample.get("solver", {})
                if solver.get("returned_endpoint_residual_inf") is not None:
                    self.assertGreaterEqual(
                        solver["returned_endpoint_residual_inf"], 0.0
                    )
        self.assertTrue(
            self.analysis_path.exists(),
            "complete raw replay requires analyze_expAC11.py output",
        )
        analysis = load(self.analysis_path)
        self.assertTrue(
            analysis["completion_label_join"]["complete_and_unique"]
        )
        self.assertTrue((ROOT / "fig_conditioning_audit.pdf").is_file())
        self.assertTrue((ROOT / "fig_conditioning_audit.png").is_file())


if __name__ == "__main__":
    unittest.main()
