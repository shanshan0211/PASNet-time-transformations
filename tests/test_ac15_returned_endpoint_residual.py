"""Unit and integrity tests for the AC15 correction experiment."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from expAC15_returned_endpoint_residual_correction import (  # noqa: E402
    PARENT_COUNTS,
    TOTAL_REPLAYS,
    atomic_json,
    replay_manifest,
)
from implicit_midpoint_solvers import (  # noqa: E402
    IterationLimitError,
    endpoint_residual,
    fixed_point_endpoint,
)
from register_expAC15 import canonical_hash  # noqa: E402


class ReturnedEndpointResidualSolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_success_is_verified_at_exact_returned_endpoint(self):
        state = torch.tensor([[1.0, -0.5]])

        def field(value):
            return 0.8 * value + torch.tensor([[0.2, -0.1]])

        endpoint, diagnostics = fixed_point_endpoint(
            state,
            0.25,
            field,
            tolerance=1e-11,
            maximum_iterations=100,
        )
        actual = float(torch.max(torch.abs(
            endpoint_residual(state, endpoint, 0.25, field)
        )))
        self.assertTrue(diagnostics.converged)
        self.assertLess(actual, 1e-11)
        self.assertAlmostEqual(
            diagnostics.returned_endpoint_residual_inf,
            actual,
            delta=1e-16,
        )
        self.assertGreaterEqual(
            diagnostics.vector_field_evaluations,
            diagnostics.iterations + 2,
        )
        self.assertLessEqual(len(diagnostics.residual_history_inf), 2)

    def test_iteration_limit_retains_endpoint_and_true_residual(self):
        state = torch.tensor([[1.0]])

        def field(value):
            return 4.0 * value

        with self.assertRaises(IterationLimitError) as context:
            fixed_point_endpoint(
                state,
                1.0,
                field,
                tolerance=1e-14,
                maximum_iterations=3,
            )
        error = context.exception
        actual = float(torch.max(torch.abs(
            endpoint_residual(state, error.endpoint, 1.0, field)
        )))
        self.assertEqual(error.diagnostics.failure_reason, "iteration_limit")
        self.assertFalse(error.diagnostics.converged)
        self.assertAlmostEqual(
            error.diagnostics.returned_endpoint_residual_inf,
            actual,
            delta=1e-14,
        )
        self.assertGreaterEqual(
            len(error.diagnostics.residual_history_inf), 1
        )


class AC15ProtocolAndManifestTests(unittest.TestCase):
    def test_manifest_is_minimal_complete_and_parent_linked(self):
        manifest = replay_manifest()
        self.assertEqual(len(manifest), TOTAL_REPLAYS)
        counts = {
            parent: sum(item["parent"] == parent for item in manifest)
            for parent in PARENT_COUNTS
        }
        self.assertEqual(counts, PARENT_COUNTS)
        self.assertEqual(len({
            item["replay_id"] for item in manifest
        }), TOTAL_REPLAYS)
        ac8 = [item for item in manifest if item["parent"] == "AC8"]
        self.assertTrue(ac8)
        self.assertTrue(all(
            item["parent_record"]["solver"] == "fixed_point"
            for item in ac8
        ))

    def test_atomic_json_leaves_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            atomic_json(str(path), {"records": [{"id": 1}]})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"records": [{"id": 1}]},
            )
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])

    def test_sealed_protocol_hash_and_registered_count(self):
        path = ROOT / "results" / "expAC15" / "preregistration.json"
        if not path.exists():
            self.skipTest("run register_expAC15.py before integrity check")
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            canonical_hash(protocol),
            protocol["integrity"]["protocol_sha256"],
        )
        decision = protocol["minimum_rerun_decision"]
        self.assertEqual(
            decision["total_registered_replays"], TOTAL_REPLAYS
        )
        self.assertIn("AC8_damped_newton", decision["excluded"])


if __name__ == "__main__":
    unittest.main()
