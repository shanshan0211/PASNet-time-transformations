"""Tests for the prospectively specified AC14 correction."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_expAC14 import exact_step_rows, terminal_rows, window_rows  # noqa: E402
from monitor_conditioning import (  # noqa: E402
    canonical_symplectic_matrix,
    kepler_shell_scaling,
    nondimensional_endpoint_diagnostics,
)


def protocol_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample(step, q, kind="accepted_stride"):
    return {
        "sample_kind": kind,
        "accepted_physical_step": step,
        "corrected_conditioning": {
            "contraction_q": q,
            "scaled_endpoint_jacobian_sigma_min": 1.0 / q,
            "scaled_endpoint_jacobian_condition_2": q + 1.0,
        },
    }


class EndpointScalingTests(unittest.TestCase):
    def test_similarity_transform_uses_h_without_extra_time_factor(self):
        d2_k = torch.tensor(
            [
                [2.0, 0.1, 0.2, 0.0],
                [0.1, 3.0, 0.0, -0.3],
                [0.2, 0.0, 4.0, 0.4],
                [0.0, -0.3, 0.4, 5.0],
            ],
            dtype=torch.float64,
        )
        h = 0.08
        a = 4.0
        actual = nondimensional_endpoint_diagnostics(d2_k, h, a)
        scaling, inverse = kepler_shell_scaling(a)
        operator = (
            inverse @ canonical_symplectic_matrix() @ d2_k @ scaling
        )
        expected = torch.eye(4) - 0.5 * h * operator
        singular = torch.linalg.svdvals(expected)
        self.assertEqual(
            actual["endpoint_jacobian_formula"],
            "I-(h/2) S^-1 J D2zzK S",
        )
        self.assertAlmostEqual(
            actual["contraction_q"],
            float(0.5 * h * torch.linalg.matrix_norm(operator, ord=2)),
            places=12,
        )
        self.assertAlmostEqual(
            actual["scaled_endpoint_jacobian_sigma_min"],
            float(torch.min(singular)),
            places=12,
        )
        erroneous = torch.eye(4) - 0.5 * h * a**-1.5 * operator
        self.assertFalse(torch.allclose(expected, erroneous))


class MatchedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "rollout_key": "complete",
                "eventual_completion_status": "complete",
                "samples": [
                    sample(10, 1.0),
                    sample(20, 2.0),
                    sample(21, 99.0, "accepted_horizon_trim"),
                ],
            },
            {
                "rollout_key": "failed",
                "eventual_completion_status": "solver_failure",
                "samples": [
                    sample(10, 3.0),
                    sample(20, 4.0),
                    sample(21, 1000.0, "failed_step_final_iterate"),
                ],
            },
        ]

    def test_matched_step_and_window_exclude_terminal_iterates(self):
        exact = exact_step_rows(self.records, 20)
        window = window_rows(self.records, 20)
        self.assertEqual([row["q"] for row in exact], [2.0, 4.0])
        self.assertEqual([row["q"] for row in window], [2.0, 4.0])

    def test_terminal_sensitivity_is_separate(self):
        terminal = terminal_rows(self.records)
        self.assertEqual([row["q"] for row in terminal], [99.0, 1000.0])


class AC14ProtocolTests(unittest.TestCase):
    def test_protocol_if_registered(self):
        path = ROOT / "results" / "expAC14" / "preregistration.json"
        if not path.exists():
            self.skipTest("AC14 has not yet been registered")
        with path.open(encoding="utf-8") as handle:
            protocol = json.load(handle)
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(
            protocol["scaling_and_derivation"]["endpoint_jacobian"],
            "I-(h/2) S^-1 J D2zzK S",
        )
        self.assertTrue(
            protocol["analysis"][
                "terminal_failed_iterates_excluded_from_matched_analyses"
            ]
        )
        self.assertTrue(protocol["analysis"]["no_causal_claims"])

    def test_completed_artifacts_if_generated(self):
        out = ROOT / "results" / "expAC14"
        raw_path = out / "raw_results.json"
        analysis_path = out / "analysis.json"
        if not raw_path.exists():
            self.skipTest("AC14 correction has not yet been run")
        with raw_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        with analysis_path.open(encoding="utf-8") as handle:
            analysis = json.load(handle)
        self.assertEqual(len(raw["records"]), 1056)
        self.assertEqual(raw["source_sample_count"], 71069)
        self.assertEqual(
            len({record["rollout_key"] for record in raw["records"]}), 1056
        )
        for step in ("10", "50", "100"):
            self.assertEqual(
                analysis["matched_step_accepted_only"][step]["n_rollouts"],
                1056,
            )
            self.assertEqual(
                analysis["matched_window_accepted_only"][step]["n_rollouts"],
                1056,
            )
        self.assertFalse(
            analysis["figure_estimand"]["terminal_failed_iterates_included"]
        )
        self.assertTrue((out / "conditioning_association.pdf").is_file())
        self.assertTrue((out / "conditioning_association.png").is_file())


if __name__ == "__main__":
    unittest.main()
