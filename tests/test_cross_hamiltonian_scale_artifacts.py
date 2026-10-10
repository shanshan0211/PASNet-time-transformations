"""Protocol and artifact integrity tests for sealed AC13."""

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
EXPERIMENT = ROOT / "experiments" / "expAC13_cross_hamiltonian_scale.py"
SPEC = importlib.util.spec_from_file_location("expAC13", EXPERIMENT)
AC13 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AC13
SPEC.loader.exec_module(AC13)


def load(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
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


class CrossHamiltonianScaleTests(unittest.TestCase):
    def test_reversal_even_features_enforce_only_momentum_reversal(self):
        for system_name, state_dimension in [("modal", 32), ("duffing", 2)]:
            state = torch.randn(5, state_dimension, dtype=torch.float64)
            reversed_state = state.clone()
            reversed_state[:, state_dimension // 2 :] *= -1
            self.assertTrue(
                torch.equal(
                    AC13.reversal_even_features(state, system_name),
                    AC13.reversal_even_features(reversed_state, system_name),
                )
            )
            negated = -state
            self.assertFalse(
                torch.equal(
                    AC13.reversal_even_features(state, system_name),
                    AC13.reversal_even_features(negated, system_name),
                )
            )

    def test_system_dimensions_and_short_resolved_references(self):
        modal, modal_state = AC13.initial_state("modal", 1.0, 0, kappa=10)
        duffing, duffing_state = AC13.initial_state("duffing", 1.0, 0)
        self.assertEqual(modal_state.shape, (32,))
        self.assertEqual(duffing_state.shape, (2,))
        for name, system, state in [
            ("modal", modal, modal_state),
            ("duffing", duffing, duffing_state),
        ]:
            times, states = AC13.velocity_verlet_trajectory(
                name, system, state, 0.01, 0.001, 0.001
            )
            self.assertAlmostEqual(times[-1], 0.01, places=14)
            self.assertEqual(states.shape[1], state.shape[0])

    def test_sealed_protocol_sources_and_registered_factorial(self):
        protocol = load("results/expAC13/preregistration.json")
        self.assertEqual(
            protocol_hash(protocol),
            protocol["integrity"]["protocol_sha256"],
        )
        for source in protocol["frozen_sources"]:
            assert_frozen_or_registered_remediation(self, source)
        self.assertEqual(protocol["systems"]["modal"]["state_dimension"], 32)
        self.assertEqual(
            protocol["training"]["seed_manifest"][0]["checkpoint_seed"], 9101
        )
        self.assertEqual(
            protocol["training"]["seed_manifest"][-1]["checkpoint_seed"], 9103
        )
        self.assertEqual(protocol["evaluation"]["modal"]["dtau"], 0.05)
        self.assertEqual(protocol["evaluation"]["duffing"]["dtau"], 0.1)
        self.assertFalse(protocol["evaluation"]["newton_included"])
        self.assertEqual(
            protocol["evaluation"]["total_registered_rollouts"], 288
        )
        modal = AC13.scenarios(protocol, "modal")
        duffing = AC13.scenarios(protocol, "duffing")
        self.assertEqual(len(modal), 24)
        self.assertEqual(len(duffing), 12)
        learned = 6 * (len(modal) + len(duffing))
        controls = 2 * (len(modal) + len(duffing))
        self.assertEqual(learned + controls, 288)

    def test_runtime_artifacts_when_present(self):
        summary_path = ROOT / "results" / "expAC13" / "summary.json"
        if not summary_path.exists():
            self.skipTest("long AC13 experiment intentionally not run")
        training = load("results/expAC13/training.json")
        raw = load("results/expAC13/raw_results.json")
        summary = load("results/expAC13/summary.json")
        self.assertEqual(len(training["records"]), 12)
        self.assertEqual(len(raw["records"]), 288)
        self.assertEqual(
            Counter(record["method"] for record in raw["records"]),
            {
                "learned_monitor": 216,
                "prescribed_monitor": 36,
                "resolved_velocity_verlet": 36,
            },
        )
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        for record in training["records"]:
            if record["status"] != "training_failure":
                checkpoint = ROOT / record["checkpoint_path"]
                self.assertEqual(
                    sha256_file(checkpoint), record["checkpoint_sha256"]
                )


if __name__ == "__main__":
    unittest.main()
