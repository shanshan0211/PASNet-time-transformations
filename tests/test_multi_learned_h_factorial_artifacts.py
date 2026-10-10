"""Unit, smoke, protocol, and optional artifact tests for sealed AC17."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.source_integrity import assert_frozen_or_registered_remediation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


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


def load_module(filename, name):
    path = ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AC17UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.runner = load_module(
            "expAC17_multi_learned_h_factorial.py", "expAC17_runner"
        )
        cls.analysis = load_module("analyze_expAC17.py", "analyze_expAC17")

    def test_exact_gradient_and_vector_field_are_canonical(self):
        anomalies = np.asarray([0.2, 1.7, 5.1])
        states, gradients, fields, labels = self.runner.states_and_fields(
            [0.7, 0.99], anomalies
        )
        self.assertEqual(states.shape, (6, 4))
        self.assertEqual(labels.tolist(), [0.7] * 3 + [0.99] * 3)
        np.testing.assert_allclose(fields[:, :2], gradients[:, 2:])
        np.testing.assert_allclose(fields[:, 2:], -gradients[:, :2])
        q, p = states[:, :2], states[:, 2:]
        radius = np.linalg.norm(q, axis=1, keepdims=True)
        np.testing.assert_allclose(gradients[:, :2], q / radius**3)
        np.testing.assert_allclose(gradients[:, 2:], p)

    def test_h_gradient_autograd_path_and_float64(self):
        model = self.runner.make_h_template()
        states = torch.tensor(
            self.runner.states_and_fields([0.9], np.asarray([0.4, 2.0]))[0],
            dtype=torch.float64,
        )
        prediction = self.runner.learned_gradient(
            model, states, create_graph=True
        )
        self.assertEqual(prediction.shape, states.shape)
        self.assertEqual(prediction.dtype, torch.float64)
        loss = prediction.square().mean()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

    def test_monitor_seed_manifest_is_nested_and_unique(self):
        manifests = [
            self.runner.monitor_seed_entry(h_seed, replicate)
            for h_seed in [1701, 1702, 1703]
            for replicate in [1, 2]
        ]
        self.assertEqual(
            len({entry["sampling_seed"] for entry in manifests}), 6
        )
        self.assertEqual(
            [entry["checkpoint_seed"] for entry in manifests], [1, 2] * 3
        )

    def test_h_level_analysis_does_not_count_rollouts_as_clusters(self):
        records = []
        for h_seed in [1, 2, 3]:
            for architecture in ["raw", "invariant"]:
                for solver in ["fixed_point", "monotone_residual_newton"]:
                    for replicate in [1, 2]:
                        for scenario in range(4):
                            records.append({
                                "h_seed": h_seed,
                                "architecture": architecture,
                                "solver": solver,
                                "monitor_replicate": replicate,
                                "completion_status": (
                                    "complete"
                                    if architecture == "invariant"
                                    else "solver_failure"
                                ),
                                "scenario": scenario,
                            })
        result = self.analysis.h_level_estimands(records, n_draws=100)
        self.assertEqual(result["n_h_checkpoints"], 3)
        self.assertTrue(
            result[
                "monitor_replicates_are_nested_not_independent_h_replicates"
            ]
        )
        self.assertEqual(
            result["estimands"]["invariant_minus_raw"][
                "mean_across_h_checkpoints"
            ],
            1.0,
        )

    def test_gate_failure_is_valid_zero_factorial_branch(self):
        protocol = {
            "experiment": "test_ac17",
            "hamiltonian_training": {"candidate_seeds": [1, 2, 3, 4, 5]},
            "monitor_training": {
                "architectures": ["raw", "invariant"],
                "replicates": [1, 2],
            },
            "evaluation": {
                "eccentricities": [0.7, 0.9, 0.99],
                "orientations_radians": [0.0, np.pi / 2],
                "momentum_directions": ["forward", "reversed"],
                "acceptance_tolerance": 1e-12,
            },
            "claim_limits": [],
            "integrity": {"protocol_sha256": "test"},
        }
        audit = {
            "gate_was_applied_before_monitor_training": True,
            "ensemble_qualified": False,
            "records": [
                {
                    "h_seed": seed,
                    "accepted": False,
                    "rejection_reasons": ["vector_field:normalized_rmse"],
                }
                for seed in [1, 2, 3, 4, 5]
            ],
        }
        training = {"records": []}
        raw = {"records": []}
        original_paths = (
            self.runner.H_AUDIT_PATH,
            self.runner.TRAINING_PATH,
            self.runner.RAW_PATH,
        )
        try:
            self.runner.H_AUDIT_PATH = __file__
            self.runner.TRAINING_PATH = __file__
            self.runner.RAW_PATH = __file__
            summary = self.runner.summarize(
                protocol, audit, training, raw
            )
        finally:
            (
                self.runner.H_AUDIT_PATH,
                self.runner.TRAINING_PATH,
                self.runner.RAW_PATH,
            ) = original_paths
        self.assertEqual(
            summary["counts"]["registered_rollouts_conditional_on_gate"], 0
        )
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])


class AC17ProtocolAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = ROOT / "results" / "expAC17"
        cls.protocol_path = cls.out / "preregistration.json"

    def test_protocol_integrity_if_registered(self):
        if not self.protocol_path.exists():
            self.skipTest("AC17 has not yet been sealed")
        protocol = load(self.protocol_path)
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        gate = protocol["hamiltonian_fidelity_gate"]
        self.assertTrue(gate["prospectively_fixed"])
        self.assertTrue(gate["held_out_from_h_training"])
        self.assertEqual(gate["minimum_accepted_checkpoints"], 3)
        self.assertEqual(
            protocol["hamiltonian_training"]["candidate_seeds"],
            [1701, 1702, 1703, 1704, 1705],
        )
        self.assertEqual(
            protocol["design"]["primary_cluster"], "learned-H checkpoint"
        )
        self.assertEqual(protocol["monitor_training"]["replicates"], [1, 2])
        for source in protocol["frozen_sources"]:
            assert_frozen_or_registered_remediation(self, source)

    def test_result_artifacts_if_generated(self):
        audit_path = self.out / "hamiltonian_audit.json"
        if not audit_path.exists():
            self.skipTest("AC17 full experiment has not been run")
        audit = load(audit_path)
        training = load(self.out / "training.json")
        raw = load(self.out / "raw_results.json")
        summary = load(self.out / "summary.json")
        self.assertEqual(len(audit["records"]), 5)
        self.assertTrue(all(
            "rejection_reasons" in record for record in audit["records"]
        ))
        accepted = sum(record["accepted"] for record in audit["records"])
        if accepted < 3:
            self.assertFalse(audit["ensemble_qualified"])
            self.assertEqual(training["records"], [])
            self.assertEqual(raw["records"], [])
        else:
            self.assertTrue(audit["ensemble_qualified"])
            self.assertEqual(len(training["records"]), accepted * 4)
            self.assertEqual(len(raw["records"]), accepted * 96)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])


if __name__ == "__main__":
    unittest.main()
