"""Regression, smoke, protocol, and optional artifact tests for AC18."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))


def load_module(filename, name):
    path = ROOT / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def minimal_protocol():
    return {
        "experiment": "expAC18_smoke",
        "hamiltonian_training": {
            "candidate_seeds": [1801, 1802, 1803, 1804, 1805],
            "training_eccentricities": [0.72, 0.81, 0.9, 0.96, 0.985],
            "states_per_eccentricity": 512,
            "phase_offset": 0.5,
            "learning_rate": 0.001,
        },
        "hamiltonian_fidelity_gate": {
            "held_out_eccentricities": [0.7, 0.75, 0.85, 0.95, 0.99],
            "held_out_states_per_eccentricity": 257,
            "held_out_phase_offset": 0.1732050807568877,
            "maximum_normalized_rmse": 0.01,
            "maximum_stratum_normalized_rmse": 0.02,
            "maximum_relative_error_p95": 0.03,
            "maximum_relative_error_max": 0.15,
            "minimum_accepted_checkpoints": 3,
        },
        "monitor_training": {
            "training_orbit_eccentricity": 0.9,
        },
        "evaluation": {
            "eccentricities": [0.7, 0.9, 0.99],
            "initial_mean_anomaly": math.pi,
        },
        "state_generation": {
            "residual_tolerance": 2e-14,
            "bisection_iterations": 80,
        },
    }


class AC18StateGenerationAndSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.runner = load_module(
            "expAC18_robust_multi_learned_h_factorial.py", "expAC18_runner"
        )
        cls.legacy = load_module(
            "expAC12_learned_h_monitor_solver_factorial.py",
            "expAC12_legacy_solver_regression",
        )

    def test_ac17_registered_grid_reproduces_legacy_solver_failure(self):
        protocol = minimal_protocol()
        eccentricities, anomalies = self.runner.registered_anomaly_grids(
            protocol
        )["hamiltonian_training"]
        failures = 0
        for eccentricity in eccentricities:
            try:
                self.legacy.solve_eccentric_anomaly(
                    anomalies, eccentricity
                )
            except RuntimeError:
                failures += 1
        self.assertGreater(
            failures,
            0,
            "regression fixture must expose AC17's infrastructure failure",
        )

    def test_robust_solver_residual_on_every_registered_grid_point(self):
        protocol = minimal_protocol()
        records = self.runner.validate_registered_state_generation(protocol)
        self.assertEqual(len(records), 14)
        self.assertTrue(all(record["all_states_finite"] for record in records))
        self.assertTrue(all(record["minimum_radius"] > 0 for record in records))
        self.assertLessEqual(
            max(record["maximum_residual"] for record in records),
            protocol["state_generation"]["residual_tolerance"],
        )

    def test_solver_handles_boundaries_cycles_and_high_eccentricity(self):
        anomalies = np.asarray([
            -4.0 * math.pi,
            -0.1,
            0.0,
            np.nextafter(2.0 * math.pi, 0.0),
            2.0 * math.pi,
            12.0 * math.pi + 0.3,
        ])
        for eccentricity in [0.0, 0.7, 0.99, 0.999999]:
            solution = self.runner.solve_eccentric_anomaly_bisection(
                anomalies, eccentricity
            )
            residual = (
                solution - eccentricity * np.sin(solution) - anomalies
            )
            self.assertLessEqual(float(np.max(np.abs(residual))), 2e-14)

    def test_smoke_enters_optimizer_and_held_out_gate(self):
        self.runner.configure_ac17_runtime()
        result = self.runner.smoke_h_training_and_gating(
            minimal_protocol()
        )
        self.assertEqual(result["status"], "smoke_complete")
        self.assertEqual(result["optimizer_steps_completed"], 1)
        self.assertTrue(result["finite_training_loss"])
        self.assertTrue(result["gate_metrics_computed"])
        self.assertTrue(math.isfinite(
            result["held_out_gradient_normalized_rmse"]
        ))
        self.assertTrue(math.isfinite(
            result["held_out_vector_field_normalized_rmse"]
        ))
        self.assertFalse(result["scientific_thresholds_applied_to_smoke"])

    def test_ac17_is_additively_invalidated(self):
        invalidation = load(ROOT / "results" / "expAC17" / "invalidation.json")
        self.assertEqual(
            invalidation["status"], "invalidated_infrastructure_failure"
        )
        self.assertEqual(
            invalidation["evidence"]["candidates_entering_optimizer_training"],
            0,
        )
        self.assertIn(
            "No fidelity-gate null",
            invalidation["scientific_interpretation"],
        )


class AC18ProtocolAndArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = ROOT / "results" / "expAC18"
        cls.protocol_path = cls.out / "preregistration.json"

    def test_protocol_integrity_if_registered(self):
        if not self.protocol_path.exists():
            self.skipTest("AC18 has not yet been sealed")
        protocol = load(self.protocol_path)
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertNotIn(
            "training", protocol,
            "monitor configuration must not be duplicated",
        )
        self.assertIn("monitor_training", protocol)
        gate = protocol["hamiltonian_fidelity_gate"]
        self.assertEqual(gate["maximum_normalized_rmse"], 0.01)
        self.assertEqual(gate["maximum_stratum_normalized_rmse"], 0.02)
        self.assertEqual(gate["maximum_relative_error_p95"], 0.03)
        self.assertEqual(gate["maximum_relative_error_max"], 0.15)
        self.assertEqual(gate["minimum_accepted_checkpoints"], 3)
        self.assertTrue(protocol["state_generation"]["preflight_required"])
        self.assertTrue(
            protocol["infrastructure_failure_policy"][
                "cannot_be_interpreted_as_fidelity_rejection"
            ]
        )
        for source in protocol["frozen_sources"]:
            path = ROOT / source["path"]
            self.assertTrue(path.is_file(), source["path"])
            self.assertEqual(sha256_file(path), source["sha256"])

    def test_result_artifacts_if_generated(self):
        audit_path = self.out / "hamiltonian_audit.json"
        if not audit_path.exists():
            self.skipTest("AC18 full experiment has not been run")
        self.assertFalse(
            (self.out / "infrastructure_failure.json").exists(),
            "a scientific result is invalid if infrastructure failed",
        )
        audit = load(audit_path)
        self.assertEqual(len(audit["records"]), 5)
        self.assertTrue(all(
            record["training_status"] != "training_or_audit_failure"
            for record in audit["records"]
        ))


if __name__ == "__main__":
    unittest.main()
