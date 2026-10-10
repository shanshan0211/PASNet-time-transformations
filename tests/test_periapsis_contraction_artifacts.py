"""Tests for the sealed AC19 periapsis contraction diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from expAC19_periapsis_contraction_diagnostic import (  # noqa: E402
    manual_radius_monitor,
    orbit_diagnostic,
    radius_at_physical_time,
    solve_kepler_equation,
    state_from_eccentric_anomaly,
)
from expAC9_end_to_end_monitor_generalisation import kepler_state  # noqa: E402
from systems import Kepler2D  # noqa: E402


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


class OrbitConstructionTests(unittest.TestCase):
    def test_eccentric_anomaly_states_match_ac9_parameterisation(self):
        anomalies = np.linspace(0.2, 6.0, 9)
        mean = anomalies - 0.9 * np.sin(anomalies)
        expected = kepler_state(1.0, 0.9, mean)
        actual = state_from_eccentric_anomaly(1.0, 0.9, anomalies)
        self.assertTrue(np.allclose(actual, expected, atol=1e-10))

    def test_states_lie_on_energy_shell_and_reach_periapsis(self):
        anomalies = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        states = state_from_eccentric_anomaly(0.75, 0.99, anomalies)
        energy = Kepler2D(0.99).H(states[:, :2], states[:, 2:])
        self.assertTrue(np.allclose(energy, -0.5 / 0.75, atol=1e-10))
        radii = np.linalg.norm(states[:, :2], axis=1)
        self.assertAlmostEqual(radii.min(), 0.75 * 0.01, places=12)
        self.assertAlmostEqual(radii.max(), 0.75 * 1.99, places=12)

    def test_kepler_equation_solver_handles_high_eccentricity(self):
        for mean in (1e-4, 0.3, math.pi, 5.9):
            anomaly = solve_kepler_equation(mean, 0.99)
            self.assertAlmostEqual(
                anomaly - 0.99 * math.sin(anomaly), mean, places=12
            )

    def test_failure_time_maps_apoapsis_start_to_periapsis_at_half_period(self):
        scenario = {
            "semi_major_axis": 1.0,
            "eccentricity": 0.99,
            "initial_mean_anomaly": math.pi,
            "momentum_direction": "forward",
        }
        radius, _ = radius_at_physical_time(scenario, math.pi)
        self.assertAlmostEqual(radius, 0.01, places=9)
        scenario["momentum_direction"] = "reversed"
        radius, _ = radius_at_physical_time(scenario, math.pi)
        self.assertAlmostEqual(radius, 0.01, places=9)


class ContractionOperatorTests(unittest.TestCase):
    def test_manual_monitor_spectral_radius_matches_closed_form(self):
        """For g=r at periapsis of a Kepler orbit the linearised operator is
        (dtau/2) J D2K with an eigenvalue magnitude that must agree with a
        direct dense computation."""
        torch.set_default_dtype(torch.float64)
        anomalies = np.array([0.0, 1.0, math.pi])
        states = state_from_eccentric_anomaly(1.0, 0.99, anomalies)
        system = Kepler2D(0.99)
        diagnostic = orbit_diagnostic(
            manual_radius_monitor, system, states, 0.04, batch=2
        )
        z = torch.tensor(states[:1], requires_grad=True)
        p0 = -system.H_torch(torch.tensor(states[:1])).detach()

        def extended(value):
            return (
                manual_radius_monitor(value)
                * (system.H_torch(value) + p0)
            ).sum()

        hessian = torch.autograd.functional.hessian(extended, z).reshape(4, 4)
        j_matrix = torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]],
            dtype=torch.float64,
        )
        operator = 0.02 * j_matrix @ hessian
        expected = torch.linalg.eigvals(operator).abs().max().item()
        self.assertAlmostEqual(
            diagnostic["spectral_radius"][0], expected, places=10
        )
        self.assertAlmostEqual(diagnostic["radius"][0], 0.01, places=12)
        self.assertAlmostEqual(diagnostic["monitor"][0], 0.01, places=12)
        self.assertGreaterEqual(
            diagnostic["spectral_norm"][0], diagnostic["spectral_radius"][0]
        )

    def test_spectral_radius_is_linear_in_dtau(self):
        torch.set_default_dtype(torch.float64)
        states = state_from_eccentric_anomaly(1.0, 0.9, np.array([0.0, 2.0]))
        system = Kepler2D(0.9)
        coarse = orbit_diagnostic(manual_radius_monitor, system, states, 0.04)
        fine = orbit_diagnostic(manual_radius_monitor, system, states, 0.01)
        self.assertTrue(
            np.allclose(
                coarse["spectral_radius"], 4.0 * fine["spectral_radius"],
                rtol=1e-10,
            )
        )


class AC19ArtifactTests(unittest.TestCase):
    def test_protocol_if_registered(self):
        path = ROOT / "results" / "expAC19" / "preregistration.json"
        if not path.exists():
            self.skipTest("AC19 has not yet been registered")
        protocol = load("results/expAC19/preregistration.json")
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(protocol["phase_a"]["contraction_threshold"], 1.0)
        self.assertEqual(protocol["phase_b"]["rho_target"], 0.5)
        self.assertTrue(protocol["acceptance"]["scientific_success_not_required"])
        frozen = {source["path"] for source in protocol["frozen_sources"]}
        self.assertIn("results/expAC9/raw_results.json", frozen)
        for source in protocol["frozen_sources"]:
            if source["role"] in ("runner", "prospective_analysis"):
                continue
            self.assertEqual(
                sha256_file(ROOT / source["path"]), source["sha256"],
                source["path"],
            )

    def test_completed_artifacts_if_generated(self):
        out = ROOT / "results" / "expAC19"
        if not (out / "analysis.json").exists():
            self.skipTest("AC19 has not yet been analysed")
        protocol = load("results/expAC19/preregistration.json")
        raw = load("results/expAC19/raw_results.json")
        intervention = load("results/expAC19/intervention.json")
        analysis = load("results/expAC19/analysis.json")
        self.assertEqual(raw["protocol_sha256"], protocol["integrity"]["protocol_sha256"])
        self.assertEqual(
            intervention["protocol_sha256"],
            protocol["integrity"]["protocol_sha256"],
        )
        self.assertEqual(len(raw["orbit_records"]), 396)
        self.assertEqual(
            Counter(record["architecture"] for record in raw["orbit_records"]),
            {"manual_r": 36, "raw": 180, "invariant": 180},
        )
        self.assertEqual(
            len(raw["failure_locations"]),
            protocol["acceptance"]["expected_failed_rollouts"],
        )
        self.assertEqual(
            len(intervention["records"]),
            protocol["acceptance"]["expected_failed_rollouts"],
        )
        self.assertTrue(
            all(
                record["completion_status"] != "running"
                for record in intervention["records"]
            )
        )
        self.assertEqual(
            analysis["classification"]["all_792_rollouts"]["n"], 792
        )
        self.assertTrue(analysis["acceptance"]["artifact_gate_passed"])
        self.assertEqual(
            analysis["raw_results_sha256"],
            sha256_file(out / "raw_results.json"),
        )
        self.assertEqual(
            analysis["intervention_sha256"],
            sha256_file(out / "intervention.json"),
        )
        for stem in ("fig_periapsis_mechanism.pdf", "fig_periapsis_mechanism.png"):
            self.assertTrue((ROOT / stem).is_file(), stem)
            self.assertTrue((out / stem).is_file(), stem)
        # the manual monitor must satisfy g/r = 1 at periapsis by construction
        for record in raw["orbit_records"]:
            if record["architecture"] == "manual_r":
                self.assertAlmostEqual(
                    record["monitor_over_radius_at_periapsis"], 1.0, places=9
                )


if __name__ == "__main__":
    unittest.main()
