"""Tests for the sealed AC20 deployment-domain / Sundman-factored study."""

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

import expAC20_deployment_domain_training as ac20  # noqa: E402
from expAC9_end_to_end_monitor_generalisation import (  # noqa: E402
    Monitor,
    kepler_state,
)
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


class SundmanMonitorTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_cold_start_equals_manual_sundman_monitor(self):
        torch.manual_seed(0)
        monitor = ac20.SundmanMonitor()
        # zero the final layer so the sigmoid factor is exactly 1/2*2 = 1
        with torch.no_grad():
            monitor.network[-1].weight.zero_()
            monitor.network[-1].bias.zero_()
        anomalies = np.linspace(0.0, 2.0 * math.pi, 50, endpoint=False)
        states = torch.tensor(
            ac20.kepler_state_robust(1.0, 0.99, anomalies)
        )
        radii = torch.linalg.vector_norm(states[:, :2], dim=-1)
        self.assertTrue(torch.allclose(monitor(states), radii, atol=1e-12))

    def test_monitor_is_bounded_by_twice_the_radius_and_positive(self):
        torch.manual_seed(1)
        monitor = ac20.SundmanMonitor()
        with torch.no_grad():
            monitor.network[-1].bias.fill_(50.0)  # saturate the sigmoid
        states = torch.tensor(
            ac20.kepler_state_robust(0.75, 0.9, np.linspace(0, 6, 20))
        )
        radii = torch.linalg.vector_norm(states[:, :2], dim=-1)
        values = monitor(states)
        self.assertTrue(torch.all(values > 0))
        self.assertTrue(torch.all(values <= 2.0 * radii + 1e-12))

    def test_monitor_is_rotation_and_reversal_invariant(self):
        torch.manual_seed(2)
        monitor = ac20.SundmanMonitor()
        states = torch.tensor(
            ac20.kepler_state_robust(1.0, 0.9, np.linspace(0.1, 6.1, 15))
        )
        angle = 0.7
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle)],
             [math.sin(angle), math.cos(angle)]]
        )
        rotated = torch.cat(
            [states[:, :2] @ rotation.T, states[:, 2:] @ rotation.T], dim=1
        )
        reversed_momentum = torch.cat([states[:, :2], -states[:, 2:]], dim=1)
        self.assertTrue(torch.allclose(monitor(states), monitor(rotated),
                                       atol=1e-12))
        self.assertTrue(torch.allclose(monitor(states),
                                       monitor(reversed_momentum), atol=1e-12))

    def test_build_monitor_dispatch(self):
        self.assertIsInstance(ac20.build_monitor("sundman_e09"),
                              ac20.SundmanMonitor)
        self.assertIsInstance(ac20.build_monitor("sundman_e099"),
                              ac20.SundmanMonitor)
        invariant = ac20.build_monitor("invariant_e099")
        self.assertIsInstance(invariant, Monitor)
        self.assertEqual(invariant.architecture, "invariant")

    def test_robust_kepler_states_match_ac9_at_moderate_eccentricity(self):
        mean = np.linspace(0.1, 6.0, 7)
        expected = kepler_state(1.0, 0.9, mean)
        actual = ac20.kepler_state_robust(1.0, 0.9, mean)
        self.assertTrue(np.allclose(actual, expected, atol=1e-10))
        single = ac20.kepler_state_robust(1.0, 0.99, math.pi)
        self.assertEqual(single.shape, (4,))
        energy = Kepler2D(0.99).H(single[None, :2], single[None, 2:])
        self.assertAlmostEqual(float(energy[0]), -0.5, places=10)
        self.assertAlmostEqual(np.linalg.norm(single[:2]), 1.99, places=10)


class AC20ArtifactTests(unittest.TestCase):
    def test_protocol_if_registered(self):
        path = ROOT / "results" / "expAC20" / "preregistration.json"
        if not path.exists():
            self.skipTest("AC20 has not yet been registered")
        protocol = load("results/expAC20/preregistration.json")
        self.assertEqual(
            protocol_hash(protocol), protocol["integrity"]["protocol_sha256"]
        )
        self.assertEqual(set(protocol["arms"]), set(ac20.ARMS))
        self.assertEqual(protocol["diagnostic"]["contraction_threshold"], 1.0)
        self.assertEqual(protocol["acceptance"]["expected_rollouts"], 1080)
        self.assertTrue(protocol["acceptance"]["scientific_success_not_required"])
        self.assertEqual(len(protocol["training"]["seed_manifest"]), 5)
        ac9 = load("results/expAC9/preregistration.json")
        self.assertEqual(protocol["training"], ac9["training"])
        for source in protocol["frozen_sources"]:
            if source["role"] in ("runner", "prospective_analysis"):
                continue
            self.assertEqual(
                sha256_file(ROOT / source["path"]), source["sha256"],
                source["path"],
            )

    def test_completed_artifacts_if_generated(self):
        out = ROOT / "results" / "expAC20"
        if not (out / "analysis.json").exists():
            self.skipTest("AC20 has not yet been analysed")
        protocol = load("results/expAC20/preregistration.json")
        analysis = load("results/expAC20/analysis.json")
        sha = protocol["integrity"]["protocol_sha256"]
        self.assertEqual(analysis["protocol_sha256"], sha)
        self.assertTrue(analysis["acceptance"]["artifact_gate_passed"])
        self.assertEqual(analysis["counts"]["checkpoints"], 15)
        self.assertEqual(analysis["counts"]["orbit_records"], 540)
        self.assertEqual(analysis["counts"]["rollouts"], 1080)
        self.assertTrue(
            analysis["prospective_prediction"]
            ["diagnostic_completed_before_rollouts_started"]
        )
        rollouts = 0
        for arm in ac20.ARMS:
            for seed in ac20.SEEDS:
                directory = out / "parts" / f"{arm}_seed{seed}"
                training = load(directory.relative_to(ROOT) / "training.json")
                diagnostic = load(directory.relative_to(ROOT) / "diagnostic.json")
                part = load(directory.relative_to(ROOT) / "rollouts.json")
                for artifact in (training, diagnostic, part):
                    self.assertEqual(artifact["protocol_sha256"], sha)
                self.assertEqual(training["record"]["arm"], arm)
                self.assertEqual(training["record"]["checkpoint_seed"], seed)
                self.assertEqual(len(diagnostic["orbit_records"]), 36)
                self.assertLessEqual(
                    diagnostic["completed_at_utc"], part["started_at_utc"]
                )
                self.assertEqual(len(part["records"]), 72)
                self.assertEqual(
                    Counter(r["eccentricity"] for r in part["records"]),
                    {0.7: 24, 0.9: 24, 0.99: 24},
                )
                self.assertTrue(
                    all(r["completion_status"] != "running"
                        for r in part["records"])
                )
                rollouts += len(part["records"])
                checkpoint = ROOT / training["record"]["checkpoint_path"]
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(
                    sha256_file(checkpoint),
                    training["record"]["checkpoint_sha256"],
                )
                # Sundman-factored monitors must satisfy g <= 2r everywhere
                if arm.startswith("sundman"):
                    for record in diagnostic["orbit_records"]:
                        self.assertLessEqual(
                            record["monitor_over_radius_at_periapsis"],
                            2.0 + 1e-9,
                        )
        self.assertEqual(rollouts, 1080)
        for relative, digest in analysis["part_sha256"].items():
            self.assertEqual(sha256_file(ROOT / relative), digest, relative)
        for stem in ("fig_deployment_domain_training.pdf",
                     "fig_deployment_domain_training.png"):
            self.assertTrue((ROOT / stem).is_file(), stem)
            self.assertTrue((out / stem).is_file(), stem)
        # the baseline blocks must reproduce the frozen AC9 counts
        self.assertEqual(analysis["completion"]["ac9_invariant"]["complete"], 240)
        self.assertEqual(
            analysis["completion"]["ac9_invariant"]["by_eccentricity"]["0.99"]
            ["complete"], 0,
        )
        self.assertEqual(analysis["completion"]["ac9_manual_r"]["complete"], 72)


if __name__ == "__main__":
    unittest.main()
