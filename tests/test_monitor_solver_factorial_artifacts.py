"""Integrity tests for the sealed AC8 monitor/solver factorial."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


class MonitorSolverFactorialArtifactTests(unittest.TestCase):
    def test_ac8_protocol_sources_and_artifact_gate(self):
        protocol = load("results/expAC8/preregistration.json")
        training = load("results/expAC8/training.json")
        raw = load("results/expAC8/raw_results.json")
        summary = load("results/expAC8/summary.json")

        self.assertEqual(
            protocol_hash(protocol),
            protocol["integrity"]["protocol_sha256"],
        )
        for source in [
            protocol["frozen_teacher"],
            *protocol["frozen_sources"],
        ]:
            self.assertEqual(
                sha256_file(ROOT / source["path"]),
                source["sha256"],
                source["path"],
            )
        self.assertEqual(len(training["records"]), 10)
        self.assertEqual(len(raw["records"]), 264)
        self.assertEqual(summary["counts"]["attempted"], 264)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        self.assertEqual(
            sha256_file(ROOT / "results/expAC8/training.json"),
            summary["training_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / "results/expAC8/raw_results.json"),
            summary["raw_results_sha256"],
        )

    def test_ac8_factorial_is_complete_and_unique(self):
        raw = load("results/expAC8/raw_results.json")
        records = raw["records"]
        architecture_counts = Counter(
            record["architecture"] for record in records
        )
        solver_counts = Counter(record["solver"] for record in records)
        self.assertEqual(
            architecture_counts,
            {"raw": 120, "invariant": 120, "manual_r": 24},
        )
        self.assertEqual(
            solver_counts,
            {"fixed_point": 132, "damped_newton": 132},
        )
        keys = {
            (
                record["eccentricity"],
                record["orientation_radians"],
                record["momentum_direction"],
                record["architecture"],
                record["seed"],
                record["solver"],
            )
            for record in records
        }
        self.assertEqual(len(keys), len(records))
        self.assertTrue(
            all(record["completion_status"] != "running" for record in records)
        )

    def test_invariant_students_satisfy_exact_symmetries(self):
        training = load("results/expAC8/training.json")
        invariant = [
            record
            for record in training["records"]
            if record["architecture"] == "invariant"
        ]
        self.assertEqual(len(invariant), 5)
        for record in invariant:
            self.assertLess(
                record["symmetry"]["maximum_rotation_relative_error"],
                1e-12,
            )
            self.assertLess(
                record["symmetry"]["momentum_reversal_relative_error"],
                1e-12,
            )


if __name__ == "__main__":
    unittest.main()
