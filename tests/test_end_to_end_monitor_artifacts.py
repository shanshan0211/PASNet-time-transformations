"""Integrity tests for sealed AC9 end-to-end monitor artifacts."""

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


class EndToEndMonitorArtifactTests(unittest.TestCase):
    def test_ac9_protocol_sources_and_artifact_gate(self):
        protocol = load("results/expAC9/preregistration.json")
        training = load("results/expAC9/training.json")
        raw = load("results/expAC9/raw_results.json")
        summary = load("results/expAC9/summary.json")
        analysis = load("results/expAC9/analysis.json")

        self.assertEqual(
            protocol_hash(protocol),
            protocol["integrity"]["protocol_sha256"],
        )
        for source in protocol["frozen_sources"]:
            self.assertEqual(
                sha256_file(ROOT / source["path"]),
                source["sha256"],
                source["path"],
            )
        self.assertEqual(len(training["records"]), 10)
        self.assertEqual(len(raw["records"]), 792)
        self.assertEqual(summary["counts"]["attempted"], 792)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        self.assertTrue(analysis["artifact_gate"]["artifact_gate_passed"])
        self.assertEqual(
            sha256_file(ROOT / "results/expAC9/training.json"),
            summary["training_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / "results/expAC9/raw_results.json"),
            summary["raw_results_sha256"],
        )

    def test_ac9_registered_factorial_is_complete_and_unique(self):
        raw = load("results/expAC9/raw_results.json")
        records = raw["records"]
        self.assertEqual(
            Counter(record["architecture"] for record in records),
            {"raw": 360, "invariant": 360, "manual_r": 72},
        )
        keys = {
            (
                record["architecture"],
                record["checkpoint_seed"],
                record["semi_major_axis"],
                record["eccentricity"],
                record["initial_mean_anomaly"],
                record["orientation_radians"],
                record["momentum_direction"],
            )
            for record in records
        }
        self.assertEqual(len(keys), 792)
        self.assertEqual(
            {record["semi_major_axis"] for record in records},
            {0.75, 1.0, 1.25},
        )
        self.assertTrue(
            all(record["completion_status"] != "running" for record in records)
        )

    def test_ac9_checkpoint_and_symmetry_counts(self):
        training = load("results/expAC9/training.json")
        records = training["records"]
        self.assertEqual(
            Counter(record["architecture"] for record in records),
            {"raw": 5, "invariant": 5},
        )
        for record in records:
            checkpoint = ROOT / record["checkpoint_path"]
            self.assertEqual(
                sha256_file(checkpoint), record["checkpoint_sha256"]
            )
            if record["architecture"] == "invariant":
                self.assertLess(
                    max(record["symmetry"].values()), 1e-12
                )


if __name__ == "__main__":
    unittest.main()
