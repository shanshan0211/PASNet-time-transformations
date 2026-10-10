"""Integrity tests for AC6/AC7 monitor-generalisation artifacts."""

from __future__ import annotations

import hashlib
import json
import unittest
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


class MonitorGeneralisationArtifactTests(unittest.TestCase):
    def verify_protocol_sources(self, protocol):
        self.assertEqual(
            protocol_hash(protocol),
            protocol["integrity"]["protocol_sha256"],
        )
        for source in [
            protocol["frozen_checkpoint"],
            *protocol["frozen_sources"],
        ]:
            self.assertEqual(
                sha256_file(ROOT / source["path"]),
                source["sha256"],
                source["path"],
            )

    def test_ac6_protocol_and_artifact_gate(self):
        protocol = load("results/expAC6/preregistration.json")
        raw = load("results/expAC6/raw_results.json")
        summary = load("results/expAC6/summary.json")
        self.verify_protocol_sources(protocol)
        self.assertEqual(len(raw["records"]), 80)
        self.assertEqual(summary["counts"]["attempted"], 80)
        self.assertEqual(summary["counts"]["complete"], 65)
        self.assertEqual(summary["counts"]["by_status"]["solver_failure"], 15)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        self.assertEqual(
            sha256_file(ROOT / "results/expAC6/raw_results.json"),
            summary["raw_results_sha256"],
        )

        manual = [
            record for record in raw["records"]
            if record["method"] == "manual_r"
        ]
        learned = [
            record for record in raw["records"]
            if record["method"] == "learned_g"
        ]
        self.assertEqual(len(manual), 40)
        self.assertTrue(
            all(record["completion_status"] == "complete" for record in manual)
        )
        self.assertEqual(
            sum(record["completion_status"] == "complete" for record in learned),
            25,
        )

    def test_ac7_is_exact_failed_scenario_refinement(self):
        protocol = load("results/expAC7/preregistration.json")
        raw = load("results/expAC7/raw_results.json")
        summary = load("results/expAC7/summary.json")
        self.verify_protocol_sources(protocol)
        self.assertEqual(len(protocol["failed_scenarios"]), 15)
        self.assertEqual(len(raw["records"]), 60)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        self.assertEqual(
            sha256_file(ROOT / "results/expAC7/raw_results.json"),
            summary["raw_results_sha256"],
        )
        scenario_keys = {
            (
                item["eccentricity"],
                item["orientation_radians"],
                item["momentum_direction"],
            )
            for item in protocol["failed_scenarios"]
        }
        self.assertEqual(
            {
                (
                    record["eccentricity"],
                    record["orientation_radians"],
                    record["momentum_direction"],
                )
                for record in raw["records"]
            },
            scenario_keys,
        )

    def test_recovered_multiseed_excludes_invalid_known_h(self):
        recovered = load("results/expMS/recovered_learnedH_summary.json")
        self.assertNotIn("ablation_A", recovered)
        self.assertIn("ablation_A", recovered["excluded_invalidated_fields"])
        self.assertEqual(recovered["n_seeds"], 3)
        source = ROOT / recovered["provenance"]["source_path"]
        if not source.is_file():
            # The local archive nests each paper in a second same-named folder.
            # Resolve the relocated input only; the recorded digest still gates it.
            source = ROOT.parents[1] / "paper1/paper1/results/expMS/multiseed_summary.json"
        self.assertEqual(
            sha256_file(source),
            recovered["provenance"]["source_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
