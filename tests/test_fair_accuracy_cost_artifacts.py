"""Integrity tests for sealed AC10 fair accuracy--cost artifacts."""

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


class FairAccuracyCostArtifactTests(unittest.TestCase):
    def test_ac10_protocol_sources_and_gate(self):
        protocol = load("results/expAC10/preregistration.json")
        raw = load("results/expAC10/raw_results.json")
        summary = load("results/expAC10/summary.json")
        analysis = load("results/expAC10/analysis.json")
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
        self.assertEqual(len(raw["records"]), 684)
        self.assertEqual(summary["counts"]["attempted"], 684)
        self.assertTrue(summary["acceptance"]["artifact_gate_passed"])
        self.assertTrue(analysis["artifact_gate"]["artifact_gate_passed"])
        self.assertEqual(
            sha256_file(ROOT / "results/expAC10/raw_results.json"),
            summary["raw_results_sha256"],
        )

    def test_ac10_configuration_memory_and_counts(self):
        raw = load("results/expAC10/raw_results.json")
        records = raw["records"]
        counts = Counter(
            record["configuration_id"] for record in records
        )
        self.assertEqual(len(counts), 38)
        self.assertTrue(all(count == 18 for count in counts.values()))
        self.assertEqual(len(raw["configuration_memory"]), 38)
        self.assertTrue(
            all(
                record["configuration_peak_rss_kib"] > 0
                for record in records
            )
        )

    def test_ac10_robust_front_excludes_incomplete_methods(self):
        analysis = load("results/expAC10/analysis.json")
        fronts = analysis["robust_pareto_fronts"]
        for identifiers in fronts.values():
            self.assertTrue(
                all(
                    identifier.startswith("manual_r_")
                    or identifier.startswith("adaptive_")
                    for identifier in identifiers
                )
            )
        failures = analysis["failure_classification"]
        self.assertEqual(
            failures["raw_artifact_statuses"]["unexpected_error"], 20
        )
        self.assertEqual(
            failures["analysis_normalized_reasons"][
                "direct_fixed_point_nonconvergence"
            ],
            20,
        )


if __name__ == "__main__":
    unittest.main()
