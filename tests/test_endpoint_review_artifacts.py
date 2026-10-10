"""Gate completed current replay artifacts, status changes and provenance."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/returned_endpoint_review"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class EndpointReviewArtifactTests(unittest.TestCase):
    def test_all_inputs_and_checkpoints_remain_hash_linked(self):
        manifest = read(OUT / "manifest.json")
        self.assertEqual(manifest["expected_records"], 1332)
        for s in manifest["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / s["path"]).read_bytes()).hexdigest(), s["sha256"], s["path"])
        verified = read(OUT / "verified/manifest.json")
        self.assertEqual(hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(), verified["parent_manifest_sha256"])
        for s in verified["sources"]:
            self.assertEqual(hashlib.sha256((ROOT / s["path"]).read_bytes()).hexdigest(), s["sha256"])

    def test_coverage_failures_and_actual_returned_residuals(self):
        manifest = read(OUT / "manifest.json")
        observed, verified = [], []
        for group in manifest["groups"]:
            op = read(OUT / "parts" / (group["id"] + ".json"))
            vp = read(OUT / "verified/parts" / (group["id"] + ".json"))
            self.assertIsNone(op["limit_steps"])
            self.assertIsNone(vp["limit_steps"])
            self.assertEqual(len(op["records"]), len(vp["records"]))
            observed.extend(op["records"])
            verified.extend(vp["records"])
        self.assertEqual(len(observed), 1332)
        self.assertEqual(len(verified), 1332)
        self.assertEqual(sum(r["returned_residual_violations"] for r in observed), 155)
        self.assertTrue(all(r["completion_status"] == r["historical_status"] for r in observed))
        self.assertTrue(all(r["returned_residual_violations"] == 0 for r in verified))
        self.assertTrue(all(r["maximum_all_returned_residual_inf"] < 1e-12 for r in verified))
        intervention = [r for r in verified if r["study"] == "AC19"]
        self.assertEqual(sum(r["completion_status"] == "complete" for r in intervention), 150)
        changes = [r for r in verified if r["historical_status"] != r["completion_status"]]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["completion_status"], "solver_failure")
        pairing = read(OUT / "paired_analysis.json")
        self.assertEqual(pairing["changed_case_scalar_confirmation"]["steps"], changes[0]["steps"])
        self.assertEqual(pairing["changed_case_scalar_confirmation"]["completion_status"], changes[0]["completion_status"])


if __name__ == "__main__":
    unittest.main()
