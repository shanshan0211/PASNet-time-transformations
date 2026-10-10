import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIFT_PATH = ROOT / "results" / "remediation_source_drift.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_frozen_or_registered_remediation(testcase, source):
    path = ROOT / source["path"]
    testcase.assertTrue(path.is_file(), source["path"])
    actual = sha256_file(path)
    if actual == source["sha256"]:
        return

    registry = json.loads(DRIFT_PATH.read_text(encoding="utf-8"))
    matches = [
        record
        for record in registry["records"]
        if record["path"] == source["path"]
        and record["sealed_sha256"] == source["sha256"]
        and record["current_sha256"] == actual
    ]
    testcase.assertEqual(
        len(matches),
        1,
        f"unregistered source drift: {source['path']} "
        f"{source['sha256']} -> {actual}",
    )
    successor = ROOT / matches[0]["successor_protocol"]
    testcase.assertTrue(successor.is_file(), str(successor))
