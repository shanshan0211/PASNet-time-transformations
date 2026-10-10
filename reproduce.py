"""Restore the minimal evidence bundle and validate the published numerical inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path outside repository: " + name)
    return path


def verify_inputs(restore=True):
    manifest = json.loads((ROOT / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    evidence = [r for r in manifest["files"] if r["path"].startswith("results/")]
    bundle = manifest["evidence_bundle"]
    for part in bundle["parts"]:
        path = local_path(part["path"])
        if path.stat().st_size != part["bytes"] or sha(path) != part["sha256"]:
            raise ValueError("Evidence part mismatch: " + part["path"])
    if restore:
        cache = ROOT / ".evidence-cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            packed = Path(temp) / "evidence.zip"
            with packed.open("wb") as out:
                for part in bundle["parts"]:
                    with local_path(part["path"]).open("rb") as inp:
                        shutil.copyfileobj(inp, out)
            if packed.stat().st_size != bundle["bytes"] or sha(packed) != bundle["sha256"]:
                raise ValueError("Combined evidence archive mismatch")
            with zipfile.ZipFile(packed) as archive:
                names = [r["path"] for r in evidence]
                if set(archive.namelist()) != set(names) or len(archive.namelist()) != len(names):
                    raise ValueError("Unexpected or duplicate archive entries")
                for record in evidence:
                    target = local_path(record["path"])
                    if target.exists():
                        if target.stat().st_size != record["bytes"] or sha(target) != record["sha256"]:
                            raise ValueError("Existing result differs; not overwriting: " + record["path"])
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix(target.suffix + ".restore-tmp")
                    try:
                        with archive.open(record["path"]) as inp, tmp.open("wb") as out:
                            shutil.copyfileobj(inp, out)
                        if tmp.stat().st_size != record["bytes"] or sha(tmp) != record["sha256"]:
                            raise ValueError("Restored evidence mismatch: " + record["path"])
                        tmp.replace(target)
                    finally:
                        tmp.unlink(missing_ok=True)
    checked = 0
    for record in manifest["files"]:
        if record.get("mutable_report"):
            continue
        if not restore and record["path"].startswith("results/"):
            continue
        target = local_path(record["path"])
        if target.stat().st_size != record["bytes"] or sha(target) != record["sha256"]:
            raise ValueError("Input mismatch: " + record["path"])
        checked += 1
    print(f"Verified {checked} immutable files and {len(bundle['parts'])} evidence parts", flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify package hashes without restoring evidence")
    parser.add_argument("--restore-only", action="store_true", help="Restore and verify evidence without running tests")
    args = parser.parse_args()
    if args.verify_only and args.restore_only:
        parser.error("choose one mode")
    checked = verify_inputs(restore=not args.verify_only)
    if args.verify_only or args.restore_only:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    log_dir = ROOT / "validation"
    log_dir.mkdir(exist_ok=True)
    commands = [[sys.executable, "new/run_validation.py", "--scope", "archive"]]
    commands += [[sys.executable, f"new/audit_section_5_{i}.py"] for i in range(1, 5)]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log = log_dir / f"check_{index}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Validation failed; see " + str(log))
        print("Passed: " + " ".join(command[1:]), flush=True)
    tests = json.loads((ROOT / "new/validation_tests_archive.json").read_text(encoding="utf-8"))
    if not tests["passed"] or tests["tests_run"] != 83 or tests["skipped"]:
        raise RuntimeError("Unexpected numerical-test count or status")
    report = {"passed": True, "immutable_inputs_verified": checked, "tests": tests,
              "section_5_saved_record_audits": 4, "retraining_performed": False,
              "full_rollouts_rerun": False, "manuscript_style_or_submission_audit": False,
              "additional_exclusions": ["test_claim_coverage.py: three historical-manuscript documentation tests; those manuscripts are not distributed"]}
    (log_dir / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("83 numerical/integrity tests and 4 saved-record audits passed; no retraining or full rollouts.")


if __name__ == "__main__":
    main()
