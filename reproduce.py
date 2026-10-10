"""Restore losslessly compressed evidence, verify inputs and validate saved records."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

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
    checked = 0
    for record in manifest["files"]:
        if record.get("mutable_report"):
            continue
        target = local_path(record["path"])
        if "stored_path" in record:
            packed = local_path(record["stored_path"])
            if sha(packed) != record["stored_sha256"]:
                raise ValueError("Compressed input mismatch: " + record["stored_path"])
            if restore and not target.exists():
                temp = target.with_suffix(target.suffix + ".restore-tmp")
                try:
                    with gzip.open(packed, "rb") as inp, temp.open("wb") as out:
                        shutil.copyfileobj(inp, out)
                    if temp.stat().st_size != record["bytes"] or sha(temp) != record["sha256"]:
                        raise ValueError("Restored input mismatch: " + record["path"])
                    temp.replace(target)
                finally:
                    temp.unlink(missing_ok=True)
            if not restore and not target.exists():
                checked += 1
                continue
        if target.stat().st_size != record["bytes"] or sha(target) != record["sha256"]:
            raise ValueError("Input mismatch: " + record["path"])
        checked += 1
    print(f"Verified {checked} immutable inputs", flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check packaged hashes without extracting large records")
    parser.add_argument("--restore-only", action="store_true", help="Restore and verify evidence; do not run tests")
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
              "additional_exclusions": [
                  "test_claim_coverage.py: three historical-manuscript documentation tests; those manuscripts are not distributed"]}
    (log_dir / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("83 numerical/integrity tests and 4 saved-record audits passed; no retraining or full rollouts.")


if __name__ == "__main__":
    main()
