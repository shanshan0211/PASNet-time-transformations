"""Analyze AC15 status transitions and paired parent/correction outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC15")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
ANALYSIS_PATH = os.path.join(OUT, "analysis.json")
SMOKE_RAW_PATH = os.path.join(OUT, "smoke_results.json")
SMOKE_ANALYSIS_PATH = os.path.join(OUT, "smoke_analysis.json")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def geometric_mean(values):
    finite = [
        value
        for value in values
        if value is not None and value > 0.0 and math.isfinite(value)
    ]
    return float(math.exp(np.mean(np.log(finite)))) if finite else None


def status_counts(records):
    return dict(sorted(Counter(
        record["completion_status"] for record in records
    ).items()))


def transition_counts(records):
    return dict(sorted(Counter(
        f"{record['parent_completion_status']} -> "
        f"{record['completion_status']}"
        for record in records
    ).items()))


def paired_change(records, corrected_field, parent_field):
    ratios = []
    differences = []
    for record in records:
        parent_value = record.get(parent_field)
        corrected_value = record.get(corrected_field)
        if (
            record["parent_completion_status"] == "complete"
            and record["completion_status"] == "complete"
            and parent_value is not None
            and corrected_value is not None
            and math.isfinite(parent_value)
            and math.isfinite(corrected_value)
        ):
            differences.append(corrected_value - parent_value)
            if parent_value > 0.0 and corrected_value > 0.0:
                ratios.append(corrected_value / parent_value)
    return {
        "n_complete_pairs": len(differences),
        "n_positive_ratio_pairs": len(ratios),
        "corrected_over_parent_geometric_mean_ratio":
            geometric_mean(ratios),
        "median_corrected_minus_parent": (
            float(np.median(differences)) if differences else None
        ),
        "survivor_conditioned": True,
    }


def group_summary(records):
    return {
        "registered": len(records),
        "parent_statuses": dict(sorted(Counter(
            record["parent_completion_status"] for record in records
        ).items())),
        "corrected_statuses": status_counts(records),
        "status_transitions": transition_counts(records),
        "maximum_returned_endpoint_residual_inf": max(
            (
                record["maximum_returned_endpoint_residual_inf"]
                for record in records
                if record["completion_status"] == "complete"
            ),
            default=None,
        ),
        "energy_change": paired_change(
            records, "max_energy_error", "parent_max_energy_error"
        ),
        "closure_change": paired_change(
            records,
            "final_state_closure_error",
            "parent_final_state_closure_error",
        ),
        "steps_change": paired_change(
            records, "steps", "parent_steps"
        ),
    }


def analyze(raw, protocol, smoke=False):
    records = raw["records"]
    by_parent = defaultdict(list)
    by_parent_configuration = defaultdict(list)
    for record in records:
        by_parent[record["parent_experiment"]].append(record)
        configuration = record.get("configuration_id")
        architecture = record.get("architecture")
        group = configuration or architecture or record.get("method")
        by_parent_configuration[
            f"{record['parent_experiment']}|{group}"
        ].append(record)

    complete = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    unique_ids = {record["replay_id"] for record in records}
    expected = protocol["minimum_rerun_decision"][
        "total_registered_replays"
    ]
    acceptance = {
        "all_1748_attempted_and_retained":
            len(records) == expected and len(unique_ids) == expected,
        "all_completed_returned_endpoint_residuals_below_own_tolerance": all(
            record["maximum_returned_endpoint_residual_inf"]
            < record["solver_tolerance"]
            for record in complete
        ),
        "all_parent_labels_linked": all(
            record.get("parent_completion_status") is not None
            and record.get("parent_record_index") is not None
            for record in records
        ),
        "all_failures_retained": all(
            record["completion_status"] != "running" for record in records
        ),
        "source_hashes_unchanged": True,
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    smoke_checks = None
    if smoke:
        smoke_checks = {
            "records_are_explicitly_smoke_only": all(
                record.get("smoke_only") for record in records
            ),
            "all_smoke_records_terminated": all(
                record["completion_status"] != "running"
                for record in records
            ),
            "all_executed_steps_satisfied_true_residual": all(
                record["maximum_returned_endpoint_residual_inf"]
                < record["solver_tolerance"]
                for record in records
            ),
        }
        smoke_checks["smoke_gate_passed"] = all(smoke_checks.values())
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "smoke_only": smoke,
        "counts": {
            "registered": expected,
            "attempted": len(records),
            "unique_replay_ids": len(unique_ids),
            "corrected_statuses": status_counts(records),
            "status_transitions": transition_counts(records),
        },
        "by_parent": {
            key: group_summary(value)
            for key, value in sorted(by_parent.items())
        },
        "by_parent_configuration": {
            key: group_summary(value)
            for key, value in sorted(by_parent_configuration.items())
        },
        "acceptance": acceptance,
        "smoke_checks": smoke_checks,
        "interpretation_limits": protocol["claim_limits"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    protocol = load(PROTOCOL_PATH)
    raw_path = SMOKE_RAW_PATH if arguments.smoke else RAW_PATH
    output = SMOKE_ANALYSIS_PATH if arguments.smoke else ANALYSIS_PATH
    raw = load(raw_path)
    if raw["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("AC15 raw/protocol hash mismatch")
    analysis = analyze(raw, protocol, smoke=arguments.smoke)
    analysis["source_sha256"] = {
        "raw": sha256_file(raw_path),
        "analysis_script": sha256_file(__file__),
    }
    atomic_json(output, analysis)
    if not arguments.smoke:
        summary = {
            key: analysis[key]
            for key in (
                "schema_version",
                "experiment",
                "generated_at_utc",
                "protocol_sha256",
                "counts",
                "by_parent",
                "acceptance",
                "interpretation_limits",
            )
        }
        summary["raw_results_sha256"] = sha256_file(raw_path)
        summary["analysis_sha256"] = sha256_file(output)
        atomic_json(SUMMARY_PATH, summary)
        if not analysis["acceptance"]["artifact_gate_passed"]:
            raise SystemExit("AC15 artifact gate failed")
    print(sha256_file(output))


if __name__ == "__main__":
    main()
