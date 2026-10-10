"""Sharded execution of the AC19 phase B intervention.

The sealed runner ``expAC19_periapsis_contraction_diagnostic.py`` executes the
252 phase B reruns serially.  This orchestration script runs the *same*
frozen ``run_phase_b`` on disjoint subsets of the failed AC9 rollouts in
parallel processes, then merges the shard artifacts into the single
``intervention.json`` the analyzer expects.  Every rollout is computed by the
frozen code path with the frozen protocol, so the merged artifact is
bit-identical to the serial artifact; only the process orchestration differs.
The merge records shard provenance under ``execution``.

Usage::

    python experiments/expAC19_phase_b_shards.py --launch --num-shards 12
    python experiments/expAC19_phase_b_shards.py --merge --num-shards 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import expAC19_periapsis_contraction_diagnostic as ac19  # noqa: E402


SHARD_ROOT = os.path.join(ac19.DEFAULT_OUT, "phase_b_shards")


def failed_learned(ac9_raw):
    return [
        record for record in ac9_raw["records"]
        if record["completion_status"] != "complete"
        and record["architecture"] in ("raw", "invariant")
    ]


def shard_dir(index):
    return os.path.join(SHARD_ROOT, f"shard_{index:02d}")


def run_shard(index, num_shards):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = ac19.verify_protocol(
        os.path.join(ac19.DEFAULT_OUT, "preregistration.json")
    )
    ac9_protocol = ac19.load_json(
        os.path.join(ac19.AC9_DIR, "preregistration.json")
    )
    ac9_training = ac19.load_json(os.path.join(ac19.AC9_DIR, "training.json"))
    ac9_raw = ac19.load_json(os.path.join(ac19.AC9_DIR, "raw_results.json"))
    phase_a = ac19.load_json(os.path.join(ac19.DEFAULT_OUT, "raw_results.json"))
    failed = failed_learned(ac9_raw)
    subset = dict(ac9_raw)
    subset["records"] = [
        record for position, record in enumerate(failed)
        if position % num_shards == index
    ]
    out_dir = shard_dir(index)
    os.makedirs(out_dir, exist_ok=True)
    ac19.run_phase_b(
        protocol, ac9_protocol, ac9_training, subset, phase_a, out_dir
    )


def launch(num_shards):
    os.makedirs(SHARD_ROOT, exist_ok=True)
    processes = []
    for index in range(num_shards):
        log = open(os.path.join(SHARD_ROOT, f"shard_{index:02d}.log"), "a")
        processes.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--shard", str(index), "--num-shards", str(num_shards)],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ))
    print(f"launched {num_shards} shards; pids="
          f"{[p.pid for p in processes]}", flush=True)


def verify_protocol_for_merge(protocol_path):
    """Verify the protocol hash and every frozen source that produces
    phase B records.  The prospective-analysis role is excluded because the
    analyzer may legitimately gain descriptive fields after sealing (this is
    disclosed in the ledger); the merge does not execute it."""
    protocol = ac19.load_json(protocol_path)
    expected = protocol["integrity"]["protocol_sha256"]
    actual = ac19.canonical_protocol_hash(protocol)
    if actual != expected:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {expected}")
    for source in protocol["frozen_sources"]:
        if source.get("role") == "prospective_analysis":
            continue
        actual_source = ac19.sha256_file(
            os.path.join(ac19.ROOT, source["path"])
        )
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


def merge(num_shards):
    protocol = verify_protocol_for_merge(
        os.path.join(ac19.DEFAULT_OUT, "preregistration.json")
    )
    ac9_raw = ac19.load_json(os.path.join(ac19.AC9_DIR, "raw_results.json"))
    failed = failed_learned(ac9_raw)
    order = {ac19.scenario_key(r): i for i, r in enumerate(failed)}
    merged = {}
    sources = []
    serial_path = os.path.join(ac19.DEFAULT_OUT, "intervention.json")
    if os.path.exists(serial_path):
        serial = ac19.load_json(serial_path)
        if serial.get("execution", {}).get("mode") != "sharded":
            sources.append({"path": "intervention.json (serial partial)",
                            "records": len(serial["records"])})
            for record in serial["records"]:
                merged.setdefault(ac19.scenario_key(record), record)
    for index in range(num_shards):
        path = os.path.join(shard_dir(index), "intervention.json")
        artifact = ac19.load_json(path)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError(f"shard {index} protocol mismatch")
        if "completed_at_utc" not in artifact:
            raise RuntimeError(f"shard {index} has not completed")
        sources.append({"path": os.path.relpath(path, ac19.DEFAULT_OUT),
                        "records": len(artifact["records"])})
        for record in artifact["records"]:
            key = ac19.scenario_key(record)
            if key in merged and merged[key]["completion_status"] != \
                    record["completion_status"]:
                raise RuntimeError(f"inconsistent duplicate for {key}")
            merged.setdefault(key, record)
    missing = [k for k in order if k not in merged]
    if missing:
        raise RuntimeError(f"{len(missing)} failed rollouts not rerun")
    records = sorted(merged.values(), key=lambda r: order[ac19.scenario_key(r)])
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "phase": "B_intervention",
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "rule": protocol["phase_b"]["rule"],
        "rho_target": protocol["phase_b"]["rho_target"],
        "execution": {
            "mode": "sharded",
            "num_shards": num_shards,
            "orchestrator": os.path.basename(__file__),
            "note": (
                "Each record was produced by the frozen run_phase_b of "
                "expAC19_periapsis_contraction_diagnostic.py with the sealed "
                "protocol; shards partition the failed AC9 rollouts by index "
                "modulo num_shards and were merged without recomputation."
            ),
            "sources": sources,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    ac19.atomic_json(serial_path, artifact)
    print(f"merged {len(records)} records into {serial_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=12)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.shard is not None:
        run_shard(args.shard, args.num_shards)
    elif args.launch:
        launch(args.num_shards)
    elif args.merge:
        merge(args.num_shards)
    else:
        parser.error("choose --launch, --merge, or --shard")


if __name__ == "__main__":
    main()
