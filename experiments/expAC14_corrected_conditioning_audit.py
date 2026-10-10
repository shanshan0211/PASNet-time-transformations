"""AC14 correction of AC11 diagnostics from frozen stored Hessians.

AC11 is retained unchanged as invalidated diagnostic history.  Its stored
midpoint D2K, step size, and shell scale are sufficient to recompute the
similarity-transformed endpoint Jacobian without rerunning either model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC14")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
sys.path.insert(0, os.path.join(ROOT, "src"))

from monitor_conditioning import nondimensional_endpoint_diagnostics  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def canonical_protocol_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_protocol():
    protocol = load_json(PROTOCOL_PATH)
    actual = canonical_protocol_hash(protocol)
    if actual != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("AC14 protocol hash mismatch")
    for source in protocol["frozen_sources"]:
        path = os.path.join(ROOT, source["path"])
        if sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"sealed source changed: {source['path']}")
    return protocol


def corrected_sample(sample, semi_major_axis):
    conditioning = sample.get("conditioning")
    if conditioning is None:
        return {
            "sample_kind": sample["sample_kind"],
            "accepted_physical_step": sample["accepted_physical_step"],
            "conditioning_error": sample.get("conditioning_error"),
        }
    d2_k = torch.tensor(conditioning["D2zzK"], dtype=torch.float64)
    corrected = nondimensional_endpoint_diagnostics(
        d2_k, sample["step_size"], semi_major_axis
    )
    solver = sample.get("solver", {})
    return {
        "sample_kind": sample["sample_kind"],
        "accepted_physical_step": sample["accepted_physical_step"],
        "physical_time": sample["physical_time"],
        "step_size": sample["step_size"],
        "horizon_trim": sample["horizon_trim"],
        "source_midpoint": sample["midpoint"],
        "off_surface_defect_abs": conditioning["off_surface_defect_abs"],
        "corrected_conditioning": corrected,
        # These are intentionally distinct fields; neither is relabeled.
        "stopping_update_inf": solver.get("stopping_update_inf"),
        "returned_endpoint_residual_inf":
            solver.get("returned_endpoint_residual_inf"),
    }


def main():
    torch.set_num_threads(1)
    protocol = verify_protocol()
    source_path = os.path.join(ROOT, protocol["source"]["ac11_raw_path"])
    source = load_json(source_path)
    records = []
    for index, record in enumerate(source["records"], start=1):
        records.append(
            {
                "rollout_key": record["rollout_key"],
                "arm": record["arm"],
                "architecture": record["architecture"],
                "checkpoint_seed": record["checkpoint_seed"],
                "solver": record["solver"],
                "semi_major_axis": record["semi_major_axis"],
                "eccentricity": record["eccentricity"],
                "eventual_completion_status":
                    record["replay_completion_status"],
                "samples": [
                    corrected_sample(sample, record["semi_major_axis"])
                    for sample in record["samples"]
                ],
            }
        )
        if index % 100 == 0:
            print(f"corrected {index}/{len(source['records'])}", flush=True)
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "source_ac11_raw_sha256": sha256_file(source_path),
        "source_sample_count": sum(
            len(record["samples"]) for record in source["records"]
        ),
        "records": records,
    }
    temporary = RAW_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, RAW_PATH)
    print(sha256_file(RAW_PATH))


if __name__ == "__main__":
    main()
