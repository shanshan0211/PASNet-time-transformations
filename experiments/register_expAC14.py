"""Prospectively seal the AC14 correction before corrected analysis."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC14", "preregistration.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sealed_entry(relative_path, role):
    absolute = os.path.join(ROOT, relative_path)
    if not os.path.isfile(absolute):
        raise FileNotFoundError(f"missing frozen {role}: {relative_path}")
    return {
        "path": relative_path,
        "sha256": sha256_file(absolute),
        "role": role,
    }


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")
    if os.path.exists(os.path.dirname(OUTPUT)):
        raise SystemExit(
            "refusing registration because results/expAC14 already exists"
        )

    checkpoint_paths = []
    for arm in ("AC8", "AC9"):
        training = load_json(
            os.path.join(ROOT, f"results/exp{arm}/training.json")
        )
        checkpoint_paths.extend(
            record["checkpoint_path"] for record in training["records"]
        )
    if len(checkpoint_paths) != 20 or len(set(checkpoint_paths)) != 20:
        raise RuntimeError("expected 20 unique frozen AC8/AC9 checkpoints")

    source_roles = [
        ("results/expAC11/raw_results.json", "stored_midpoint_hessians"),
        ("results/expAC11/preregistration.json", "invalidated_parent_protocol"),
        ("results/expAC8/preregistration.json", "frozen_scenario_protocol"),
        ("results/expAC9/preregistration.json", "frozen_scenario_protocol"),
        ("results/expAC8/raw_results.json", "frozen_completion_labels"),
        ("results/expAC9/raw_results.json", "frozen_completion_labels"),
        ("results/expAC8/training.json", "frozen_checkpoint_manifest"),
        ("results/expAC9/training.json", "frozen_checkpoint_manifest"),
        (
            "experiments/expAC14_corrected_conditioning_audit.py",
            "correction_runner",
        ),
        ("experiments/analyze_expAC14.py", "prospective_analysis"),
        ("src/monitor_conditioning.py", "corrected_conditioning_module"),
        (
            "theory/monitor_solver_conditioning.md",
            "endpoint_jacobian_derivation",
        ),
    ]
    frozen_sources = [
        sealed_entry(path, role) for path, role in source_roles
    ]
    frozen_sources.extend(
        sealed_entry(path, "frozen_parent_checkpoint")
        for path in checkpoint_paths
    )

    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC14_corrected_conditioning_audit",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_ac14_correction_or_analysis",
        "supersedes": {
            "experiment": "expAC11_monitor_solver_conditioning_audit",
            "disposition":
                "invalidated for nondimensional endpoint-Jacobian scaling "
                "and outcome-dependent pseudoreplicated figure estimands; "
                "artifacts preserved unchanged as diagnostic history",
        },
        "question":
            "Are correctly nondimensionalized accepted-iterate conditioning "
            "diagnostics associated with eventual AC8/AC9 rollout completion?",
        "hypotheses": {
            "fixed_point":
                "Eventual failures may be associated with larger corrected q.",
            "newton":
                "Eventual failures may be associated with smaller corrected "
                "sigma_min and larger corrected condition number.",
            "scope":
                "These observational associations do not identify a causal "
                "effect of conditioning on completion.",
        },
        "frozen_sources": frozen_sources,
        "source": {
            "ac11_raw_path": "results/expAC11/raw_results.json",
            "mathematical_equivalence":
                "At each frozen midpoint, stored D2zzK, h, and a are sufficient "
                "to evaluate S^-1 J D2zzK S and "
                "I-(h/2)S^-1 J D2zzK S exactly; model reevaluation is therefore "
                "mathematically equivalent but unnecessary.",
            "all_stored_samples_recomputed": True,
            "expected_source_samples": 71069,
        },
        "scaling_and_derivation": {
            "coordinate_map": "z=S z_tilde",
            "S": "diag(a,a,a^-1/2,a^-1/2)",
            "scaled_residual": "R_tilde(z_tilde)=S^-1 R(S z_tilde)",
            "chain_rule": "D R_tilde=S^-1 (D R) S",
            "endpoint_jacobian": "I-(h/2) S^-1 J D2zzK S",
            "fixed_point_derivative": "(h/2) S^-1 J D2zzK S",
            "correction":
                "No extra a^-3/2 multiplies h; coordinate similarity does "
                "not reparameterize fictitious time.",
            "norm": "spectral 2-norm",
        },
        "evaluation": {
            "expected_rollouts": 1056,
            "expected_AC8_rollouts": 264,
            "expected_AC9_rollouts": 792,
            "dtype": "float64",
            "device": "CPU",
            "no_training": True,
            "no_parent_artifact_writes": True,
        },
        "analysis": {
            "accepted_step_landmarks": [10, 50, 100],
            "landmark_selection":
                "fixed prospectively after verifying only that all 1056 "
                "historical rollouts reached accepted step 100; no corrected "
                "diagnostic values were inspected",
            "matched_step":
                "one accepted_stride diagnostic at the exact landmark per "
                "rollout, grouped by eventual completion",
            "matched_window":
                "per rollout through each landmark: maximum q, minimum "
                "sigma_min, and maximum condition number over the same "
                "accepted steps 10,20,...,landmark",
            "terminal_failed_iterates_excluded_from_matched_analyses": True,
            "terminal_iterate_sensitivity":
                "separate outcome-defined comparison: final failed iterate "
                "for failures versus final accepted sample for completions",
            "primary_figure_step": 100,
            "figure_observation_unit":
                "one exact-step accepted observation per rollout",
            "association_language_only": True,
            "no_causal_claims": True,
            "returned_endpoint_residual":
                "record and summarize separately from stopping update; ratio "
                "is explicitly secondary",
        },
        "acceptance": {
            "protocol_sealed_before_runner_or_analysis": True,
            "all_frozen_hashes_verified_before_correction": True,
            "all_1056_unique_rollouts_retained": True,
            "all_71069_stored_samples_recomputed": True,
            "matched_sets_have_1056_rollouts_at_each_landmark": True,
            "scientific_success_not_required": True,
        },
        "hardware_registration": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "integrity": {"protocol_sha256": None},
    }
    protocol["integrity"]["protocol_sha256"] = canonical_hash(protocol)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=False)
    with open(OUTPUT, "x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(protocol["integrity"]["protocol_sha256"])


if __name__ == "__main__":
    main()
