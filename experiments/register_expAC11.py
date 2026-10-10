"""Preregister and seal the AC11 solver-conditioning replay."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC11", "preregistration.json")


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
        raise FileNotFoundError(f"required {role} is absent: {relative_path}")
    return {
        "path": relative_path,
        "sha256": sha256_file(absolute),
        "role": role,
    }


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")

    parent_protocols = {}
    checkpoint_paths = []
    for arm in ["AC8", "AC9"]:
        prefix = f"results/exp{arm}"
        protocol_path = f"{prefix}/preregistration.json"
        training_path = f"{prefix}/training.json"
        raw_path = f"{prefix}/raw_results.json"
        parent_protocols[arm] = {
            "protocol_path": protocol_path,
            "training_path": training_path,
            "raw_results_path": raw_path,
            "protocol": load_json(os.path.join(ROOT, protocol_path)),
        }
        training = load_json(os.path.join(ROOT, training_path))
        paths = [record["checkpoint_path"] for record in training["records"]]
        if len(paths) != 10 or len(set(paths)) != 10:
            raise RuntimeError(f"{arm} does not provide ten unique checkpoints")
        checkpoint_paths.extend(paths)
    if len(checkpoint_paths) != 20 or len(set(checkpoint_paths)) != 20:
        raise RuntimeError("expected exactly 20 unique frozen checkpoints")

    source_roles = [
        ("results/expAC8/preregistration.json", "parent_protocol"),
        ("results/expAC9/preregistration.json", "parent_protocol"),
        ("results/expAC8/training.json", "parent_checkpoint_manifest"),
        ("results/expAC9/training.json", "parent_checkpoint_manifest"),
        ("results/expAC8/raw_results.json", "parent_completion_labels"),
        ("results/expAC9/raw_results.json", "parent_completion_labels"),
        (
            "experiments/expAC8_monitor_architecture_solver_factorial.py",
            "parent_runner",
        ),
        (
            "experiments/expAC9_end_to_end_monitor_generalisation.py",
            "parent_runner",
        ),
        (
            "experiments/expAC11_monitor_solver_conditioning_audit.py",
            "ac11_runner",
        ),
        ("experiments/analyze_expAC11.py", "ac11_analysis"),
        ("src/implicit_midpoint_solvers.py", "corrected_solver_module"),
        ("src/monitor_conditioning.py", "conditioning_module"),
        ("src/systems.py", "parent_source_module"),
        ("src/models/pasnet.py", "parent_source_module"),
        (
            "theory/monitor_solver_conditioning.md",
            "conditioning_audit_design",
        ),
    ]
    frozen_sources = [
        sealed_entry(path, role) for path, role in source_roles
    ]
    frozen_sources.extend(
        sealed_entry(path, "parent_checkpoint") for path in checkpoint_paths
    )

    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC11_monitor_solver_conditioning_audit",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_ac11_replay",
        "question":
            "Do frozen AC8/AC9 completion failures concentrate at worse "
            "scaled implicit-midpoint conditioning diagnostics?",
        "hypotheses": {
            "fixed_point":
                "Failures are associated with larger scaled contraction q.",
            "newton":
                "Newton failures are associated with smaller scaled endpoint-"
                "Jacobian minimum singular values and larger condition numbers.",
            "stopping_semantics":
                "The historical stopping update and true residual are close "
                "but are not interchangeable quantities.",
        },
        "parents": parent_protocols,
        "frozen_sources": frozen_sources,
        "evaluation": {
            "mode": "read-only replay; no training and no parent writes",
            "AC8_arm_rollouts": 264,
            "AC9_arm_rollouts": 792,
            "total_registered_rollouts": 1056,
            "dtype": "float64",
            "device": "CPU",
            "torch_threads": 1,
            "parent_stopping_semantics_for_label_replay": True,
            "true_returned_endpoint_residual_recomputed_separately": True,
            "horizon_trimming":
                "reproduce each parent's 40-iteration final-step bisection "
                "and tag retained trim samples",
            "resume":
                "atomic raw_results.json replacement after every rollout; "
                "existing rollout keys are skipped only under matching protocol",
        },
        "sampling": {
            "accepted_physical_step_stride": 10,
            "stride_status": "fixed_preregistered",
            "retained":
                "every tenth accepted physical step, every accepted horizon "
                "trim, every failed step final iterate, and the preceding "
                "accepted sampled state embedded when feasible",
            "not_retained":
                "unsampled accepted steps and intermediate horizon-bisection "
                "candidate solves",
        },
        "scaling_and_norm": {
            "state_scaling": "S=diag(a,a,a^-1/2,a^-1/2)",
            "scaled_step": "h_tilde=h*a^-3/2",
            "scaled_operator": "S^-1 J D2zzK S",
            "matrix_norm": "spectral 2-norm",
            "vector_norm": "Euclidean 2-norm except solver infinity norms",
            "contraction_q": "(h_tilde/2)*||S^-1 J D2zzK S||_2",
            "scaled_endpoint_jacobian":
                "I-(h_tilde/2) S^-1 J D2zzK S",
            "condition_number": "sigma_max/sigma_min in scaled coordinates",
            "raw_coordinate_metrics": "secondary only",
        },
        "primary_estimands": [
            "parent/replay completion-label agreement by arm and solver",
            "failure-versus-completed sampled scaled contraction q",
            "failure-versus-completed sampled scaled endpoint sigma_min",
            "failure-versus-completed sampled scaled endpoint condition number",
            "returned endpoint residual divided by stopping update",
            "conditioning summaries stratified by arm, architecture, solver, "
            "eccentricity, and energy shell",
        ],
        "cluster_units_and_uncertainty": {
            "AC8_learned": "checkpoint seed within architecture",
            "AC9_learned": "checkpoint seed within architecture",
            "manual_controls": "deterministic scenario units; descriptive only",
            "repeated_steps":
                "samples are nested within rollout and never treated as "
                "independent rollout replicates",
            "bootstrap":
                "checkpoint-cluster bootstrap with 20000 draws where at least "
                "two checkpoint clusters are available",
            "bootstrap_seed": 11111,
        },
        "failure_reasons": [
            "fixed_point_iteration_cap",
            "newton_iteration_cap",
            "newton_monotone_line_search_failed",
            "singular_newton_jacobian",
            "nonfinite_iterate",
            "invalid_physical_time_increment",
            "maximum_steps",
            "unexpected_error",
        ],
        "claim_limits": [
            "observational conditioning audit; association is not causation",
            "replay does not intervene on conditioning or solver settings",
            "q is a local sufficient-condition proxy, not a necessary condition",
            "sampled every ten accepted steps, so unsampled extrema may be missed",
            "failed rollouts contribute a failure iterate, while completed "
            "rollouts contribute survivor-conditioned accepted samples",
            "scaled diagnostics depend on the preregistered coordinates and norm",
            "exact-H planar Kepler scope with twenty frozen learned checkpoints",
            "no claims about learned-H interaction or global trajectory stability",
        ],
        "acceptance": {
            "all_1056_parent_rollouts_attempted_and_retained": True,
            "all_parent_completion_labels_joined": True,
            "completion_label_mismatches_reported_not_discarded": True,
            "all_sealed_hashes_verified_before_replay": True,
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
        json.dump(protocol, handle, indent=2)
        handle.write("\n")
    print(protocol["integrity"]["protocol_sha256"])


if __name__ == "__main__":
    main()
