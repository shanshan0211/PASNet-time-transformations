"""Prospectively seal the AC19 periapsis contraction diagnostic and
step-size intervention before any full-grid computation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC19", "preregistration.json")


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
            "refusing registration because results/expAC19 already exists"
        )
    ac9_training = load_json(
        os.path.join(ROOT, "results", "expAC9", "training.json")
    )
    ac9_raw = load_json(
        os.path.join(ROOT, "results", "expAC9", "raw_results.json")
    )
    checkpoint_paths = [
        record["checkpoint_path"] for record in ac9_training["records"]
    ]
    if len(checkpoint_paths) != 10 or len(set(checkpoint_paths)) != 10:
        raise RuntimeError("expected 10 unique frozen AC9 checkpoints")
    failed_learned = sum(
        record["completion_status"] != "complete"
        and record["architecture"] in ("raw", "invariant")
        for record in ac9_raw["records"]
    )
    if len(ac9_raw["records"]) != 792:
        raise RuntimeError("expected 792 frozen AC9 rollouts")

    source_roles = [
        ("results/expAC9/preregistration.json", "frozen_scenario_protocol"),
        ("results/expAC9/raw_results.json", "frozen_completion_labels"),
        ("results/expAC9/training.json", "frozen_checkpoint_manifest"),
        (
            "experiments/expAC9_end_to_end_monitor_generalisation.py",
            "frozen_rollout_code_reused_for_phase_b",
        ),
        (
            "experiments/expAC19_periapsis_contraction_diagnostic.py",
            "runner",
        ),
        ("experiments/analyze_expAC19.py", "prospective_analysis"),
        ("src/systems.py", "exact_hamiltonian"),
        ("src/models/pasnet.py", "monitor_mlp_constructor"),
        (
            "theory/monitor_solver_conditioning.md",
            "fixed_point_linearisation_derivation",
        ),
    ]
    frozen_sources = [sealed_entry(path, role) for path, role in source_roles]
    frozen_sources.extend(
        sealed_entry(path, "frozen_ac9_checkpoint") for path in checkpoint_paths
    )

    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC19_periapsis_contraction_diagnostic",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_full_grid_diagnostic_or_intervention",
        "question":
            "Is the AC9 high-eccentricity completion boundary explained by "
            "a prospectively computable loss of fixed-point contraction at "
            "periapsis, caused by learned monitors that do not shrink with "
            "the radius below their training domain?",
        "design_pilot_disclosure":
            "Before sealing, the runner was exercised on AC9 checkpoints "
            "raw/invariant seed 7101 and invariant seed 7103 on the a=1 "
            "e=0.9, a=1 e=0.99 and a=0.75 e=0.99 forward orbits, and on a "
            "two-monitor two-orbit dry run, to verify that the batched "
            "Hessian, eigenvalue, and orbit construction code executes and "
            "to choose the safety target rho_target=0.5. No threshold, "
            "estimand, or figure choice was changed after inspecting the "
            "full grid, which had not been computed. A first registration "
            "(protocol 86a90ae7eb99954dca7761c1289f345004a5339b5cde000958d9f"
            "85b7c1f12a2) was voided after its phase-A run aborted with a "
            "filesystem PermissionError from the cloud-synchronised results "
            "folder after 40 of 396 orbit records; the runner's JSON writer "
            "gained a retry loop and the partial artifact was discarded. "
            "The diagnostic is deterministic, so the discarded records are "
            "reproduced identically by the resealed run.",
        "hypotheses": {
            "H1_extrapolation":
                "On e=0.99 orbits, learned monitors take values at periapsis "
                "that are several times the radius (g/r >> 1), whereas "
                "manual g=r has g/r=1 by construction.",
            "H2_contraction":
                "The maximum over the analytic reference orbit of the "
                "spectral radius of (dtau/2) J D2zzK at dtau=0.04 is at "
                "least 1 for learned monitors on e=0.99 orbits and below 1 "
                "for manual g=r on every registered orbit.",
            "H3_classification":
                "The prespecified rule 'rho_max >= 1 predicts solver "
                "failure' classifies the 792 frozen AC9 outcomes with "
                "accuracy above 0.9; misclassifications are reported.",
            "H4_location":
                "Failure radii of the failed AC9 rollouts concentrate within "
                "a factor 2 of the periapsis radius.",
            "H5_intervention":
                "Rerunning each failed learned AC9 rollout with dtau* = "
                "dtau*min(1, 0.5/rho_max) restores completion for the "
                "majority of rollouts, at a vector-field cost above manual "
                "g=r at dtau=0.04.",
            "scope":
                "The criterion is a linearised sufficient condition for "
                "local fixed-point contraction evaluated on the exact "
                "reference orbit; it is not a necessary condition and is "
                "not a theorem about the discrete trajectory.",
        },
        "distinction_from_ac14":
            "AC14 tested early-window conditioning diagnostics computed on "
            "accepted discrete iterates (steps 10, 50, 100) as an early "
            "warning and found no discrimination. AC19 tests a whole-orbit "
            "a-priori quantity computed from the monitor and the analytic "
            "reference orbit before any rollout. The estimands differ.",
        "frozen_sources": frozen_sources,
        "phase_a": {
            "monitors": "10 frozen AC9 checkpoints plus manual g=r",
            "orbit_families":
                "AC9 evaluation grid collapsed over initial anomaly: "
                "3 semi-major axes x 3 eccentricities x 2 orientations x "
                "2 momentum directions = 36 analytic orbits",
            "states_per_orbit": 720,
            "state_parameterisation":
                "uniform eccentric anomaly including exact periapsis E=0",
            "dtau": 0.04,
            "operator": "(dtau/2) J D2zzK(z) with K=g(H+p0), p0=-H(z0)",
            "hessian": "exact float64 autograd Hessian in z",
            "statistics": [
                "spectral radius (linearised fixed-point contraction)",
                "spectral norm (Banach sufficient bound, looser)",
                "monitor value and monitor/radius at periapsis",
                "physical-time fraction of orbit below training r_min",
            ],
            "training_radius_minimum": 0.1,
            "contraction_threshold": 1.0,
            "store_profiles": True,
            "profile_stride": 2,
            "failure_location":
                "map each failed AC9 rollout physical time to the analytic "
                "reference-orbit radius via Kepler's equation (bisection)",
        },
        "phase_b": {
            "population":
                "every failed learned AC9 rollout (raw and invariant)",
            "expected_failed_rollouts": failed_learned,
            "rule": "dtau* = dtau * min(1, rho_target / rho_max)",
            "rho_target": 0.5,
            "rho_target_rationale":
                "factor-2 safety margin below the linearised threshold "
                "because discrete midpoints deviate from the analytic orbit",
            "rollout_code": "frozen AC9 run_rollout with identical fixed-point "
                            "tolerance 1e-12, 100 iterations, matched final time",
            "maximum_steps": 2000000,
            "controls":
                "manual g=r AC9 dtau=0.04 rollouts of the same scenarios are "
                "the cost denominators; no new manual rollouts",
            "no_change_rule":
                "if rho_max <= rho_target the rule prescribes no change and "
                "the deterministic rerun is expected to fail again; such "
                "cases count against the rule",
        },
        "primary_estimands": [
            "rho_max by architecture and eccentricity",
            "g/r at periapsis by architecture and eccentricity",
            "confusion matrix of rho_max>=1 versus AC9 completion (792)",
            "failure radius over periapsis radius distribution",
            "phase-B completion count and cost ratio to manual g=r",
        ],
        "secondary_estimands": [
            "ROC area of rho_max for failure (descriptive)",
            "post-hoc maximum-accuracy threshold (labelled post hoc)",
            "spectral-norm Banach bound for comparison",
        ],
        "acceptance": {
            "expected_orbit_records": 11 * 36,
            "expected_failed_rollouts": failed_learned,
            "all_792_rollouts_classified": True,
            "all_intervention_outcomes_retained": True,
            "frozen_hashes_verified_before_run": True,
            "scientific_success_not_required": True,
            "claim_rule":
                "report all confusion-matrix cells, all misclassified "
                "rollouts, and all intervention failures regardless of sign",
        },
        "claim_limits": [
            "exact-H Kepler AC9 grid only",
            "linearised local criterion on the analytic orbit",
            "sufficient-condition logic: rho_max<1 does not guarantee "
            "completion",
            "step rule tested only on previously failed rollouts",
            "cost ratios are survivor-conditioned",
            "no learned Hamiltonian involved",
        ],
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
