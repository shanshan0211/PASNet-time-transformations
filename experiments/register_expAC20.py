"""Prospectively seal AC20 (deployment-domain training and Sundman-factored
monitors) before any checkpoint is trained inside ``results/expAC20``."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTPUT = os.path.join(ROOT, "results", "expAC20", "preregistration.json")
ARMS = ("invariant_e099", "sundman_e09", "sundman_e099")


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
    return {"path": relative_path, "sha256": sha256_file(absolute),
            "role": role}


def main():
    if os.path.exists(OUTPUT):
        raise SystemExit(f"refusing to overwrite sealed protocol: {OUTPUT}")
    if os.path.exists(os.path.dirname(OUTPUT)):
        raise SystemExit(
            "refusing registration because results/expAC20 already exists"
        )
    ac9_protocol = load_json(
        os.path.join(ROOT, "results", "expAC9", "preregistration.json")
    )
    ac9_raw = load_json(
        os.path.join(ROOT, "results", "expAC9", "raw_results.json")
    )
    if len(ac9_raw["records"]) != 792:
        raise RuntimeError("expected 792 frozen AC9 rollouts")
    training = json.loads(json.dumps(ac9_protocol["training"]))
    if len(training["seed_manifest"]) != 5:
        raise RuntimeError("expected the five-entry AC9 seed manifest")
    baseline_invariant = [
        r for r in ac9_raw["records"] if r["architecture"] == "invariant"
    ]
    baseline_e099 = sum(
        r["completion_status"] == "complete"
        for r in baseline_invariant if r["eccentricity"] == 0.99
    )

    source_roles = [
        ("results/expAC9/preregistration.json",
         "frozen_scenario_protocol_and_training_recipe"),
        ("results/expAC9/raw_results.json", "frozen_baseline_outcomes"),
        ("results/expAC9/training.json", "frozen_baseline_checkpoint_manifest"),
        ("experiments/expAC9_end_to_end_monitor_generalisation.py",
         "frozen_training_and_rollout_code_reused"),
        ("experiments/expAC19_periapsis_contraction_diagnostic.py",
         "frozen_contraction_diagnostic_reused"),
        ("experiments/expAC20_deployment_domain_training.py", "runner"),
        ("experiments/analyze_expAC20.py", "prospective_analysis"),
        ("src/systems.py", "exact_hamiltonian"),
        ("src/models/pasnet.py", "monitor_mlp_constructor_and_integrator"),
    ]
    frozen_sources = [sealed_entry(path, role) for path, role in source_roles]

    protocol = {
        "schema_version": "1.0",
        "experiment": "expAC20_deployment_domain_training",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_before_any_checkpoint_training",
        "question":
            "Does moving the deployment radii into the training domain, or "
            "factoring the Sundman radius out of the learned monitor, remove "
            "the AC9 high-eccentricity completion boundary that AC19 traced "
            "to a plateaued monitor and lost fixed-point contraction below "
            "the training radius?",
        "design_pilot_disclosure":
            "Before sealing, the runner's --pilot mode trained each arm for "
            "5 epochs on seed 7101 into a temporary directory outside the "
            "repository to time the recipe (about 5 s per epoch including "
            "validation) and to confirm that the unrolled loss is finite on "
            "the e=0.99 training orbit at the cold start. Observed pilot "
            "values after 5 epochs (a=1, e=0.99 orbit): invariant_e099 "
            "rho_max 6.8 and g/r 18.4 at periapsis; sundman_e09 rho_max "
            "0.20 and g/r 1.01; sundman_e099 rho_max 0.18 and g/r 0.91. "
            "These are cold-start values of a 5-epoch pilot, are not "
            "estimands, and no estimand, threshold, or arm was changed "
            "after seeing them. The runner contains a non-finite-loss "
            "guard (skip and count the epoch) that was not triggered in the "
            "pilot; its counts are reported per checkpoint. A second dry run "
            "exercised the complete job pipeline (train 2 epochs -> "
            "diagnostic -> 72 rollouts) for sundman_e09 seed 7101 into the "
            "same temporary directory to verify that the part artifacts are "
            "written and timestamped; that near-cold-start monitor completed "
            "72/72 rollouts. Both pilot directories are outside the "
            "repository and are not results.",
        "arms": {
            "invariant_e099": {
                "architecture": "invariant",
                "description":
                    "AC9 invariant-feature monitor, AC9 recipe unchanged "
                    "except the training orbit a=1, e=0.99 (periapsis 0.01, "
                    "so all AC9 grid periapsides r_p >= 0.0075 lie within "
                    "a factor 1.33 of the training minimum radius)",
                "training_eccentricity": 0.99,
                "training_radius_minimum": 0.01,
            },
            "sundman_e09": {
                "architecture": "sundman_invariant",
                "description":
                    "g = r * 2*sigmoid(NN(phi)) with the AC9 invariant "
                    "features phi, final bias 0 (cold start g=r), trained on "
                    "the AC9 orbit a=1, e=0.9 (unchanged training domain)",
                "training_eccentricity": 0.9,
                "training_radius_minimum": 0.1,
            },
            "sundman_e099": {
                "architecture": "sundman_invariant",
                "description":
                    "Sundman-factored monitor trained on a=1, e=0.99",
                "training_eccentricity": 0.99,
                "training_radius_minimum": 0.01,
            },
        },
        "paired_design":
            "Each arm uses the AC9 seed manifest (checkpoint seeds "
            "7101-7105 with the AC9 invariant initialisation, sampling and "
            "validation seeds), so every new checkpoint is paired with the "
            "AC9 invariant checkpoint of the same seed; contrasts are "
            "reported as paired per-checkpoint completion differences with "
            "a checkpoint-level bootstrap.",
        "training": training,
        "training_recipe_note":
            "Identical to AC9 (exact Hamiltonian, 600 epochs, Adam 2e-4 "
            "cosine, 100 unrolled midpoint steps at dtau=0.08 with 4 unrolled "
            "fixed-point iterations, checkpoint/energy-variance/inverse-"
            "monitor losses, periapsis-weighted sampling, validation-MSE "
            "selection) except: (i) the training orbit eccentricity is an "
            "arm parameter; (ii) Kepler states are produced by the AC19 "
            "bisection solver because the AC9 Newton solver does not converge "
            "for batched e=0.99 anomalies; (iii) a non-finite-loss guard "
            "skips and counts the epoch.",
        "diagnostic": {
            "statistic":
                "AC19 prospective contraction statistic rho_max = max over "
                "the analytic reference orbit of the spectral radius of "
                "(dtau/2) J D2zzK with K=g(H+p0), computed for every "
                "checkpoint on the 36 AC9 reference orbits BEFORE any "
                "rollout of that checkpoint (timestamped)",
            "states_per_orbit": 720,
            "dtau": 0.04,
            "training_radius_minimum": 0.1,
            "contraction_threshold": 1.0,
            "profile_stride": 2,
            "prediction_rule": "rho_max >= 1 predicts solver failure",
        },
        "evaluation":
            "the 72 registered AC9 scenarios per checkpoint with the frozen "
            "AC9 run_rollout (dtau=0.04, fixed-point tolerance 1e-12, "
            "100 iterations, max_steps 100000, one period); 15 checkpoints x "
            "72 = 1080 rollouts",
        "baselines": {
            "ac9_invariant":
                "frozen AC9 invariant checkpoints, 360 rollouts, "
                f"{sum(r['completion_status'] == 'complete' for r in baseline_invariant)}/360 "
                f"complete, {baseline_e099}/120 at e=0.99",
            "ac9_raw": "frozen AC9 raw checkpoints, 360 rollouts",
            "ac9_manual_r":
                "frozen AC9 manual g=r rollouts, 72/72 complete; accuracy "
                "and cost denominators",
            "no_new_baseline_rollouts": True,
        },
        "hypotheses": {
            "H1_deployment_domain":
                "invariant_e099 completes more of the 120 e=0.99 rollouts "
                "than the paired AC9 invariant checkpoints (0/120); the "
                "e<=0.9 completion is reported regardless of direction.",
            "H2_sundman_factor":
                "both Sundman-factored arms have rho_max < 1 on all 36 "
                "reference orbits for every checkpoint and complete all "
                "120 e=0.99 rollouts.",
            "H3_prospective_prediction":
                "the AC19 rule rho_max >= 1 computed before the rollouts "
                "predicts the 1080 new outcomes with accuracy above 0.9; "
                "all misclassifications are reported.",
            "H4_accuracy_cost":
                "descriptive: survivor-conditioned closure-error, "
                "energy-error and vector-field-evaluation ratios of each arm "
                "to manual g=r on the same scenario, by eccentricity.",
            "scope":
                "Kepler with exact H only; the AC9 grid is the deployment "
                "domain; the outcome of the arms is unknown at sealing and "
                "negative results are retained.",
        },
        "primary_estimands": [
            "completion count by arm x eccentricity (out of 120 each)",
            "paired per-checkpoint completion difference arm minus AC9 "
            "invariant, all scenarios and e=0.99, with 5000-draw "
            "checkpoint bootstrap",
            "rho_max and g/r at periapsis by arm x eccentricity",
            "confusion matrix of the prospective rule on the 1080 rollouts",
            "closure-error and cost ratios to manual g=r (survivor-"
            "conditioned)",
        ],
        "secondary_estimands": [
            "best validation state MSE, best epoch, non-finite epochs "
            "skipped per checkpoint",
            "rotation and momentum-reversal symmetry errors of the monitors",
        ],
        "acceptance": {
            "expected_checkpoints": 15,
            "expected_orbit_records": 15 * 36,
            "expected_rollouts": 15 * 72,
            "diagnostic_precedes_rollouts": True,
            "all_outcomes_retained": True,
            "frozen_hashes_verified_before_run": True,
            "scientific_success_not_required": True,
            "claim_rule":
                "report all completion counts, all confusion-matrix cells, "
                "and all training pathologies regardless of sign",
        },
        "claim_limits": [
            "exact-H Kepler, AC9 grid only",
            "five checkpoints per arm; checkpoint-level uncertainty only",
            "training orbit changed in two arms, so accuracy differences "
            "conflate architecture and training domain unless the paired "
            "arms are compared",
            "cost and accuracy ratios are survivor-conditioned",
            "no learned Hamiltonian involved",
        ],
        "execution":
            "one job per checkpoint (train -> prospective diagnostic -> "
            "rollouts), per-job artifacts under results/expAC20/parts, "
            "launched as parallel single-threaded processes; the number of "
            "processes does not affect any result",
        "hardware_registration": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "frozen_sources": frozen_sources,
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
