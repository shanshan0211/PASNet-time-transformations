"""Run sealed AC17: fidelity-gated learned-H checkpoint factorial.

Hamiltonians are trained and judged on a preregistered held-out set before any
monitor is initialized.  Monitor replicates are nested within accepted
Hamiltonian checkpoints; rollouts are never treated as independent H
replicates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC17")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
H_AUDIT_PATH = os.path.join(OUT, "hamiltonian_audit.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import expAC12_learned_h_monitor_solver_factorial as ac12  # noqa: E402
from models.pasnet import PASNet  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


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
    if not os.path.isfile(PROTOCOL_PATH):
        raise RuntimeError("seal AC17 with register_expAC17.py before execution")
    protocol = load_json(PROTOCOL_PATH)
    if canonical_protocol_hash(protocol) != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("AC17 protocol hash mismatch")
    for source in protocol["frozen_sources"]:
        if sha256_file(os.path.join(ROOT, source["path"])) != source["sha256"]:
            raise RuntimeError(f"frozen source changed: {source['path']}")
    return protocol


def make_h_template():
    return PASNet(
        dim=2,
        hidden_H=128,
        n_hidden_H=3,
        hidden_g=64,
        n_hidden_g=2,
        g_max=2.0,
        h_features=ac12.h_features,
    ).H_net.double()


def states_and_fields(eccentricities, anomalies):
    states = np.concatenate(
        [
            ac12.kepler_state(1.0, eccentricity, anomalies)
            for eccentricity in eccentricities
        ],
        axis=0,
    )
    q, p = states[:, :2], states[:, 2:]
    radius = np.linalg.norm(q, axis=1, keepdims=True)
    gradients = np.concatenate([q / radius**3, p], axis=1)
    vector_fields = np.concatenate([p, -q / radius**3], axis=1)
    labels = np.repeat(np.asarray(eccentricities), len(anomalies))
    return states, gradients, vector_fields, labels


def h_input(states):
    return torch.cat([states, ac12.h_features(states)], dim=1)


def learned_gradient(model, states, *, create_graph):
    differentiable = states.detach().requires_grad_(True)
    values = model(h_input(differentiable)).squeeze(-1)
    return torch.autograd.grad(
        values.sum(), differentiable, create_graph=create_graph
    )[0]


def h_checkpoint_path(seed):
    return os.path.join(OUT, f"learned_h_seed{seed}.pt")


def train_h_candidate(protocol, seed):
    path = h_checkpoint_path(seed)
    if os.path.exists(path):
        package = torch.load(path, map_location="cpu", weights_only=True)
        if package["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError(f"H checkpoint protocol mismatch: {path}")
        return path, "loaded_existing"
    config = protocol["hamiltonian_training"]
    torch.manual_seed(seed)
    model = make_h_template()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    anomalies = (
        2.0 * math.pi
        * (np.arange(config["states_per_eccentricity"]) + 0.5)
        / config["states_per_eccentricity"]
    )
    states, gradients, _, _ = states_and_fields(
        config["training_eccentricities"], anomalies
    )
    states = torch.tensor(states, dtype=torch.float64)
    targets = torch.tensor(gradients, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed + 100000)
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        permutation = torch.randperm(len(states), generator=generator)
        for start in range(0, len(states), config["batch_size"]):
            indices = permutation[start : start + config["batch_size"]]
            prediction = learned_gradient(model, states[indices], create_graph=True)
            loss = torch.mean((prediction - targets[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"]
            )
            optimizer.step()
        scheduler.step()
    temporary = path + ".tmp"
    torch.save(
        {
            "H_net": model.state_dict(),
            "seed": seed,
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "wall_seconds": time.perf_counter() - started,
        },
        temporary,
    )
    os.replace(temporary, path)
    return path, "trained"


def audit_h_checkpoint(protocol, seed, path, training_status):
    gate = protocol["hamiltonian_fidelity_gate"]
    model = make_h_template()
    package = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(package["H_net"])
    model.eval()
    anomalies = (
        2.0 * math.pi
        * (np.arange(gate["held_out_states_per_eccentricity"]) + 0.1732050807568877)
        / gate["held_out_states_per_eccentricity"]
    )
    states_np, gradients_np, fields_np, labels = states_and_fields(
        gate["held_out_eccentricities"], anomalies
    )
    states = torch.tensor(states_np, dtype=torch.float64)
    predicted_gradient = learned_gradient(model, states, create_graph=False).detach().numpy()
    predicted_field = np.concatenate(
        [predicted_gradient[:, 2:], -predicted_gradient[:, :2]], axis=1
    )

    def metrics(predicted, target):
        errors = np.linalg.norm(predicted - target, axis=1)
        norms = np.linalg.norm(target, axis=1)
        relative = errors / np.maximum(norms, 1e-15)
        return {
            "normalized_rmse": float(
                np.sqrt(np.mean((predicted - target) ** 2))
                / np.sqrt(np.mean(target**2))
            ),
            "relative_error_p95": float(np.quantile(relative, 0.95)),
            "relative_error_max": float(np.max(relative)),
            "all_finite": bool(np.isfinite(predicted).all()),
            "by_eccentricity_normalized_rmse": {
                str(eccentricity): float(
                    np.sqrt(np.mean((predicted[labels == eccentricity]
                                     - target[labels == eccentricity]) ** 2))
                    / np.sqrt(np.mean(target[labels == eccentricity] ** 2))
                )
                for eccentricity in gate["held_out_eccentricities"]
            },
        }

    gradient_metrics = metrics(predicted_gradient, gradients_np)
    field_metrics = metrics(predicted_field, fields_np)
    reasons = []
    for name, values in [
        ("hamiltonian_gradient", gradient_metrics),
        ("vector_field", field_metrics),
    ]:
        if not values["all_finite"]:
            reasons.append(f"{name}:nonfinite")
        if values["normalized_rmse"] > gate["maximum_normalized_rmse"]:
            reasons.append(f"{name}:normalized_rmse")
        if values["relative_error_p95"] > gate["maximum_relative_error_p95"]:
            reasons.append(f"{name}:relative_error_p95")
        if values["relative_error_max"] > gate["maximum_relative_error_max"]:
            reasons.append(f"{name}:relative_error_max")
        if max(values["by_eccentricity_normalized_rmse"].values()) > gate[
            "maximum_stratum_normalized_rmse"
        ]:
            reasons.append(f"{name}:stratum_normalized_rmse")
    return {
        "h_seed": seed,
        "training_status": training_status,
        "checkpoint_path": os.path.relpath(path, ROOT),
        "checkpoint_sha256": sha256_file(path),
        "held_out_sample_count": len(states_np),
        "hamiltonian_gradient": gradient_metrics,
        "vector_field": field_metrics,
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }


def train_and_gate_hamiltonians(protocol):
    records = []
    for seed in protocol["hamiltonian_training"]["candidate_seeds"]:
        try:
            path, status = train_h_candidate(protocol, seed)
            record = audit_h_checkpoint(protocol, seed, path, status)
        except Exception as error:
            record = {
                "h_seed": seed,
                "training_status": "training_or_audit_failure",
                "accepted": False,
                "rejection_reasons": ["exception"],
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        records.append(record)
        atomic_json(
            H_AUDIT_PATH,
            {
                "schema_version": "1.0",
                "experiment": protocol["experiment"],
                "protocol_sha256": protocol["integrity"]["protocol_sha256"],
                "gate_was_applied_before_monitor_training": True,
                "records": records,
            },
        )
    accepted = [record for record in records if record["accepted"]]
    ensemble_qualified = len(accepted) >= protocol[
        "hamiltonian_fidelity_gate"
    ]["minimum_accepted_checkpoints"]
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_was_applied_before_monitor_training": True,
        "minimum_accepted_checkpoints": protocol["hamiltonian_fidelity_gate"][
            "minimum_accepted_checkpoints"
        ],
        "ensemble_qualified": ensemble_qualified,
        "decision": (
            "proceed_with_accepted_checkpoints"
            if ensemble_qualified
            else "no_checkpoint_ensemble_qualifies; skip_monitor_training_and_factorial"
        ),
        "records": records,
    }
    atomic_json(H_AUDIT_PATH, artifact)
    return artifact


def monitor_seed_entry(h_seed, replicate):
    base = 170000 + 100 * h_seed + replicate
    return {
        "checkpoint_seed": replicate,
        "sampling_seed": base + 1,
        "validation_seed": base + 2,
        "raw_initialization_seed": base + 3,
        "invariant_initialization_seed": base + 4,
    }


def train_monitors(protocol, audit):
    records = []
    if not audit["ensemble_qualified"]:
        artifact = {
            "schema_version": "1.0",
            "experiment": protocol["experiment"],
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "status": "skipped_no_qualified_h_ensemble",
            "records": records,
        }
        atomic_json(TRAINING_PATH, artifact)
        return artifact
    ac12.OUT = OUT
    original_factory = ac12.make_hamiltonian
    try:
        for h_record in audit["records"]:
            if not h_record["accepted"]:
                continue
            h_seed = h_record["h_seed"]
            checkpoint = os.path.join(ROOT, h_record["checkpoint_path"])
            ac12.make_hamiltonian = (
                lambda _protocol, _mode, path=checkpoint:
                ac12.HamiltonianMode("learned", path)
            )
            for architecture in protocol["monitor_training"]["architectures"]:
                for replicate in protocol["monitor_training"]["replicates"]:
                    seed_entry = monitor_seed_entry(h_seed, replicate)
                    mode = f"h{h_seed}"
                    try:
                        record = ac12.train_one(
                            protocol, mode, architecture, seed_entry
                        )
                        monitor = ac12.Monitor(architecture)
                        package = torch.load(
                            os.path.join(ROOT, record["checkpoint_path"]),
                            map_location="cpu",
                            weights_only=True,
                        )
                        monitor.load_state_dict(package["state_dict"])
                        record["symmetry"] = ac12.symmetry_diagnostics(monitor)
                        record["status"] = "trained_or_loaded"
                    except Exception as error:
                        record = {
                            "h_mode": mode,
                            "architecture": architecture,
                            "checkpoint_seed": replicate,
                            "status": "training_failure",
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    record["h_seed"] = h_seed
                    record["monitor_replicate"] = replicate
                    records.append(record)
                    atomic_json(
                        TRAINING_PATH,
                        {
                            "schema_version": "1.0",
                            "experiment": protocol["experiment"],
                            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
                            "status": "running",
                            "records": records,
                        },
                    )
    finally:
        ac12.make_hamiltonian = original_factory
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "status": "complete",
        "records": records,
    }
    atomic_json(TRAINING_PATH, artifact)
    return artifact


def unavailable_rollout(training_record, solver, scenario):
    return {
        "rollout_key": "|".join(map(str, [
            training_record["h_seed"], training_record["architecture"],
            training_record["monitor_replicate"], solver,
            scenario["eccentricity"], scenario["orientation_radians"],
            scenario["momentum_direction"],
        ])),
        "h_seed": training_record["h_seed"],
        "architecture": training_record["architecture"],
        "monitor_replicate": training_record["monitor_replicate"],
        "solver": solver,
        **scenario,
        "completion_status": "monitor_checkpoint_unavailable",
        "error": training_record.get("error"),
        "steps": 0,
        "wall_seconds": 0.0,
    }


def evaluate_all(protocol, audit, training):
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "records": [],
    }
    if os.path.exists(RAW_PATH):
        artifact = load_json(RAW_PATH)
    existing = {record["rollout_key"] for record in artifact["records"]}
    if not audit["ensemble_qualified"]:
        atomic_json(RAW_PATH, artifact)
        return artifact
    h_paths = {
        record["h_seed"]: os.path.join(ROOT, record["checkpoint_path"])
        for record in audit["records"] if record["accepted"]
    }
    scenarios = ac12.scenarios_from_protocol(protocol)
    for training_record in training["records"]:
        h_seed = training_record["h_seed"]
        architecture = training_record["architecture"]
        replicate = training_record["monitor_replicate"]
        for solver in protocol["evaluation"]["solvers"]:
            for scenario in scenarios:
                empty = unavailable_rollout(training_record, solver, scenario)
                if empty["rollout_key"] in existing:
                    continue
                if training_record["status"] == "training_failure":
                    record = empty
                else:
                    monitor = ac12.Monitor(architecture)
                    package = torch.load(
                        os.path.join(ROOT, training_record["checkpoint_path"]),
                        map_location="cpu", weights_only=True,
                    )
                    monitor.load_state_dict(package["state_dict"])
                    monitor.eval()
                    hamiltonian = ac12.HamiltonianMode("learned", h_paths[h_seed])
                    record = ac12.run_rollout(
                        monitor, hamiltonian, f"h{h_seed}", architecture,
                        replicate, solver, scenario, protocol["evaluation"],
                    )
                    record["h_seed"] = h_seed
                    record["monitor_replicate"] = replicate
                    record["rollout_key"] = empty["rollout_key"]
                artifact["records"].append(record)
                existing.add(record["rollout_key"])
                atomic_json(RAW_PATH, artifact)
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    return artifact


def summarize(protocol, audit, training, raw):
    accepted = [record for record in audit["records"] if record["accepted"]]
    expected = (
        len(accepted)
        * len(protocol["monitor_training"]["architectures"])
        * len(protocol["monitor_training"]["replicates"])
        * len(protocol["evaluation"]["solvers"])
        * len(ac12.scenarios_from_protocol(protocol))
        if audit["ensemble_qualified"] else 0
    )
    complete = [
        record for record in raw["records"]
        if record["completion_status"] == "complete"
    ]
    invariant = [
        record for record in training["records"]
        if record["architecture"] == "invariant"
        and record["status"] != "training_failure"
    ]
    acceptance = {
        "all_h_candidates_retained":
            len(audit["records"]) == len(
                protocol["hamiltonian_training"]["candidate_seeds"]
            ),
        "all_h_gate_decisions_transparent": all(
            "accepted" in record and "rejection_reasons" in record
            for record in audit["records"]
        ),
        "monitor_training_followed_gate": audit[
            "gate_was_applied_before_monitor_training"
        ],
        "registered_factorial_retained": len(raw["records"]) == expected,
        "completed_final_times_aligned": all(
            record["final_time_error"] < 1e-10 for record in complete
        ),
        "completed_endpoint_residuals_accepted": all(
            record["maximum_returned_endpoint_residual_inf"]
            < protocol["evaluation"]["acceptance_tolerance"]
            for record in complete
        ),
        "invariant_symmetry_valid": all(
            max(record["symmetry"].values()) < 1e-12 for record in invariant
        ),
        "source_hashes_valid": True,
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    statuses = {}
    for record in raw["records"]:
        status = record["completion_status"]
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "h_checkpoint_counts": {
            "candidate": len(audit["records"]),
            "accepted": len(accepted),
            "rejected": len(audit["records"]) - len(accepted),
            "ensemble_qualified": audit["ensemble_qualified"],
        },
        "counts": {
            "registered_rollouts_conditional_on_gate": expected,
            "attempted": len(raw["records"]),
            "by_status": statuses,
        },
        "acceptance": acceptance,
        "hamiltonian_audit_sha256": sha256_file(H_AUDIT_PATH),
        "training_sha256": sha256_file(TRAINING_PATH),
        "raw_results_sha256": sha256_file(RAW_PATH),
        "claim_limits": protocol["claim_limits"],
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    audit = train_and_gate_hamiltonians(protocol)
    training = train_monitors(protocol, audit)
    raw = evaluate_all(protocol, audit, training)
    summary = summarize(protocol, audit, training, raw)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("AC17 artifact gate failed")


if __name__ == "__main__":
    main()
