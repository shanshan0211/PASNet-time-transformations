"""Run the sealed AC8 monitor-architecture by nonlinear-solver factorial."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import socket
import sys
import time
import traceback
import tracemalloc
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC8")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.pasnet import PASNet, make_mlp  # noqa: E402
from systems import Kepler2D  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_protocol_hash(document):
    payload = json.loads(json.dumps(document))
    payload["integrity"].pop("protocol_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def verify_protocol():
    with open(PROTOCOL_PATH, encoding="utf-8") as handle:
        protocol = json.load(handle)
    actual = canonical_protocol_hash(protocol)
    expected = protocol["integrity"]["protocol_sha256"]
    if actual != expected:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {expected}")
    sources = [protocol["frozen_teacher"], *protocol["frozen_sources"]]
    for source in sources:
        path = os.path.join(ROOT, source["path"])
        actual_source = sha256_file(path)
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


def h_features(z):
    radius = torch.linalg.vector_norm(z[:, :2], dim=1, keepdim=True)
    return torch.cat([radius, 1.0 / radius], dim=1)


def load_teacher(protocol):
    teacher = PASNet(
        dim=2,
        hidden_H=128,
        n_hidden_H=3,
        hidden_g=64,
        n_hidden_g=2,
        g_max=2.0,
        h_features=h_features,
    )
    state = torch.load(
        os.path.join(ROOT, protocol["frozen_teacher"]["path"]),
        map_location="cpu",
        weights_only=True,
    )
    teacher.H_net.load_state_dict(state["H_net"])
    teacher.g_net.load_state_dict(state["g_net"])
    return teacher.double().eval()


def invariant_features(z):
    q, p = z[..., :2], z[..., 2:]
    radius = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    p_squared = torch.sum(p * p, dim=-1, keepdim=True)
    q_dot_p_squared = torch.sum(q * p, dim=-1, keepdim=True) ** 2
    angular_momentum_squared = (
        q[..., 0:1] * p[..., 1:2] - q[..., 1:2] * p[..., 0:1]
    ) ** 2
    return torch.cat(
        [
            radius,
            1.0 / radius,
            p_squared,
            q_dot_p_squared,
            angular_momentum_squared,
        ],
        dim=-1,
    )


class StudentMonitor(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        input_dimension = 4 if architecture == "raw" else 5
        self.network = make_mlp(input_dimension, 64, 2).double()

    def features(self, z):
        if self.architecture == "raw":
            return z
        return invariant_features(z)

    def forward(self, z):
        return 2.0 * torch.sigmoid(
            self.network(self.features(z)).squeeze(-1)
        )


def exact_training_states(n_samples=4096, eccentricity=0.9):
    mean_anomaly = (
        math.pi
        + 2.0 * math.pi * np.arange(n_samples, dtype=np.float64) / n_samples
    )
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(20):
        correction = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if np.max(np.abs(correction)) < 1e-14:
            break
    else:
        raise RuntimeError("eccentric-anomaly solve did not converge")
    root = math.sqrt(1.0 - eccentricity**2)
    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    qx = eccentricity - np.cos(eccentric_anomaly)
    qy = -root * np.sin(eccentric_anomaly)
    px = np.sin(eccentric_anomaly) / denominator
    py = -root * np.cos(eccentric_anomaly) / denominator
    return torch.as_tensor(
        np.column_stack([qx, qy, px, py]), dtype=torch.float64
    )


def symmetry_diagnostics(model, states):
    angles = [0.37, 1.19, 2.41]
    baseline = model(states)
    maximum_relative_error = 0.0
    for angle in angles:
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = torch.tensor(
            [[cosine, -sine], [sine, cosine]], dtype=states.dtype
        )
        transformed = torch.cat(
            [states[:, :2] @ rotation.T, states[:, 2:] @ rotation.T], dim=1
        )
        rotated = model(transformed)
        denominator = torch.maximum(
            torch.maximum(torch.abs(baseline), torch.abs(rotated)),
            torch.tensor(1e-15),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            float(torch.max(torch.abs(rotated - baseline) / denominator)),
        )
    reversed_states = states.clone()
    reversed_states[:, 2:] *= -1.0
    reversed_values = model(reversed_states)
    denominator = torch.maximum(
        torch.maximum(torch.abs(baseline), torch.abs(reversed_values)),
        torch.tensor(1e-15),
    )
    reversal_error = float(
        torch.max(torch.abs(reversed_values - baseline) / denominator)
    )
    return {
        "maximum_rotation_relative_error": maximum_relative_error,
        "momentum_reversal_relative_error": reversal_error,
    }


def train_students(protocol, teacher):
    states = exact_training_states()
    with torch.no_grad():
        targets = teacher.g(states)
    validation_mask = torch.arange(len(states)) % 4 == 3
    train_states, train_targets = states[~validation_mask], targets[
        ~validation_mask
    ]
    validation_states, validation_targets = (
        states[validation_mask],
        targets[validation_mask],
    )
    records = []
    config = protocol["training"]
    for architecture in config["architectures"]:
        for seed in config["seeds"]:
            checkpoint_path = os.path.join(
                OUT, f"monitor_{architecture}_seed{seed}.pt"
            )
            if os.path.exists(checkpoint_path):
                model = StudentMonitor(architecture)
                model.load_state_dict(
                    torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                )
                with torch.no_grad():
                    train_mse = float(
                        torch.mean((model(train_states) - train_targets) ** 2)
                    )
                    validation_mse = float(
                        torch.mean(
                            (model(validation_states) - validation_targets) ** 2
                        )
                    )
                records.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "status": "loaded_existing",
                        "train_mse": train_mse,
                        "validation_mse": validation_mse,
                        "checkpoint_path": os.path.relpath(
                            checkpoint_path, ROOT
                        ),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "symmetry": symmetry_diagnostics(
                            model, validation_states[:128]
                        ),
                    }
                )
                continue
            torch.manual_seed(seed)
            model = StudentMonitor(architecture)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config["learning_rate"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
            generator = torch.Generator().manual_seed(seed + 100000)
            best_loss, best_state, best_epoch = math.inf, None, None
            started = time.perf_counter()
            for epoch in range(config["epochs"]):
                permutation = torch.randperm(
                    len(train_states), generator=generator
                )
                for start in range(0, len(train_states), config["batch_size"]):
                    indices = permutation[start : start + config["batch_size"]]
                    prediction = model(train_states[indices])
                    loss = torch.mean(
                        (prediction - train_targets[indices]) ** 2
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if (epoch + 1) % 20 == 0:
                    with torch.no_grad():
                        validation_loss = float(
                            torch.mean(
                                (
                                    model(validation_states)
                                    - validation_targets
                                )
                                ** 2
                            )
                        )
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        best_epoch = epoch + 1
                        best_state = {
                            key: value.detach().clone()
                            for key, value in model.state_dict().items()
                        }
            if best_state is None:
                raise RuntimeError("no student checkpoint was selected")
            model.load_state_dict(best_state)
            torch.save(best_state, checkpoint_path)
            with torch.no_grad():
                train_mse = float(
                    torch.mean((model(train_states) - train_targets) ** 2)
                )
                validation_mse = float(
                    torch.mean(
                        (model(validation_states) - validation_targets) ** 2
                    )
                )
            record = {
                "architecture": architecture,
                "seed": seed,
                "status": "trained",
                "best_epoch": best_epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "wall_seconds": time.perf_counter() - started,
                "checkpoint_path": os.path.relpath(checkpoint_path, ROOT),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "symmetry": symmetry_diagnostics(
                    model, validation_states[:128]
                ),
            }
            records.append(record)
            print(
                f"trained {architecture} seed={seed} "
                f"val={validation_mse:.3e}",
                flush=True,
            )
            atomic_json(
                TRAINING_PATH,
                {
                    "schema_version": "1.0",
                    "protocol_sha256": protocol["integrity"][
                        "protocol_sha256"
                    ],
                    "records": records,
                },
            )
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "n_training_states": len(train_states),
        "n_validation_states": len(validation_states),
        "teacher_target_range": [
            float(torch.min(targets)),
            float(torch.max(targets)),
        ],
        "records": records,
    }
    atomic_json(TRAINING_PATH, artifact)
    return artifact


def load_student(architecture, seed):
    model = StudentMonitor(architecture)
    model.load_state_dict(
        torch.load(
            os.path.join(OUT, f"monitor_{architecture}_seed{seed}.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    return model.eval()


def rotate_state(q, p, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return rotation @ q, rotation @ p


def angular_momentum(z):
    return float(z[0] * z[3] - z[1] * z[2])


class FieldCounter:
    def __init__(self, system, monitor, p0_extended):
        self.system = system
        self.monitor = monitor
        self.p0_extended = p0_extended
        self.vector_evaluations = 0
        self.jacobian_evaluations = 0

    def vector_field(self, state, create_graph=False):
        self.vector_evaluations += 1
        differentiable = state
        if not differentiable.requires_grad:
            differentiable = differentiable.detach().requires_grad_(True)
        extended = self.monitor(differentiable) * (
            self.system.H_torch(differentiable) + self.p0_extended
        )
        gradient = torch.autograd.grad(
            extended.sum(), differentiable, create_graph=create_graph
        )[0]
        field = torch.cat([gradient[:, 2:], -gradient[:, :2]], dim=1)
        return field if create_graph else field.detach()

    def residual_and_jacobian(self, trial, state, step_size):
        self.jacobian_evaluations += 1
        flat_state = state.reshape(-1)

        def residual_function(flat_trial):
            shaped = flat_trial.reshape_as(state)
            midpoint = 0.5 * (state + shaped)
            return (
                shaped
                - state
                - step_size * self.vector_field(midpoint, create_graph=True)
            ).reshape(-1)

        flat_trial = trial.detach().reshape(-1).requires_grad_(True)
        residual = residual_function(flat_trial)
        jacobian = torch.autograd.functional.jacobian(
            residual_function, flat_trial, vectorize=True
        )
        return residual.detach(), jacobian.detach(), flat_state


def fixed_point_step(state, step_size, field, config):
    trial = state + step_size * field.vector_field(state)
    iterations = 0
    residual_norm = math.inf
    for iterations in range(1, config["maximum_iterations"] + 1):
        proposal = state + step_size * field.vector_field(
            0.5 * (state + trial)
        )
        residual_norm = float(torch.max(torch.abs(proposal - trial)))
        trial = proposal
        if not torch.isfinite(trial).all():
            raise FloatingPointError("nonfinite fixed-point iterate")
        if residual_norm < config["tolerance"]:
            return trial, residual_norm, iterations, 0
    raise RuntimeError(
        f"fixed point did not converge: h={step_size:.6g}, "
        f"residual={residual_norm:.3e}"
    )


def damped_newton_step(state, step_size, field, config):
    trial = state + step_size * field.vector_field(state)
    total_backtracks = 0
    for iteration in range(1, config["maximum_iterations"] + 1):
        residual, jacobian, _ = field.residual_and_jacobian(
            trial, state, step_size
        )
        residual_norm = float(torch.max(torch.abs(residual)))
        if residual_norm < config["tolerance"]:
            return trial, residual_norm, iteration, total_backtracks
        try:
            direction = torch.linalg.solve(jacobian, -residual).reshape_as(
                trial
            )
        except torch.linalg.LinAlgError as error:
            raise RuntimeError(f"singular Newton Jacobian: {error}") from error
        damping = 1.0
        accepted = False
        for _ in range(config["maximum_backtracks"] + 1):
            candidate = trial + damping * direction
            candidate_residual = (
                candidate
                - state
                - step_size
                * field.vector_field(0.5 * (state + candidate))
            )
            candidate_norm = float(
                torch.max(torch.abs(candidate_residual))
            )
            if math.isfinite(candidate_norm) and candidate_norm < residual_norm:
                trial = candidate.detach()
                accepted = True
                break
            damping *= 0.5
            total_backtracks += 1
        if not accepted or damping < config["minimum_damping"]:
            raise RuntimeError(
                f"Newton line search failed: h={step_size:.6g}, "
                f"residual={residual_norm:.3e}"
            )
    raise RuntimeError(
        f"Newton did not converge: h={step_size:.6g}, "
        f"residual={residual_norm:.3e}"
    )


def solve_step(state, step_size, field, solver, evaluation):
    if solver == "fixed_point":
        return fixed_point_step(
            state, step_size, field, evaluation["fixed_point"]
        )
    return damped_newton_step(
        state, step_size, field, evaluation["damped_newton"]
    )


def run_rollout(
    system,
    monitor,
    q0,
    p0,
    architecture,
    seed,
    solver,
    evaluation,
):
    initial_numpy = np.concatenate([q0, p0])
    state = torch.tensor(initial_numpy[None, :], dtype=torch.float64)
    energy0 = float(system.H(q0, p0))
    momentum0 = angular_momentum(initial_numpy)
    p0_extended = torch.tensor([-energy0], dtype=torch.float64)
    field = FieldCounter(system, monitor, p0_extended)
    target_time = evaluation["physical_horizon"]
    record = {
        "completion_status": "running",
        "architecture": architecture,
        "seed": seed,
        "solver": solver,
        "steps": 0,
        "nonlinear_iterations": 0,
        "backtracks": 0,
        "maximum_nonlinear_residual": 0.0,
        "maximum_nonlinear_iterations": 0,
        "max_energy_error": 0.0,
        "max_angular_momentum_error": 0.0,
        "physical_time": 0.0,
    }
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    started = time.perf_counter()
    try:
        while record["physical_time"] < target_time:
            if record["steps"] >= evaluation["maximum_steps_per_rollout"]:
                record["completion_status"] = "step_cap"
                break
            step_size = evaluation["dtau"]
            trial, residual, iterations, backtracks = solve_step(
                state, step_size, field, solver, evaluation
            )
            midpoint = 0.5 * (state + trial)
            physical_dt = step_size * float(monitor(midpoint))
            if not math.isfinite(physical_dt) or physical_dt <= 0.0:
                raise FloatingPointError(
                    f"invalid physical time increment {physical_dt}"
                )
            if record["physical_time"] + physical_dt > target_time:
                remaining = target_time - record["physical_time"]
                lower, upper = 0.0, step_size
                for _ in range(40):
                    candidate_size = 0.5 * (lower + upper)
                    candidate, cand_residual, cand_iterations, cand_backs = (
                        solve_step(
                            state,
                            candidate_size,
                            field,
                            solver,
                            evaluation,
                        )
                    )
                    candidate_dt = candidate_size * float(
                        monitor(0.5 * (state + candidate))
                    )
                    if candidate_dt < remaining:
                        lower = candidate_size
                    else:
                        upper = candidate_size
                    trial, residual = candidate, cand_residual
                    iterations, backtracks = cand_iterations, cand_backs
                physical_dt = remaining
            state = trial
            record["physical_time"] += physical_dt
            record["steps"] += 1
            record["nonlinear_iterations"] += iterations
            record["backtracks"] += backtracks
            record["maximum_nonlinear_residual"] = max(
                record["maximum_nonlinear_residual"], residual
            )
            record["maximum_nonlinear_iterations"] = max(
                record["maximum_nonlinear_iterations"], iterations
            )
            state_numpy = state.detach().numpy()[0]
            q, p = state_numpy[:2], state_numpy[2:]
            record["max_energy_error"] = max(
                record["max_energy_error"],
                abs(float(system.H(q, p)) - energy0),
            )
            record["max_angular_momentum_error"] = max(
                record["max_angular_momentum_error"],
                abs(angular_momentum(state_numpy) - momentum0),
            )
        if record["completion_status"] == "running":
            record["completion_status"] = "complete"
    except RuntimeError as error:
        record["completion_status"] = "solver_failure"
        record["error"] = str(error)
    except FloatingPointError as error:
        record["completion_status"] = "nonfinite"
        record["error"] = str(error)
    except Exception as error:
        record["completion_status"] = "unexpected_error"
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final_numpy = state.detach().numpy()[0]
    record.update(
        vector_field_evaluations=field.vector_evaluations,
        jacobian_evaluations=field.jacobian_evaluations,
        final_state_closure_error=float(
            np.linalg.norm(final_numpy - initial_numpy)
        ),
        final_time_error=abs(record["physical_time"] - target_time),
        wall_seconds=time.perf_counter() - started,
        python_tracemalloc_peak_bytes=python_peak,
        process_peak_rss_kib_before=rss_start,
        process_peak_rss_kib_after=resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    )
    return record


def geometric_mean(values):
    finite = [
        value
        for value in values
        if value is not None and value > 0.0 and math.isfinite(value)
    ]
    if not finite:
        return None
    return float(math.exp(np.mean(np.log(finite))))


def summarize(records, training, protocol):
    statuses = {}
    groups = {}
    for record in records:
        status = record["completion_status"]
        statuses[status] = statuses.get(status, 0) + 1
        key = f"{record['architecture']}|{record['solver']}"
        group = groups.setdefault(
            key,
            {
                "registered": 0,
                "complete": 0,
                "energy_errors": [],
                "closure_errors": [],
                "steps": [],
                "wall_seconds": [],
            },
        )
        group["registered"] += 1
        if status == "complete":
            group["complete"] += 1
            group["energy_errors"].append(record["max_energy_error"])
            group["closure_errors"].append(
                record["final_state_closure_error"]
            )
            group["steps"].append(record["steps"])
            group["wall_seconds"].append(record["wall_seconds"])
    for group in groups.values():
        group["completion_fraction"] = (
            group["complete"] / group["registered"]
        )
        for metric in [
            "energy_errors",
            "closure_errors",
            "steps",
            "wall_seconds",
        ]:
            values = group.pop(metric)
            group[f"{metric}_geometric_mean"] = geometric_mean(values)
            group[f"{metric}_median"] = (
                float(np.median(values)) if values else None
            )

    index = {
        (
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
            record["seed"],
            record["architecture"],
            record["solver"],
        ): record
        for record in records
        if record["architecture"] != "manual_r"
    }
    recovery = {"raw": 0, "invariant": 0}
    fixed_failures = {"raw": 0, "invariant": 0}
    architecture_pairs = {"fixed_point": [], "damped_newton": []}
    evaluation = protocol["evaluation"]
    for eccentricity in evaluation["eccentricities"]:
        for angle in evaluation["orientations_radians"]:
            for direction in evaluation["momentum_directions"]:
                for seed in protocol["training"]["seeds"]:
                    for architecture in ["raw", "invariant"]:
                        fixed = index[
                            (
                                eccentricity,
                                angle,
                                direction,
                                seed,
                                architecture,
                                "fixed_point",
                            )
                        ]
                        newton = index[
                            (
                                eccentricity,
                                angle,
                                direction,
                                seed,
                                architecture,
                                "damped_newton",
                            )
                        ]
                        if fixed["completion_status"] != "complete":
                            fixed_failures[architecture] += 1
                            if newton["completion_status"] == "complete":
                                recovery[architecture] += 1
                    for solver in ["fixed_point", "damped_newton"]:
                        raw = index[
                            (
                                eccentricity,
                                angle,
                                direction,
                                seed,
                                "raw",
                                solver,
                            )
                        ]
                        invariant = index[
                            (
                                eccentricity,
                                angle,
                                direction,
                                seed,
                                "invariant",
                                solver,
                            )
                        ]
                        if (
                            raw["completion_status"] == "complete"
                            and invariant["completion_status"] == "complete"
                        ):
                            architecture_pairs[solver].append(
                                {
                                    "energy_invariant_over_raw":
                                        invariant["max_energy_error"]
                                        / max(raw["max_energy_error"], 1e-300),
                                    "closure_invariant_over_raw":
                                        invariant[
                                            "final_state_closure_error"
                                        ]
                                        / max(
                                            raw[
                                                "final_state_closure_error"
                                            ],
                                            1e-300,
                                        ),
                                }
                            )
    invariant_symmetry = [
        record["symmetry"]
        for record in training["records"]
        if record["architecture"] == "invariant"
    ]
    complete = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    acceptance = {
        "all_ten_students_retained": len(training["records"]) == 10,
        "all_264_rollouts_attempted_and_retained": len(records)
        == evaluation["total_registered_rollouts"],
        "completed_final_times_aligned": all(
            record["final_time_error"] < 1e-10 for record in complete
        ),
        "completed_solver_residuals_valid": all(
            record["maximum_nonlinear_residual"] < 1e-11
            for record in complete
        ),
        "invariant_symmetry_valid": all(
            max(
                diagnostic["maximum_rotation_relative_error"],
                diagnostic["momentum_reversal_relative_error"],
            )
            < 1e-12
            for diagnostic in invariant_symmetry
        ),
        "source_hashes_valid": True,
        "all_failures_reported": sum(statuses.values()) == len(records),
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    paired_summary = {}
    for solver, pairs in architecture_pairs.items():
        paired_summary[solver] = {
            "n_complete_pairs": len(pairs),
            "energy_invariant_over_raw_geometric_mean": geometric_mean(
                [pair["energy_invariant_over_raw"] for pair in pairs]
            ),
            "closure_invariant_over_raw_geometric_mean": geometric_mean(
                [pair["closure_invariant_over_raw"] for pair in pairs]
            ),
        }
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "counts": {
            "registered": evaluation["total_registered_rollouts"],
            "attempted": len(records),
            "by_status": statuses,
        },
        "groups": groups,
        "fixed_point_failures": fixed_failures,
        "newton_recoveries_of_fixed_point_failures": recovery,
        "paired_invariant_over_raw": paired_summary,
        "acceptance": acceptance,
        "memory_measurement_limit":
            "tracemalloc excludes Torch native allocations; process peak RSS "
            "is cumulative and is reported only as an implementation "
            "diagnostic, not a per-rollout estimand",
        "claim_limits": {
            "exact_hamiltonian_only": True,
            "teacher_distillation_not_end_to_end_monitor_retraining": True,
            "one_canonical_training_orbit": True,
            "five_optimization_seeds": True,
        },
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    teacher = load_teacher(protocol)
    training = train_students(protocol, teacher)
    evaluation = protocol["evaluation"]
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "records": [],
    }
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing["protocol_sha256"] != artifact["protocol_sha256"]:
            raise RuntimeError("existing raw artifact has another protocol")
        artifact = existing
    completed_keys = {
        (
            record["eccentricity"],
            record["orientation_radians"],
            record["momentum_direction"],
            record["architecture"],
            record["seed"],
            record["solver"],
        )
        for record in artifact["records"]
    }

    scenarios = []
    for eccentricity in evaluation["eccentricities"]:
        system = Kepler2D(eccentricity)
        base_q, base_p = system.initial_condition()
        for angle in evaluation["orientations_radians"]:
            rotated_q, rotated_p = rotate_state(base_q, base_p, angle)
            for direction in evaluation["momentum_directions"]:
                momentum = (
                    rotated_p if direction == "forward" else -rotated_p
                )
                scenarios.append(
                    (eccentricity, system, angle, direction, rotated_q, momentum)
                )

    for architecture in ["raw", "invariant"]:
        for seed in protocol["training"]["seeds"]:
            monitor = load_student(architecture, seed)
            for eccentricity, system, angle, direction, q0, p0 in scenarios:
                for solver in evaluation["solvers"]:
                    key = (
                        eccentricity,
                        angle,
                        direction,
                        architecture,
                        seed,
                        solver,
                    )
                    if key in completed_keys:
                        continue
                    print(
                        f"{architecture} seed={seed} e={eccentricity} "
                        f"a={angle:.3f} {direction} {solver}",
                        flush=True,
                    )
                    record = run_rollout(
                        system,
                        monitor,
                        q0,
                        p0,
                        architecture,
                        seed,
                        solver,
                        evaluation,
                    )
                    record.update(
                        eccentricity=eccentricity,
                        orientation_radians=angle,
                        momentum_direction=direction,
                    )
                    artifact["records"].append(record)
                    atomic_json(RAW_PATH, artifact)
                    print(
                        f"  {record['completion_status']} "
                        f"dE={record['max_energy_error']:.3e} "
                        f"steps={record['steps']} "
                        f"wall={record['wall_seconds']:.2f}s",
                        flush=True,
                    )

    def manual_monitor(state):
        return torch.linalg.vector_norm(state[:, :2], dim=1)

    for eccentricity, system, angle, direction, q0, p0 in scenarios:
        for solver in evaluation["solvers"]:
            key = (
                eccentricity,
                angle,
                direction,
                "manual_r",
                None,
                solver,
            )
            if key in completed_keys:
                continue
            print(
                f"manual_r e={eccentricity} a={angle:.3f} "
                f"{direction} {solver}",
                flush=True,
            )
            record = run_rollout(
                system,
                manual_monitor,
                q0,
                p0,
                "manual_r",
                None,
                solver,
                evaluation,
            )
            record.update(
                eccentricity=eccentricity,
                orientation_radians=angle,
                momentum_direction=direction,
            )
            artifact["records"].append(record)
            atomic_json(RAW_PATH, artifact)
            print(
                f"  {record['completion_status']} "
                f"dE={record['max_energy_error']:.3e} "
                f"steps={record['steps']} "
                f"wall={record['wall_seconds']:.2f}s",
                flush=True,
            )

    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    summary = summarize(artifact["records"], training, protocol)
    summary["training_sha256"] = sha256_file(TRAINING_PATH)
    summary["raw_results_sha256"] = sha256_file(RAW_PATH)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("Exp AC8 artifact gate failed")


if __name__ == "__main__":
    main()
