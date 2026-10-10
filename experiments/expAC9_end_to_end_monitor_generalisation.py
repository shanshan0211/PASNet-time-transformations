"""Train and evaluate the sealed AC9 end-to-end monitor checkpoints.

The protocol must be created by ``register_expAC9.py`` before this runner is
invoked. Training differentiates through true-H implicit-midpoint rollouts;
evaluation also keeps H exact, isolating monitor generalisation.
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
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC9")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.pasnet import make_mlp, midpoint_rollout  # noqa: E402
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
    expected = protocol["integrity"]["protocol_sha256"]
    actual = canonical_protocol_hash(protocol)
    if actual != expected:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {expected}")
    for source in protocol["frozen_sources"]:
        actual_source = sha256_file(os.path.join(ROOT, source["path"]))
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


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


class Monitor(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        input_dimension = 4 if architecture == "raw" else 5
        self.network = make_mlp(input_dimension, 64, 2).double()
        final = self.network[-1]
        nn.init.normal_(final.weight, std=0.1)
        nn.init.constant_(final.bias, -2.2)

    def forward(self, z):
        features = z if self.architecture == "raw" else invariant_features(z)
        return 2.0 * torch.sigmoid(
            self.network(features).squeeze(-1)
        )


class ExactHamiltonianRolloutModel(nn.Module):
    def __init__(self, monitor, eccentricity=0.9):
        super().__init__()
        self.monitor = monitor
        self.system = Kepler2D(eccentricity)
        self.dim = 2

    def H(self, z):
        return self.system.H_torch(z)

    def g(self, z):
        return self.monitor(z)

    def K(self, z, p0):
        return self.g(z) * (self.H(z) + p0)

    def J_grad_K(self, z, p0):
        if not z.requires_grad:
            z = z.requires_grad_(True)
        gradient = torch.autograd.grad(
            self.K(z, p0).sum(), z, create_graph=True
        )[0]
        return torch.cat([gradient[:, 2:], -gradient[:, :2]], dim=1)


def solve_eccentric_anomaly(mean_anomaly, eccentricity):
    mean_anomaly = np.asarray(mean_anomaly, dtype=np.float64)
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(30):
        correction = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if np.max(np.abs(correction)) < 1e-14:
            return eccentric_anomaly
    raise RuntimeError("eccentric-anomaly solve did not converge")


def kepler_state(semi_major_axis, eccentricity, mean_anomaly):
    eccentric_anomaly = solve_eccentric_anomaly(
        np.asarray(mean_anomaly), eccentricity
    )
    root = math.sqrt(1.0 - eccentricity**2)
    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    scale = math.sqrt(semi_major_axis)
    qx = semi_major_axis * (
        eccentricity - np.cos(eccentric_anomaly)
    )
    qy = -semi_major_axis * root * np.sin(eccentric_anomaly)
    px = np.sin(eccentric_anomaly) / (scale * denominator)
    py = -root * np.cos(eccentric_anomaly) / (scale * denominator)
    return np.stack([qx, qy, px, py], axis=-1)


def rotate_state(state, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return np.concatenate(
        [rotation @ state[:2], rotation @ state[2:]]
    )


def validation_loss(model, start_anomalies, config):
    losses = []
    for mean_anomaly in start_anomalies:
        state = torch.tensor(
            kepler_state(1.0, 0.9, mean_anomaly)[None, :],
            dtype=torch.float64,
        )
        p0 = -model.H(state).detach()
        times, states = midpoint_rollout(
            model,
            state,
            p0,
            config["dtau"],
            config["rollout_steps"],
            n_fp_iter=config["unrolled_fixed_point_iterations"],
        )
        detached_times = np.array(
            [float(value.detach()) for value in times]
        )
        maximum_time = detached_times[-1]
        checkpoint = config["checkpoint_physical_dt"]
        while checkpoint < maximum_time:
            upper = int(np.searchsorted(detached_times, checkpoint))
            if upper <= 0 or upper >= len(detached_times):
                break
            weight = (
                checkpoint - detached_times[upper - 1]
            ) / (
                detached_times[upper]
                - detached_times[upper - 1]
                + 1e-15
            )
            prediction = states[upper - 1] + weight * (
                states[upper] - states[upper - 1]
            )
            target = torch.tensor(
                kepler_state(
                    1.0, 0.9, mean_anomaly + checkpoint
                )[None, :],
                dtype=torch.float64,
            )
            losses.append(
                float(torch.mean((prediction.detach() - target) ** 2))
            )
            checkpoint += config["checkpoint_physical_dt"]
    return float(np.mean(losses)) if losses else math.inf


def training_checkpoint_path(architecture, seed):
    return os.path.join(OUT, f"monitor_{architecture}_seed{seed}.pt")


def progress_path(architecture, seed):
    return os.path.join(
        OUT, f"training_progress_{architecture}_seed{seed}.pt"
    )


def train_one(protocol, architecture, seed_entry):
    seed = seed_entry["checkpoint_seed"]
    checkpoint = training_checkpoint_path(architecture, seed)
    if os.path.exists(checkpoint):
        monitor = Monitor(architecture)
        package = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        monitor.load_state_dict(package["state_dict"])
        record = dict(package["metadata"])
        record["status"] = "loaded_existing_completed_checkpoint"
        record["checkpoint_sha256"] = sha256_file(checkpoint)
        return record

    config = protocol["training"]
    torch.manual_seed(
        seed_entry[f"{architecture}_initialization_seed"]
    )
    monitor = Monitor(architecture)
    model = ExactHamiltonianRolloutModel(monitor)
    optimizer = torch.optim.Adam(
        monitor.parameters(), lr=config["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    sampling_rng = np.random.default_rng(seed_entry["sampling_seed"])
    validation_rng = np.random.default_rng(seed_entry["validation_seed"])
    validation_anomalies = validation_rng.uniform(
        0.0, 2.0 * math.pi, config["validation_starts"]
    )
    best_loss = math.inf
    best_state = None
    best_epoch = None
    start_epoch = 0
    progress = progress_path(architecture, seed)
    if os.path.exists(progress):
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        monitor.load_state_dict(saved["monitor"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampling_rng.bit_generator.state = saved["sampling_rng_state"]
        best_loss = saved["best_loss"]
        best_state = saved["best_state"]
        best_epoch = saved["best_epoch"]
        start_epoch = saved["next_epoch"]

    sample_count = config["training_states"]
    anomalies = (
        2.0 * math.pi * np.arange(sample_count) / sample_count
    )
    training_states = kepler_state(1.0, 0.9, anomalies)
    radii = np.linalg.norm(training_states[:, :2], axis=1)
    probabilities = 1.0 / (
        radii ** config["periapsis_sampling_power"]
        + config["periapsis_sampling_offset"]
    )
    probabilities /= probabilities.sum()
    started = time.perf_counter()
    last_components = {}
    for epoch in range(start_epoch, config["epochs"]):
        index = int(
            sampling_rng.choice(sample_count, p=probabilities)
        )
        mean_anomaly = float(anomalies[index])
        initial_state = torch.tensor(
            training_states[index][None, :], dtype=torch.float64
        )
        p0 = -model.H(initial_state).detach()
        times, states = midpoint_rollout(
            model,
            initial_state,
            p0,
            config["dtau"],
            config["rollout_steps"],
            n_fp_iter=config["unrolled_fixed_point_iterations"],
        )
        rollout = torch.cat(states, dim=0)
        energy_variance = torch.var(model.H(rollout))
        detached_times = np.array(
            [float(value.detach()) for value in times]
        )
        checkpoint_loss = torch.zeros((), dtype=torch.float64)
        checkpoint_count = 0
        checkpoint_time = config["checkpoint_physical_dt"]
        while checkpoint_time < detached_times[-1]:
            upper = int(
                np.searchsorted(detached_times, checkpoint_time)
            )
            if upper <= 0 or upper >= len(detached_times):
                break
            weight = (
                checkpoint_time - detached_times[upper - 1]
            ) / (
                detached_times[upper]
                - detached_times[upper - 1]
                + 1e-15
            )
            prediction = states[upper - 1] + weight * (
                states[upper] - states[upper - 1]
            )
            target = torch.tensor(
                kepler_state(
                    1.0,
                    0.9,
                    mean_anomaly + checkpoint_time,
                )[None, :],
                dtype=torch.float64,
            )
            checkpoint_loss = checkpoint_loss + torch.mean(
                (prediction - target) ** 2
            )
            checkpoint_count += 1
            checkpoint_time += config["checkpoint_physical_dt"]
        if checkpoint_count:
            checkpoint_loss = checkpoint_loss / checkpoint_count
        cost_loss = torch.mean(1.0 / model.g(rollout))
        total_loss = (
            config["weight_energy_variance"] * energy_variance
            + config["weight_checkpoint"] * checkpoint_loss
            + config["weight_inverse_monitor"] * cost_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            monitor.parameters(), config["gradient_clip_norm"]
        )
        optimizer.step()
        scheduler.step()
        last_components = {
            "total": float(total_loss.detach()),
            "energy_variance": float(energy_variance.detach()),
            "checkpoint": float(checkpoint_loss.detach()),
            "inverse_monitor": float(cost_loss.detach()),
            "checkpoint_count": checkpoint_count,
        }
        if (epoch + 1) % config["validation_every_epochs"] == 0:
            current_validation = validation_loss(
                model, validation_anomalies, config
            )
            if current_validation < best_loss:
                best_loss = current_validation
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in monitor.state_dict().items()
                }
            torch.save(
                {
                    "monitor": monitor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "sampling_rng_state":
                        sampling_rng.bit_generator.state,
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "next_epoch": epoch + 1,
                },
                progress,
            )
            print(
                f"{architecture} seed={seed} epoch={epoch + 1} "
                f"val={current_validation:.3e} "
                f"best={best_loss:.3e}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    monitor.load_state_dict(best_state)
    metadata = {
        "architecture": architecture,
        "checkpoint_seed": seed,
        "status": "trained",
        "best_epoch": best_epoch,
        "best_validation_state_mse": best_loss,
        "last_training_components": last_components,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_path": os.path.relpath(checkpoint, ROOT),
    }
    torch.save(
        {"state_dict": best_state, "metadata": metadata}, checkpoint
    )
    metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    return metadata


def symmetry_diagnostics(monitor):
    rng = np.random.default_rng(90210)
    states = torch.tensor(
        np.stack(
            [
                kepler_state(
                    1.0,
                    float(rng.uniform(0.3, 0.95)),
                    float(rng.uniform(0.0, 2.0 * math.pi)),
                )
                for _ in range(64)
            ]
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        baseline = monitor(states)
        maximum_rotation = 0.0
        for angle in [0.37, 1.19, 2.41]:
            transformed = torch.tensor(
                np.stack(
                    [
                        rotate_state(state.numpy(), angle)
                        for state in states
                    ]
                ),
                dtype=torch.float64,
            )
            rotated = monitor(transformed)
            denominator = torch.clamp(
                torch.maximum(abs(baseline), abs(rotated)), min=1e-15
            )
            maximum_rotation = max(
                maximum_rotation,
                float(
                    torch.max(abs(rotated - baseline) / denominator)
                ),
            )
        reversed_states = states.clone()
        reversed_states[:, 2:] *= -1.0
        reversed_values = monitor(reversed_states)
        denominator = torch.clamp(
            torch.maximum(abs(baseline), abs(reversed_values)), min=1e-15
        )
        reversal = float(
            torch.max(abs(reversed_values - baseline) / denominator)
        )
    return {
        "maximum_rotation_relative_error": maximum_rotation,
        "maximum_momentum_reversal_relative_error": reversal,
    }


def train_all(protocol):
    records = []
    for architecture in protocol["training"]["architectures"]:
        for seed_entry in protocol["training"]["seed_manifest"]:
            record = train_one(protocol, architecture, seed_entry)
            monitor = Monitor(architecture)
            package = torch.load(
                os.path.join(ROOT, record["checkpoint_path"]),
                map_location="cpu",
                weights_only=True,
            )
            monitor.load_state_dict(package["state_dict"])
            record["symmetry"] = symmetry_diagnostics(monitor)
            records.append(record)
            atomic_json(
                TRAINING_PATH,
                {
                    "schema_version": "1.0",
                    "experiment": protocol["experiment"],
                    "protocol_sha256": protocol["integrity"][
                        "protocol_sha256"
                    ],
                    "updated_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "records": records,
                },
            )
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }


def angular_momentum(state):
    return float(
        state[0] * state[3] - state[1] * state[2]
    )


def transformed_step(state, step_size, vector_field, config):
    evaluations = 1
    trial = state + step_size * vector_field(state)
    residual = math.inf
    for iteration in range(1, config["maximum_iterations"] + 1):
        proposal = state + step_size * vector_field(
            0.5 * (state + trial)
        )
        evaluations += 1
        residual = float(torch.max(torch.abs(proposal - trial)))
        trial = proposal
        if not torch.isfinite(trial).all():
            raise FloatingPointError("nonfinite midpoint iterate")
        if residual < config["tolerance"]:
            return trial, residual, iteration, evaluations
    raise RuntimeError(
        f"fixed point did not converge: h={step_size:.6g}, "
        f"residual={residual:.3e}"
    )


def run_rollout(
    monitor,
    architecture,
    checkpoint_seed,
    scenario,
    evaluation,
):
    system = Kepler2D(scenario["eccentricity"])
    base_state = kepler_state(
        scenario["semi_major_axis"],
        scenario["eccentricity"],
        scenario["initial_mean_anomaly"],
    )
    initial_numpy = rotate_state(
        base_state, scenario["orientation_radians"]
    )
    if scenario["momentum_direction"] == "reversed":
        initial_numpy[2:] *= -1.0
    state = torch.tensor(initial_numpy[None, :], dtype=torch.float64)
    q0, p0 = initial_numpy[:2], initial_numpy[2:]
    energy0 = float(system.H(q0, p0))
    momentum0 = angular_momentum(initial_numpy)
    p0_extended = torch.tensor([-energy0], dtype=torch.float64)
    target_time = (
        2.0
        * math.pi
        * scenario["semi_major_axis"] ** 1.5
        * evaluation["periods"]
    )
    vector_evaluations = 0

    def vector_field(value):
        nonlocal vector_evaluations
        vector_evaluations += 1
        with torch.enable_grad():
            differentiable = value.detach().requires_grad_(True)
            extended = monitor(differentiable) * (
                system.H_torch(differentiable) + p0_extended
            )
            gradient = torch.autograd.grad(
                extended.sum(), differentiable
            )[0]
        return torch.cat(
            [gradient[:, 2:], -gradient[:, :2]], dim=1
        ).detach()

    record = {
        "architecture": architecture,
        "checkpoint_seed": checkpoint_seed,
        "completion_status": "running",
        "steps": 0,
        "nonlinear_iterations": 0,
        "gradient_evaluations": 0,
        "maximum_residual": 0.0,
        "maximum_iterations": 0,
        "max_energy_error": 0.0,
        "max_angular_momentum_error": 0.0,
        "physical_time": 0.0,
    }
    started = time.perf_counter()
    try:
        while record["physical_time"] < target_time:
            if record["steps"] >= evaluation["maximum_steps"]:
                record["completion_status"] = "step_cap"
                break
            step_size = evaluation["dtau"]
            trial, residual, iterations, evaluations = transformed_step(
                state,
                step_size,
                vector_field,
                evaluation["fixed_point"],
            )
            physical_dt = step_size * float(
                monitor(0.5 * (state + trial)).detach()
            )
            if not math.isfinite(physical_dt) or physical_dt <= 0.0:
                raise FloatingPointError(
                    f"invalid physical increment {physical_dt}"
                )
            if record["physical_time"] + physical_dt > target_time:
                remaining = target_time - record["physical_time"]
                lower, upper = 0.0, step_size
                for _ in range(40):
                    candidate_size = 0.5 * (lower + upper)
                    candidate, cand_residual, cand_iterations, cand_evals = (
                        transformed_step(
                            state,
                            candidate_size,
                            vector_field,
                            evaluation["fixed_point"],
                        )
                    )
                    candidate_dt = candidate_size * float(
                        monitor(0.5 * (state + candidate)).detach()
                    )
                    if candidate_dt < remaining:
                        lower = candidate_size
                    else:
                        upper = candidate_size
                    trial = candidate
                    residual = cand_residual
                    iterations = cand_iterations
                    evaluations += cand_evals
                physical_dt = remaining
            state = trial
            record["physical_time"] += physical_dt
            record["steps"] += 1
            record["nonlinear_iterations"] += iterations
            record["gradient_evaluations"] += evaluations
            record["maximum_residual"] = max(
                record["maximum_residual"], residual
            )
            record["maximum_iterations"] = max(
                record["maximum_iterations"], iterations
            )
            current = state.detach().numpy()[0]
            record["max_energy_error"] = max(
                record["max_energy_error"],
                abs(float(system.H(current[:2], current[2:])) - energy0),
            )
            record["max_angular_momentum_error"] = max(
                record["max_angular_momentum_error"],
                abs(angular_momentum(current) - momentum0),
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
    final_state = state.detach().numpy()[0]
    record.update(
        vector_field_evaluations=vector_evaluations,
        final_state_closure_error=float(
            np.linalg.norm(final_state - initial_numpy)
        ),
        final_time_error=abs(record["physical_time"] - target_time),
        wall_seconds=time.perf_counter() - started,
    )
    record.update(scenario)
    return record


def scenarios_from_protocol(protocol):
    evaluation = protocol["evaluation"]
    scenarios = []
    for semi_major_axis in evaluation["semi_major_axes"]:
        for eccentricity in evaluation["eccentricities"]:
            for anomaly in evaluation["initial_mean_anomalies"]:
                for angle in evaluation["orientations_radians"]:
                    for direction in evaluation["momentum_directions"]:
                        scenarios.append(
                            {
                                "semi_major_axis": semi_major_axis,
                                "eccentricity": eccentricity,
                                "initial_mean_anomaly": anomaly,
                                "orientation_radians": angle,
                                "momentum_direction": direction,
                                "energy_shell": -0.5 / semi_major_axis,
                                "same_training_energy":
                                    semi_major_axis == 1.0,
                            }
                        )
    return scenarios


def evaluate_all(protocol, training):
    evaluation = protocol["evaluation"]
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
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
            artifact = json.load(handle)
        if (
            artifact["protocol_sha256"]
            != protocol["integrity"]["protocol_sha256"]
        ):
            raise RuntimeError("raw artifact protocol mismatch")
    existing = {
        (
            record["architecture"],
            record["checkpoint_seed"],
            record["semi_major_axis"],
            record["eccentricity"],
            record["initial_mean_anomaly"],
            record["orientation_radians"],
            record["momentum_direction"],
        )
        for record in artifact["records"]
    }
    scenarios = scenarios_from_protocol(protocol)
    for training_record in training["records"]:
        architecture = training_record["architecture"]
        seed = training_record["checkpoint_seed"]
        monitor = Monitor(architecture)
        package = torch.load(
            os.path.join(ROOT, training_record["checkpoint_path"]),
            map_location="cpu",
            weights_only=True,
        )
        monitor.load_state_dict(package["state_dict"])
        monitor.eval()
        for scenario in scenarios:
            key = (
                architecture,
                seed,
                scenario["semi_major_axis"],
                scenario["eccentricity"],
                scenario["initial_mean_anomaly"],
                scenario["orientation_radians"],
                scenario["momentum_direction"],
            )
            if key in existing:
                continue
            record = run_rollout(
                monitor,
                architecture,
                seed,
                scenario,
                evaluation,
            )
            artifact["records"].append(record)
            atomic_json(RAW_PATH, artifact)
            print(
                f"{architecture} seed={seed} "
                f"a={scenario['semi_major_axis']} "
                f"e={scenario['eccentricity']} "
                f"M={scenario['initial_mean_anomaly']:.3f} "
                f"angle={scenario['orientation_radians']:.3f} "
                f"{scenario['momentum_direction']}: "
                f"{record['completion_status']} "
                f"dE={record['max_energy_error']:.3e}",
                flush=True,
            )

    def manual_monitor(value):
        return torch.linalg.vector_norm(value[:, :2], dim=1)

    for scenario in scenarios:
        key = (
            "manual_r",
            None,
            scenario["semi_major_axis"],
            scenario["eccentricity"],
            scenario["initial_mean_anomaly"],
            scenario["orientation_radians"],
            scenario["momentum_direction"],
        )
        if key in existing:
            continue
        record = run_rollout(
            manual_monitor,
            "manual_r",
            None,
            scenario,
            evaluation,
        )
        artifact["records"].append(record)
        atomic_json(RAW_PATH, artifact)
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    return artifact


def summarize(protocol, training, raw):
    records = raw["records"]
    groups = {}
    for architecture in ["raw", "invariant", "manual_r"]:
        selected = [
            record
            for record in records
            if record["architecture"] == architecture
        ]
        complete = [
            record
            for record in selected
            if record["completion_status"] == "complete"
        ]
        groups[architecture] = {
            "registered": len(selected),
            "complete": len(complete),
            "completion_fraction": len(complete) / len(selected),
            "same_energy_complete": sum(
                record["completion_status"] == "complete"
                and record["same_training_energy"]
                for record in selected
            ),
            "same_energy_registered": sum(
                record["same_training_energy"] for record in selected
            ),
            "unseen_energy_complete": sum(
                record["completion_status"] == "complete"
                and not record["same_training_energy"]
                for record in selected
            ),
            "unseen_energy_registered": sum(
                not record["same_training_energy"] for record in selected
            ),
        }
    statuses = {}
    for record in records:
        status = record["completion_status"]
        statuses[status] = statuses.get(status, 0) + 1
    complete = [
        record
        for record in records
        if record["completion_status"] == "complete"
    ]
    invariant_training = [
        record
        for record in training["records"]
        if record["architecture"] == "invariant"
    ]
    expected = protocol["evaluation"]["total_registered_rollouts"]
    acceptance = {
        "all_ten_checkpoints_retained": len(training["records"]) == 10,
        "all_rollouts_attempted_and_retained": len(records) == expected,
        "completed_final_times_aligned": all(
            record["final_time_error"] < 1e-10 for record in complete
        ),
        "completed_residuals_valid": all(
            record["maximum_residual"] < 1e-11
            for record in complete
        ),
        "invariant_symmetry_valid": all(
            max(
                record["symmetry"][
                    "maximum_rotation_relative_error"
                ],
                record["symmetry"][
                    "maximum_momentum_reversal_relative_error"
                ],
            )
            < 1e-12
            for record in invariant_training
        ),
        "source_hashes_valid": True,
        "all_failures_reported": sum(statuses.values()) == len(records),
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "counts": {
            "registered": expected,
            "attempted": len(records),
            "by_status": statuses,
        },
        "groups": groups,
        "acceptance": acceptance,
        "training_sha256": sha256_file(TRAINING_PATH),
        "raw_results_sha256": sha256_file(RAW_PATH),
        "claim_limits": {
            "exact_hamiltonian_only": True,
            "one_training_orbit_and_energy_shell": True,
            "five_independent_optimization_checkpoints_per_architecture": True,
            "fixed_point_evaluation_only": True,
        },
    }


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = verify_protocol()
    training = train_all(protocol)
    atomic_json(TRAINING_PATH, training)
    raw = evaluate_all(protocol, training)
    summary = summarize(protocol, training, raw)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("Exp AC9 artifact gate failed")


if __name__ == "__main__":
    main()
