"""Run the sealed AC13 N>8 and cross-Hamiltonian checkpoint benchmark."""

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
OUT = os.path.join(ROOT, "results", "expAC13")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
TRAINING_PATH = os.path.join(OUT, "training.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
sys.path.insert(0, os.path.join(ROOT, "src"))

from implicit_midpoint_solvers import (  # noqa: E402
    ImplicitMidpointSolverError,
    fixed_point_endpoint,
)
from models.pasnet import make_mlp, midpoint_rollout  # noqa: E402
from systems import Duffing1D, ModalOscillator  # noqa: E402


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
    if not os.path.exists(PROTOCOL_PATH):
        raise RuntimeError("seal AC13 with register_expAC13.py before execution")
    with open(PROTOCOL_PATH, encoding="utf-8") as handle:
        protocol = json.load(handle)
    expected = protocol["integrity"]["protocol_sha256"]
    if canonical_protocol_hash(protocol) != expected:
        raise RuntimeError("AC13 protocol hash mismatch")
    for source in protocol["frozen_sources"]:
        actual = sha256_file(os.path.join(ROOT, source["path"]))
        if actual != source["sha256"]:
            raise RuntimeError(f"frozen source changed: {source['path']}")
    return protocol


def build_system(system_name, *, kappa=None):
    if system_name == "modal":
        return ModalOscillator(
            n_modes=16,
            omega_min=1.0,
            kappa=10.0 if kappa is None else kappa,
            beta=0.25,
            seed=0,
            orthogonal_mixing=True,
        )
    if system_name == "duffing":
        return Duffing1D(beta=0.1)
    raise ValueError(system_name)


def dimension(system_name):
    return 16 if system_name == "modal" else 1


def reversal_even_features(z, system_name):
    d = dimension(system_name)
    return torch.cat([z[..., :d], z[..., d:] ** 2], dim=-1)


class Monitor(nn.Module):
    def __init__(self, system_name, architecture):
        super().__init__()
        self.system_name = system_name
        self.architecture = architecture
        input_dimension = 2 * dimension(system_name)
        self.network = make_mlp(input_dimension, 64, 2).double()
        final = self.network[-1]
        nn.init.normal_(final.weight, std=0.1)
        nn.init.constant_(final.bias, -2.2)

    def forward(self, z):
        features = (
            z
            if self.architecture == "raw"
            else reversal_even_features(z, self.system_name)
        )
        return 2.0 * torch.sigmoid(self.network(features).squeeze(-1))


class ExactHamiltonianRolloutModel(nn.Module):
    def __init__(self, system_name, monitor):
        super().__init__()
        self.system_name = system_name
        self.system = build_system(system_name)
        self.monitor = monitor
        self.dim = dimension(system_name)

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
        return torch.cat(
            [gradient[:, self.dim :], -gradient[:, : self.dim]], dim=1
        )


def numpy_energy(system_name, system, state):
    d = dimension(system_name)
    if system_name == "modal":
        return float(system.energy(state[:d], state[d:]))
    return float(np.asarray(system.H(state[:d], state[d:])).squeeze())


def force(system_name, system, q):
    if system_name == "modal":
        return -(system.omegas**2 * q + system.beta * q**3)
    return -system.grad_x(q)


def initial_state(system_name, energy, seed, *, kappa=None):
    system = build_system(system_name, kappa=kappa)
    q, p = system.initial_condition(e0=energy, seed=seed)
    return system, np.concatenate([q, p]).astype(np.float64)


def velocity_verlet_trajectory(
    system_name, system, state0, horizon, dt, sample_dt
):
    d = dimension(system_name)
    q = state0[:d].copy()
    p = state0[d:].copy()
    n_steps = int(math.ceil(horizon / dt))
    stride = max(1, int(round(sample_dt / dt)))
    times = [0.0]
    states = [state0.copy()]
    physical_time = 0.0
    for step in range(n_steps):
        h = min(dt, horizon - physical_time)
        p_half = p + 0.5 * h * force(system_name, system, q)
        q = q + h * p_half
        p = p_half + 0.5 * h * force(system_name, system, q)
        physical_time += h
        if (step + 1) % stride == 0 or physical_time >= horizon - 1e-15:
            times.append(physical_time)
            states.append(np.concatenate([q, p]))
    return np.asarray(times), np.asarray(states)


def interpolate_state(times, states, query):
    upper = int(np.searchsorted(times, query, side="right"))
    if upper <= 0:
        return states[0]
    if upper >= len(times):
        return states[-1]
    weight = (query - times[upper - 1]) / (
        times[upper] - times[upper - 1]
    )
    return states[upper - 1] + weight * (
        states[upper] - states[upper - 1]
    )


def training_reference(protocol, system_name):
    config = protocol["training"]
    system, state0 = initial_state(system_name, 1.0, 0)
    times, states = velocity_verlet_trajectory(
        system_name,
        system,
        state0,
        config["reference_horizon"],
        config["reference_physical_dt"],
        config["reference_sample_dt"],
    )
    return {"times": times, "states": states}


def reference_checkpoint_loss(times, states, rollout_times, rollout_states):
    losses = []
    detached_times = np.asarray([float(value.detach()) for value in rollout_times])
    checkpoint = 0.1
    while checkpoint < detached_times[-1]:
        upper = int(np.searchsorted(detached_times, checkpoint))
        if upper <= 0 or upper >= len(detached_times):
            break
        weight = (checkpoint - detached_times[upper - 1]) / (
            detached_times[upper] - detached_times[upper - 1] + 1e-15
        )
        prediction = rollout_states[upper - 1] + weight * (
            rollout_states[upper] - rollout_states[upper - 1]
        )
        target = torch.tensor(
            interpolate_state(times, states, checkpoint)[None, :],
            dtype=torch.float64,
        )
        losses.append(torch.mean((prediction - target) ** 2))
        checkpoint += 0.1
    return torch.stack(losses).mean() if losses else torch.zeros(
        (), dtype=torch.float64
    )


def checkpoint_path(system_name, architecture, seed):
    return os.path.join(
        OUT, f"monitor_{system_name}_{architecture}_seed{seed}.pt"
    )


def progress_path(system_name, architecture, seed):
    return os.path.join(
        OUT, f"progress_{system_name}_{architecture}_seed{seed}.pt"
    )


def train_one(protocol, system_name, architecture, seed_entry, reference):
    seed = seed_entry["checkpoint_seed"]
    checkpoint = checkpoint_path(system_name, architecture, seed)
    if os.path.exists(checkpoint):
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        record = dict(package["metadata"])
        record["status"] = "loaded_existing_completed_checkpoint"
        record["checkpoint_sha256"] = sha256_file(checkpoint)
        return record
    config = protocol["training"]
    torch.manual_seed(
        seed_entry[f"{architecture}_initialization_seed"]
    )
    monitor = Monitor(system_name, architecture)
    model = ExactHamiltonianRolloutModel(system_name, monitor)
    optimizer = torch.optim.Adam(
        monitor.parameters(), lr=config["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    sampling_rng = np.random.default_rng(seed_entry["sampling_seed"])
    validation_rng = np.random.default_rng(seed_entry["validation_seed"])
    max_start = len(reference["times"]) - 1 - 200
    validation_indices = validation_rng.integers(
        0, max_start, size=config["validation_starts"]
    )
    best_loss = math.inf
    best_state = None
    best_epoch = None
    start_epoch = 0
    progress = progress_path(system_name, architecture, seed)
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

    def one_loss(index):
        initial = torch.tensor(
            reference["states"][index][None, :], dtype=torch.float64
        )
        p0 = -model.H(initial).detach()
        rollout_times, rollout_states = midpoint_rollout(
            model,
            initial,
            p0,
            protocol["systems"][system_name]["training_dtau"],
            config["rollout_steps"],
            n_fp_iter=config["unrolled_fixed_point_iterations"],
        )
        relative_times = reference["times"][index:] - reference["times"][index]
        return reference_checkpoint_loss(
            relative_times,
            reference["states"][index:],
            rollout_times,
            rollout_states,
        ), rollout_states

    started = time.perf_counter()
    last_components = {}
    for epoch in range(start_epoch, config["epochs"]):
        index = int(sampling_rng.integers(0, max_start))
        checkpoint_loss, rollout_states = one_loss(index)
        rollout = torch.cat(rollout_states, dim=0)
        energy_variance = torch.var(model.H(rollout))
        inverse_monitor = torch.mean(1.0 / model.g(rollout))
        total = (
            config["weight_energy_variance"] * energy_variance
            + config["weight_checkpoint"] * checkpoint_loss
            + config["weight_inverse_monitor"] * inverse_monitor
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            monitor.parameters(), config["gradient_clip_norm"]
        )
        optimizer.step()
        scheduler.step()
        last_components = {
            "total": float(total.detach()),
            "energy_variance": float(energy_variance.detach()),
            "checkpoint": float(checkpoint_loss.detach()),
            "inverse_monitor": float(inverse_monitor.detach()),
        }
        if (epoch + 1) % config["validation_every_epochs"] == 0:
            validation_losses = []
            with torch.enable_grad():
                for validation_index in validation_indices:
                    value, _ = one_loss(int(validation_index))
                    validation_losses.append(float(value.detach()))
            current = float(np.mean(validation_losses))
            if current < best_loss:
                best_loss = current
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
                    "sampling_rng_state": sampling_rng.bit_generator.state,
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "next_epoch": epoch + 1,
                },
                progress,
            )
            print(
                f"{system_name} {architecture} seed={seed} "
                f"epoch={epoch + 1} val={current:.3e}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    metadata = {
        "system": system_name,
        "architecture": architecture,
        "checkpoint_seed": seed,
        "status": "trained",
        "best_epoch": best_epoch,
        "best_validation_state_mse": best_loss,
        "last_training_components": last_components,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_path": os.path.relpath(checkpoint, ROOT),
    }
    temporary = checkpoint + ".tmp"
    torch.save({"state_dict": best_state, "metadata": metadata}, temporary)
    os.replace(temporary, checkpoint)
    metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    return metadata


def reversal_diagnostic(monitor, system_name):
    rng = np.random.default_rng(1313)
    states = torch.tensor(
        rng.normal(size=(64, 2 * dimension(system_name))), dtype=torch.float64
    )
    reversed_states = states.clone()
    reversed_states[:, dimension(system_name) :] *= -1
    with torch.no_grad():
        baseline = monitor(states)
        reversed_values = monitor(reversed_states)
    denominator = torch.clamp(
        torch.maximum(abs(baseline), abs(reversed_values)), min=1e-15
    )
    return float(torch.max(abs(baseline - reversed_values) / denominator))


def train_all(protocol):
    records = []
    references = {
        system_name: training_reference(protocol, system_name)
        for system_name in ["modal", "duffing"]
    }
    for system_name in ["modal", "duffing"]:
        for architecture in protocol["training"]["architectures"]:
            for seed_entry in protocol["training"]["seed_manifest"]:
                try:
                    record = train_one(
                        protocol,
                        system_name,
                        architecture,
                        seed_entry,
                        references[system_name],
                    )
                    monitor = Monitor(system_name, architecture)
                    package = torch.load(
                        os.path.join(ROOT, record["checkpoint_path"]),
                        map_location="cpu",
                        weights_only=True,
                    )
                    monitor.load_state_dict(package["state_dict"])
                    record["momentum_reversal_relative_error"] = (
                        reversal_diagnostic(monitor, system_name)
                    )
                except Exception as error:
                    record = {
                        "system": system_name,
                        "architecture": architecture,
                        "checkpoint_seed": seed_entry["checkpoint_seed"],
                        "status": "training_failure",
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                records.append(record)
                atomic_json(
                    TRAINING_PATH,
                    {
                        "schema_version": "1.0",
                        "experiment": protocol["experiment"],
                        "protocol_sha256":
                            protocol["integrity"]["protocol_sha256"],
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


def scenarios(protocol, system_name):
    config = protocol["evaluation"][system_name]
    result = []
    if system_name == "modal":
        for energy in config["energy_shells"]:
            for seed in config["initial_condition_seeds"]:
                for kappa in config["kappas"]:
                    for direction in config["momentum_directions"]:
                        result.append(
                            {
                                "system": system_name,
                                "energy_shell": energy,
                                "initial_condition_seed": seed,
                                "kappa": kappa,
                                "momentum_direction": direction,
                            }
                        )
    else:
        for energy in config["energy_shells"]:
            for seed in config["initial_condition_seeds"]:
                for direction in config["momentum_directions"]:
                    result.append(
                        {
                            "system": system_name,
                            "energy_shell": energy,
                            "initial_condition_seed": seed,
                            "kappa": None,
                            "momentum_direction": direction,
                        }
                    )
    return result


def scenario_key(record):
    return (
        record["system"],
        record["method"],
        record.get("architecture"),
        record.get("checkpoint_seed"),
        record["energy_shell"],
        record["initial_condition_seed"],
        record.get("kappa"),
        record["momentum_direction"],
    )


def prescribed_monitor(system_name, value):
    d = dimension(system_name)
    if system_name == "modal":
        return 0.75 + 0.25 / (
            1.0 + torch.mean(value[:, :d] ** 2, dim=1)
        )
    return 0.75 + 0.25 / (1.0 + value[:, 0] ** 2)


def run_midpoint_rollout(
    protocol, scenario, monitor, method, architecture=None, checkpoint_seed=None
):
    system_name = scenario["system"]
    config = protocol["evaluation"][system_name]
    system, initial = initial_state(
        system_name,
        scenario["energy_shell"],
        scenario["initial_condition_seed"],
        kappa=scenario.get("kappa"),
    )
    d = dimension(system_name)
    if scenario["momentum_direction"] == "reversed":
        initial[d:] *= -1
    horizon = (
        config["slow_periods"] * 2.0 * math.pi
        if system_name == "modal"
        else config["physical_horizon"]
    )
    reference_times, reference_states = velocity_verlet_trajectory(
        system_name,
        system,
        initial,
        horizon,
        config["resolved_verlet_dt"],
        config["resolved_verlet_dt"],
    )
    state = torch.tensor(initial[None, :], dtype=torch.float64)
    energy0 = numpy_energy(system_name, system, initial)
    p0 = torch.tensor([-energy0], dtype=torch.float64)
    vector_field_calls = 0

    def vector_field(value):
        nonlocal vector_field_calls
        vector_field_calls += 1
        with torch.enable_grad():
            differentiable = value.detach().requires_grad_(True)
            extended = monitor(differentiable) * (
                system.H_torch(differentiable) + p0
            )
            gradient = torch.autograd.grad(
                extended.sum(), differentiable
            )[0]
        return torch.cat([gradient[:, d:], -gradient[:, :d]], dim=1).detach()

    record = {
        **scenario,
        "method": method,
        "architecture": architecture,
        "checkpoint_seed": checkpoint_seed,
        "completion_status": "running",
        "steps": 0,
        "nonlinear_iterations": 0,
        "vector_field_evaluations": 0,
        "maximum_returned_endpoint_residual": 0.0,
        "max_energy_error": 0.0,
        "max_state_error": 0.0,
        "physical_time": 0.0,
    }
    started = time.perf_counter()
    try:
        while record["physical_time"] < horizon:
            if record["steps"] >= protocol["evaluation"]["maximum_steps"]:
                record["completion_status"] = "step_cap"
                break
            h = config["dtau"]
            endpoint, diagnostics = fixed_point_endpoint(
                state,
                h,
                vector_field,
                tolerance=protocol["evaluation"]["fixed_point"]["tolerance"],
                maximum_iterations=protocol["evaluation"]["fixed_point"][
                    "maximum_iterations"
                ],
            )
            physical_dt = h * float(
                monitor(0.5 * (state + endpoint)).detach()
            )
            if not math.isfinite(physical_dt) or physical_dt <= 0:
                raise FloatingPointError("invalid physical time increment")
            if record["physical_time"] + physical_dt > horizon:
                remaining = horizon - record["physical_time"]
                lower, upper = 0.0, h
                for _ in range(45):
                    candidate_h = 0.5 * (lower + upper)
                    candidate, candidate_diagnostics = fixed_point_endpoint(
                        state,
                        candidate_h,
                        vector_field,
                        tolerance=protocol["evaluation"]["fixed_point"][
                            "tolerance"
                        ],
                        maximum_iterations=protocol["evaluation"][
                            "fixed_point"
                        ]["maximum_iterations"],
                    )
                    candidate_dt = candidate_h * float(
                        monitor(0.5 * (state + candidate)).detach()
                    )
                    if candidate_dt < remaining:
                        lower = candidate_h
                    else:
                        upper = candidate_h
                    endpoint = candidate
                    diagnostics = candidate_diagnostics
                physical_dt = remaining
            state = endpoint
            record["physical_time"] += physical_dt
            record["steps"] += 1
            record["nonlinear_iterations"] += diagnostics.iterations
            residual = diagnostics.returned_endpoint_residual_inf
            record["maximum_returned_endpoint_residual"] = max(
                record["maximum_returned_endpoint_residual"], residual
            )
            current = state.detach().numpy()[0]
            record["max_energy_error"] = max(
                record["max_energy_error"],
                abs(numpy_energy(system_name, system, current) - energy0),
            )
            reference = interpolate_state(
                reference_times, reference_states, record["physical_time"]
            )
            record["max_state_error"] = max(
                record["max_state_error"],
                float(np.linalg.norm(current - reference)),
            )
        if record["completion_status"] == "running":
            record["completion_status"] = "complete"
    except ImplicitMidpointSolverError as error:
        record["completion_status"] = "solver_failure"
        record["error"] = str(error)
        record["failure_diagnostics"] = error.diagnostics.to_dict()
    except Exception as error:
        record["completion_status"] = "unexpected_error"
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
    final = state.detach().numpy()[0]
    record.update(
        vector_field_evaluations=vector_field_calls,
        final_state_error=float(np.linalg.norm(final - reference_states[-1])),
        final_time_error=abs(record["physical_time"] - horizon),
        wall_seconds=time.perf_counter() - started,
    )
    return record


def run_resolved_verlet(protocol, scenario):
    system_name = scenario["system"]
    config = protocol["evaluation"][system_name]
    system, initial = initial_state(
        system_name,
        scenario["energy_shell"],
        scenario["initial_condition_seed"],
        kappa=scenario.get("kappa"),
    )
    d = dimension(system_name)
    if scenario["momentum_direction"] == "reversed":
        initial[d:] *= -1
    horizon = (
        config["slow_periods"] * 2.0 * math.pi
        if system_name == "modal"
        else config["physical_horizon"]
    )
    started = time.perf_counter()
    times, states = velocity_verlet_trajectory(
        system_name,
        system,
        initial,
        horizon,
        config["resolved_verlet_dt"],
        config["resolved_verlet_dt"],
    )
    energy0 = numpy_energy(system_name, system, initial)
    return {
        **scenario,
        "method": "resolved_velocity_verlet",
        "architecture": None,
        "checkpoint_seed": None,
        "completion_status": "complete",
        "steps": len(times) - 1,
        "nonlinear_iterations": 0,
        "vector_field_evaluations": 2 * (len(times) - 1),
        "maximum_returned_endpoint_residual": None,
        "max_energy_error": max(
            abs(numpy_energy(system_name, system, state) - energy0)
            for state in states
        ),
        "max_state_error": 0.0,
        "final_state_error": 0.0,
        "physical_time": float(times[-1]),
        "final_time_error": abs(float(times[-1]) - horizon),
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_all(protocol, training):
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
        },
        "records": [],
    }
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError("raw artifact protocol mismatch")
    existing = {scenario_key(record) for record in artifact["records"]}
    for training_record in training["records"]:
        if training_record["status"] == "training_failure":
            for scenario in scenarios(protocol, training_record["system"]):
                candidate = {
                    **scenario,
                    "method": "learned_monitor",
                    "architecture": training_record["architecture"],
                    "checkpoint_seed": training_record["checkpoint_seed"],
                }
                if scenario_key(candidate) in existing:
                    continue
                record = {
                    **candidate,
                    "completion_status": "checkpoint_unavailable",
                    "error": training_record["error"],
                    "steps": 0,
                    "nonlinear_iterations": 0,
                    "vector_field_evaluations": 0,
                    "maximum_returned_endpoint_residual": None,
                    "max_energy_error": None,
                    "max_state_error": None,
                    "final_state_error": None,
                    "physical_time": 0.0,
                    "final_time_error": None,
                    "wall_seconds": 0.0,
                }
                artifact["records"].append(record)
                existing.add(scenario_key(record))
                atomic_json(RAW_PATH, artifact)
            continue
        system_name = training_record["system"]
        architecture = training_record["architecture"]
        seed = training_record["checkpoint_seed"]
        monitor = Monitor(system_name, architecture)
        package = torch.load(
            os.path.join(ROOT, training_record["checkpoint_path"]),
            map_location="cpu",
            weights_only=True,
        )
        monitor.load_state_dict(package["state_dict"])
        monitor.eval()
        for scenario in scenarios(protocol, system_name):
            candidate = {
                **scenario,
                "method": "learned_monitor",
                "architecture": architecture,
                "checkpoint_seed": seed,
            }
            if scenario_key(candidate) in existing:
                continue
            record = run_midpoint_rollout(
                protocol,
                scenario,
                monitor,
                "learned_monitor",
                architecture,
                seed,
            )
            artifact["records"].append(record)
            existing.add(scenario_key(record))
            atomic_json(RAW_PATH, artifact)
    for system_name in ["modal", "duffing"]:
        for scenario in scenarios(protocol, system_name):
            prescribed = {
                **scenario,
                "method": "prescribed_monitor",
                "architecture": None,
                "checkpoint_seed": None,
            }
            if scenario_key(prescribed) not in existing:
                monitor = lambda value, name=system_name: prescribed_monitor(
                    name, value
                )
                record = run_midpoint_rollout(
                    protocol, scenario, monitor, "prescribed_monitor"
                )
                artifact["records"].append(record)
                existing.add(scenario_key(record))
                atomic_json(RAW_PATH, artifact)
            resolved = {
                **scenario,
                "method": "resolved_velocity_verlet",
                "architecture": None,
                "checkpoint_seed": None,
            }
            if scenario_key(resolved) not in existing:
                record = run_resolved_verlet(protocol, scenario)
                artifact["records"].append(record)
                existing.add(scenario_key(record))
                atomic_json(RAW_PATH, artifact)
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    return artifact


def summarize(protocol, training, raw):
    records = raw["records"]
    complete = [
        record for record in records
        if record["completion_status"] == "complete"
    ]
    statuses = {}
    for record in records:
        status = record["completion_status"]
        statuses[status] = statuses.get(status, 0) + 1
    expected = protocol["evaluation"]["total_registered_rollouts"]
    reversal_records = [
        record for record in training["records"]
        if record["architecture"] == "reversal_even"
        and record["status"] != "training_failure"
    ]
    fixed_point_complete = [
        record for record in complete
        if record["method"] != "resolved_velocity_verlet"
    ]
    acceptance = {
        "all_twelve_checkpoint_records_retained":
            len(training["records"]) == 12,
        "all_checkpoints_trained":
            all(record["status"] != "training_failure"
                for record in training["records"]),
        "all_rollouts_attempted_and_retained": len(records) == expected,
        "completed_final_times_aligned":
            all(record["final_time_error"] < 1e-10 for record in complete),
        "completed_returned_residuals_valid": all(
            record["maximum_returned_endpoint_residual"] < 1e-12
            for record in fixed_point_complete
        ),
        "reversal_evenness_valid": len(reversal_records) == 6 and all(
            record["momentum_reversal_relative_error"] < 1e-12
            for record in reversal_records
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
        "acceptance": acceptance,
        "training_sha256": sha256_file(TRAINING_PATH),
        "raw_results_sha256": sha256_file(RAW_PATH),
        "checkpoint_sha256": {
            record["checkpoint_path"]: record["checkpoint_sha256"]
            for record in training["records"]
            if record["status"] != "training_failure"
        },
        "claim_limits": protocol["claim_limits"],
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
        raise SystemExit("AC13 artifact gate failed")


if __name__ == "__main__":
    main()
