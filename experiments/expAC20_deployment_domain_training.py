"""AC20: deployment-domain training and Sundman-factored monitors.

AC19 identified the mechanism of the AC9 high-eccentricity boundary: below
its training radius the learned monitor plateaus instead of vanishing with
the radius, the implicit-midpoint fixed point loses contraction, and a
smaller fictitious step does not restore accuracy.  AC20 tests the two
remedies the mechanism suggests, with the AC9 recipe otherwise unchanged:

* ``invariant_e099`` -- the AC9 invariant-feature monitor trained on the
  a=1, e=0.99 orbit (periapsis 0.01), so that the deployment radii of the
  AC9 grid lie inside the training domain;
* ``sundman_e09``    -- a Sundman-factored monitor g = r * 2*sigmoid(NN(phi))
  with the AC9 invariant features, initialised at g = r, trained on the AC9
  a=1, e=0.9 orbit (unchanged training domain);
* ``sundman_e099``   -- the Sundman-factored monitor trained on the e=0.99
  orbit.

Each arm has five checkpoints that reuse the AC9 seed manifest, so every
new checkpoint is paired with the AC9 invariant checkpoint of the same seed.
For every checkpoint the runner (1) trains, (2) computes the prospective
AC19 contraction diagnostic rho_max on the 36 AC9 reference orbits, and
(3) only then runs the 72 registered AC9 scenarios with the frozen AC9
``run_rollout``.  Steps (2) and (3) are timestamped so that the prediction
demonstrably precedes the outcome.

Work is organised as one job per checkpoint (15 jobs) with per-job artifact
files under ``results/expAC20/parts``; ``--launch N`` spawns N processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC20")
PARTS = os.path.join(OUT, "parts")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
AC9_DIR = os.path.join(ROOT, "results", "expAC9")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from models.pasnet import make_mlp, midpoint_rollout  # noqa: E402
from systems import Kepler2D  # noqa: E402
import expAC9_end_to_end_monitor_generalisation as ac9  # noqa: E402
import expAC19_periapsis_contraction_diagnostic as ac19  # noqa: E402

SEEDS = (7101, 7102, 7103, 7104, 7105)
ARMS = ("invariant_e099", "sundman_e09", "sundman_e099")


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
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


def atomic_json(path, payload, attempts=60, delay_seconds=0.5):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_protocol(protocol_path, skip_roles=()):
    protocol = load_json(protocol_path)
    expected = protocol["integrity"]["protocol_sha256"]
    actual = canonical_protocol_hash(protocol)
    if actual != expected:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {expected}")
    for source in protocol["frozen_sources"]:
        if source.get("role") in skip_roles:
            continue
        actual_source = sha256_file(os.path.join(ROOT, source["path"]))
        if actual_source != source["sha256"]:
            raise RuntimeError(
                f"frozen source changed: {source['path']} "
                f"{actual_source} != {source['sha256']}"
            )
    return protocol


def environment():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


# ---------------------------------------------------------------------------
# robust Kepler states (the AC9 Newton solver fails for batched e=0.99)
# ---------------------------------------------------------------------------
def kepler_state_robust(semi_major_axis, eccentricity, mean_anomaly):
    anomalies = np.atleast_1d(np.asarray(mean_anomaly, dtype=np.float64))
    eccentric = np.array(
        [ac19.solve_kepler_equation(value, eccentricity) for value in anomalies]
    )
    states = ac19.state_from_eccentric_anomaly(
        semi_major_axis, eccentricity, eccentric
    )
    if np.ndim(mean_anomaly) == 0:
        return states[0]
    return states


# ---------------------------------------------------------------------------
# monitors
# ---------------------------------------------------------------------------
class SundmanMonitor(nn.Module):
    """g(z) = r(z) * 2*sigmoid(NN(phi(z))) with the AC9 invariant features.

    The final bias is zero so that the cold start is exactly the manual
    Sundman monitor g = r; training learns a bounded multiplicative
    correction in (0, 2).  The factor r guarantees g -> 0 like r at
    collision for every parameter value.
    """

    def __init__(self):
        super().__init__()
        self.architecture = "sundman_invariant"
        self.network = make_mlp(5, 64, 2).double()
        final = self.network[-1]
        nn.init.normal_(final.weight, std=0.1)
        nn.init.constant_(final.bias, 0.0)

    def forward(self, z):
        radius = torch.linalg.vector_norm(z[..., :2], dim=-1)
        features = ac9.invariant_features(z)
        return radius * 2.0 * torch.sigmoid(
            self.network(features).squeeze(-1)
        )


def build_monitor(arm):
    if arm.startswith("sundman"):
        return SundmanMonitor()
    return ac9.Monitor("invariant")


def arm_training_eccentricity(protocol, arm):
    return protocol["arms"][arm]["training_eccentricity"]


def checkpoint_path(arm, seed):
    return os.path.join(OUT, f"monitor_{arm}_seed{seed}.pt")


def part_dir(arm, seed):
    return os.path.join(PARTS, f"{arm}_seed{seed}")


def load_checkpoint(arm, seed):
    monitor = build_monitor(arm)
    package = torch.load(
        checkpoint_path(arm, seed), map_location="cpu", weights_only=True
    )
    monitor.load_state_dict(package["state_dict"])
    monitor.eval()
    return monitor


# ---------------------------------------------------------------------------
# training (AC9 recipe with a parameterised training orbit)
# ---------------------------------------------------------------------------
def checkpoint_loss_terms(model, states, times, mean_anomaly, eccentricity,
                          config):
    detached_times = np.array([float(value.detach()) for value in times])
    loss = torch.zeros((), dtype=torch.float64)
    count = 0
    checkpoint_time = config["checkpoint_physical_dt"]
    while checkpoint_time < detached_times[-1]:
        upper = int(np.searchsorted(detached_times, checkpoint_time))
        if upper <= 0 or upper >= len(detached_times):
            break
        weight = (checkpoint_time - detached_times[upper - 1]) / (
            detached_times[upper] - detached_times[upper - 1] + 1e-15
        )
        prediction = states[upper - 1] + weight * (
            states[upper] - states[upper - 1]
        )
        target = torch.tensor(
            kepler_state_robust(
                1.0, eccentricity, mean_anomaly + checkpoint_time
            )[None, :],
            dtype=torch.float64,
        )
        loss = loss + torch.mean((prediction - target) ** 2)
        count += 1
        checkpoint_time += config["checkpoint_physical_dt"]
    if count:
        loss = loss / count
    return loss, count


def validation_loss(model, start_anomalies, eccentricity, config):
    losses = []
    for mean_anomaly in start_anomalies:
        state = torch.tensor(
            kepler_state_robust(1.0, eccentricity, mean_anomaly)[None, :],
            dtype=torch.float64,
        )
        p0 = -model.H(state).detach()
        try:
            times, states = midpoint_rollout(
                model, state, p0, config["dtau"], config["rollout_steps"],
                n_fp_iter=config["unrolled_fixed_point_iterations"],
            )
        except Exception:  # nonfinite rollouts count as infinite loss
            return math.inf
        states = [value.detach() for value in states]
        loss, count = checkpoint_loss_terms(
            model, states, times, mean_anomaly, eccentricity, config
        )
        if count == 0 or not math.isfinite(float(loss)):
            return math.inf
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else math.inf


def train_one(protocol, arm, seed):
    config = protocol["training"]
    eccentricity = arm_training_eccentricity(protocol, arm)
    seed_entry = next(
        entry for entry in config["seed_manifest"]
        if entry["checkpoint_seed"] == seed
    )
    path = checkpoint_path(arm, seed)
    if os.path.exists(path):
        package = torch.load(path, map_location="cpu", weights_only=True)
        record = dict(package["metadata"])
        record["status"] = "loaded_existing_completed_checkpoint"
        record["checkpoint_sha256"] = sha256_file(path)
        return record

    torch.manual_seed(seed_entry["invariant_initialization_seed"])
    monitor = build_monitor(arm)
    model = ac9.ExactHamiltonianRolloutModel(monitor, eccentricity)
    optimizer = torch.optim.Adam(monitor.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    sampling_rng = np.random.default_rng(seed_entry["sampling_seed"])
    validation_rng = np.random.default_rng(seed_entry["validation_seed"])
    validation_anomalies = validation_rng.uniform(
        0.0, 2.0 * math.pi, config["validation_starts"]
    )
    sample_count = config["training_states"]
    anomalies = 2.0 * math.pi * np.arange(sample_count) / sample_count
    training_states = kepler_state_robust(1.0, eccentricity, anomalies)
    radii = np.linalg.norm(training_states[:, :2], axis=1)
    probabilities = 1.0 / (
        radii ** config["periapsis_sampling_power"]
        + config["periapsis_sampling_offset"]
    )
    probabilities /= probabilities.sum()

    best_loss, best_state, best_epoch = math.inf, None, None
    nonfinite_epochs = 0
    validation_history = []
    started = time.perf_counter()
    last_components = {}
    for epoch in range(config["epochs"]):
        index = int(sampling_rng.choice(sample_count, p=probabilities))
        mean_anomaly = float(anomalies[index])
        initial_state = torch.tensor(
            training_states[index][None, :], dtype=torch.float64
        )
        p0 = -model.H(initial_state).detach()
        try:
            times, states = midpoint_rollout(
                model, initial_state, p0, config["dtau"],
                config["rollout_steps"],
                n_fp_iter=config["unrolled_fixed_point_iterations"],
            )
            rollout = torch.cat(states, dim=0)
            energy_variance = torch.var(model.H(rollout))
            checkpoint_loss, checkpoint_count = checkpoint_loss_terms(
                model, states, times, mean_anomaly, eccentricity, config
            )
            cost_loss = torch.mean(1.0 / model.g(rollout))
            total_loss = (
                config["weight_energy_variance"] * energy_variance
                + config["weight_checkpoint"] * checkpoint_loss
                + config["weight_inverse_monitor"] * cost_loss
            )
            finite = bool(torch.isfinite(total_loss))
        except Exception:
            finite = False
        if not finite:
            # AC20 guard (disclosed): a non-finite unrolled loss is skipped
            # and counted instead of poisoning the parameters.
            nonfinite_epochs += 1
            scheduler.step()
            continue
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            monitor.parameters(), config["gradient_clip_norm"]
        )
        if not torch.isfinite(grad_norm):
            nonfinite_epochs += 1
            optimizer.zero_grad()
            scheduler.step()
            continue
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
            current = validation_loss(
                model, validation_anomalies, eccentricity, config
            )
            validation_history.append(
                {"epoch": epoch + 1,
                 "validation_state_mse": current if math.isfinite(current)
                 else None}
            )
            if current < best_loss:
                best_loss = current
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in monitor.state_dict().items()
                }
            print(
                f"[train] {arm} seed={seed} epoch={epoch + 1} "
                f"val={current:.3e} best={best_loss:.3e} "
                f"nonfinite={nonfinite_epochs}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError(
            f"no finite validation checkpoint selected for {arm} seed {seed}"
        )
    monitor.load_state_dict(best_state)
    metadata = {
        "arm": arm,
        "architecture": monitor.architecture,
        "training_eccentricity": eccentricity,
        "checkpoint_seed": seed,
        "status": "trained",
        "best_epoch": best_epoch,
        "best_validation_state_mse": best_loss,
        "validation_history": validation_history,
        "nonfinite_epochs_skipped": nonfinite_epochs,
        "last_training_components": last_components,
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_path": os.path.relpath(path, ROOT).replace("\\", "/"),
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, path)
    metadata["checkpoint_sha256"] = sha256_file(path)
    return metadata


# ---------------------------------------------------------------------------
# prospective diagnostic (AC19 statistic) and evaluation (AC9 rollouts)
# ---------------------------------------------------------------------------
def diagnose_one(protocol, ac9_protocol, arm, seed, monitor):
    config = protocol["diagnostic"]
    anomalies = np.linspace(
        0.0, 2.0 * math.pi, config["states_per_orbit"], endpoint=False
    )
    records = []
    for family in ac19.orbit_families(ac9_protocol):
        system = Kepler2D(family["eccentricity"])
        base = ac19.state_from_eccentric_anomaly(
            family["semi_major_axis"], family["eccentricity"], anomalies
        )
        states = ac19.transform_states(
            base, family["orientation_radians"], family["momentum_direction"]
        )
        diagnostic = ac19.orbit_diagnostic(
            monitor, system, states, config["dtau"]
        )
        summary = ac19.summarise_orbit(
            diagnostic, anomalies, family["semi_major_axis"],
            family["eccentricity"], config["training_radius_minimum"],
        )
        record = {
            "arm": arm,
            "architecture": monitor.architecture,
            "checkpoint_seed": seed,
            **family,
            **summary,
            "predicted_fixed_point_failure":
                summary["spectral_radius_maximum"]
                >= config["contraction_threshold"],
        }
        if (
            family["semi_major_axis"] == 1.0
            and family["orientation_radians"] == 0.0
            and family["momentum_direction"] == "forward"
        ):
            stride = config["profile_stride"]
            record["profile"] = {
                "eccentric_anomaly": anomalies[::stride].tolist(),
                "radius": diagnostic["radius"][::stride].tolist(),
                "monitor": diagnostic["monitor"][::stride].tolist(),
                "spectral_radius":
                    diagnostic["spectral_radius"][::stride].tolist(),
            }
        records.append(record)
        print(
            f"[diag] {arm} seed={seed} a={family['semi_major_axis']} "
            f"e={family['eccentricity']} "
            f"angle={family['orientation_radians']:.3f} "
            f"{family['momentum_direction']}: "
            f"rho_max={summary['spectral_radius_maximum']:.3f} "
            f"g/r(peri)={summary['monitor_over_radius_at_periapsis']:.3f}",
            flush=True,
        )
    return records


def evaluate_one(protocol, ac9_protocol, arm, seed, monitor, part_path):
    evaluation = dict(ac9_protocol["evaluation"])
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "arm": arm,
        "checkpoint_seed": seed,
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [],
    }
    if os.path.exists(part_path):
        artifact = load_json(part_path)
    existing = {ac19.scenario_key(r) for r in artifact["records"]}
    for scenario in ac9.scenarios_from_protocol(ac9_protocol):
        probe = {"architecture": monitor.architecture, "checkpoint_seed": seed,
                 **scenario}
        if ac19.scenario_key(probe) in existing:
            continue
        record = ac9.run_rollout(
            monitor, monitor.architecture, seed, scenario, evaluation
        )
        record["arm"] = arm
        artifact["records"].append(record)
        atomic_json(part_path, artifact)
        print(
            f"[eval] {arm} seed={seed} a={scenario['semi_major_axis']} "
            f"e={scenario['eccentricity']} "
            f"M={scenario['initial_mean_anomaly']:.3f} "
            f"angle={scenario['orientation_radians']:.3f} "
            f"{scenario['momentum_direction']}: "
            f"{record['completion_status']} steps={record['steps']} "
            f"closure={record['final_state_closure_error']:.3e}",
            flush=True,
        )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(part_path, artifact)
    return artifact


def run_job(protocol, ac9_protocol, arm, seed):
    directory = part_dir(arm, seed)
    os.makedirs(directory, exist_ok=True)
    training_path = os.path.join(directory, "training.json")
    diagnostic_path = os.path.join(directory, "diagnostic.json")
    rollouts_path = os.path.join(directory, "rollouts.json")

    if os.path.exists(training_path):
        training = load_json(training_path)
    else:
        record = train_one(protocol, arm, seed)
        training = {
            "schema_version": "1.0",
            "experiment": protocol["experiment"],
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "environment": environment(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "record": record,
        }
        monitor = load_checkpoint(arm, seed)
        training["record"]["symmetry"] = ac9.symmetry_diagnostics(monitor)
        atomic_json(training_path, training)
    monitor = load_checkpoint(arm, seed)

    if not os.path.exists(diagnostic_path):
        started = datetime.now(timezone.utc).isoformat()
        records = diagnose_one(protocol, ac9_protocol, arm, seed, monitor)
        atomic_json(diagnostic_path, {
            "schema_version": "1.0",
            "experiment": protocol["experiment"],
            "protocol_sha256": protocol["integrity"]["protocol_sha256"],
            "arm": arm,
            "checkpoint_seed": seed,
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dtau": protocol["diagnostic"]["dtau"],
            "orbit_records": records,
        })

    if not (os.path.exists(rollouts_path)
            and "completed_at_utc" in load_json(rollouts_path)):
        evaluate_one(protocol, ac9_protocol, arm, seed, monitor, rollouts_path)


def jobs():
    return [(arm, seed) for arm in ARMS for seed in SEEDS]


def launch(num_processes):
    os.makedirs(PARTS, exist_ok=True)
    processes = []
    for index in range(num_processes):
        log = open(os.path.join(PARTS, f"worker_{index:02d}.log"), "a")
        processes.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__),
             "--worker", str(index), "--num-workers", str(num_processes)],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
        ))
    print(f"launched {num_processes} workers; pids="
          f"{[p.pid for p in processes]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--launch", type=int, default=None)
    parser.add_argument("--pilot", action="store_true",
                        help="time a few training epochs outside the sealed "
                             "results directory")
    parser.add_argument("--pilot-epochs", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    if args.pilot:
        run_pilot(args.pilot_epochs)
        return
    if args.launch:
        launch(args.launch)
        return
    protocol = verify_protocol(
        PROTOCOL_PATH, skip_roles=("prospective_analysis",)
    )
    ac9_protocol = load_json(os.path.join(AC9_DIR, "preregistration.json"))
    for index, (arm, seed) in enumerate(jobs()):
        if args.worker is not None and index % args.num_workers != args.worker:
            continue
        run_job(protocol, ac9_protocol, arm, seed)


def run_pilot(epochs):
    """Time the training recipe for each arm; writes nothing to results."""
    global OUT
    OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "expAC20_pilot")
    os.makedirs(OUT, exist_ok=True)
    ac9_protocol = load_json(os.path.join(AC9_DIR, "preregistration.json"))
    training = dict(ac9_protocol["training"])
    training["epochs"] = epochs
    training["validation_every_epochs"] = epochs
    protocol = {
        "experiment": "expAC20_pilot",
        "integrity": {"protocol_sha256": "pilot"},
        "training": training,
        "arms": {
            "invariant_e099": {"training_eccentricity": 0.99},
            "sundman_e09": {"training_eccentricity": 0.9},
            "sundman_e099": {"training_eccentricity": 0.99},
        },
        "diagnostic": {"states_per_orbit": 720, "dtau": 0.04,
                       "training_radius_minimum": 0.1,
                       "contraction_threshold": 1.0, "profile_stride": 2},
    }
    for arm in ARMS:
        path = checkpoint_path(arm, 7101)
        if os.path.exists(path):
            os.remove(path)
        started = time.perf_counter()
        record = train_one(protocol, arm, 7101)
        elapsed = time.perf_counter() - started
        print(f"[pilot] {arm}: {epochs} epochs in {elapsed:.1f}s "
              f"({elapsed / epochs:.2f}s/epoch), best={record['best_validation_state_mse']:.3e}, "
              f"nonfinite={record['nonfinite_epochs_skipped']}", flush=True)
        monitor = load_checkpoint(arm, 7101)
        system = Kepler2D(0.99)
        anomalies = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        states = ac19.state_from_eccentric_anomaly(1.0, 0.99, anomalies)
        diagnostic = ac19.orbit_diagnostic(monitor, system, states, 0.04)
        summary = ac19.summarise_orbit(diagnostic, anomalies, 1.0, 0.99, 0.1)
        print(f"[pilot] {arm}: a=1 e=0.99 rho_max="
              f"{summary['spectral_radius_maximum']:.3f} "
              f"g/r(peri)={summary['monitor_over_radius_at_periapsis']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
