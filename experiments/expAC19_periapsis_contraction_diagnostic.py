"""Sealed AC19: prospective periapsis contraction diagnostic and step-size
intervention for the frozen AC9 exact-H monitors.

Phase A (diagnostic, no rollouts). For every frozen AC9 monitor and for
manual ``g=r``, scan the exact analytic Kepler orbit of every registered AC9
scenario family ``(a, e, orientation, momentum direction)`` and record, at
each sampled state ``z`` on the orbit,

* the radius ``r`` and monitor value ``g(z)``;
* the spectral radius ``rho`` and spectral norm of the linearised
  implicit-midpoint fixed-point map ``(dtau/2) J D2_zz K(z)`` with
  ``K = g (H + p0)`` and ``p0 = -H(z0)``.

The orbit-level statistic is ``rho_max = max_z rho(z)`` at the AC9 evaluation
step ``dtau = 0.04``. Because ``rho`` is linear in ``dtau``, the linearised
fixed-point iteration is contractive along the whole reference orbit if and
only if ``rho_max < 1``. The prespecified prediction is that AC9 rollouts
with ``rho_max >= 1`` fail and rollouts with ``rho_max < 1`` complete. The
statistic is computed from the monitor and the analytic orbit alone; no
rollout iterate is used, so it is prospective rather than outcome-dependent.

Phase A also maps the physical failure time of every failed AC9 rollout to
the radius on the analytic reference orbit, locating where failures occur.

Phase B (intervention). Every failed AC9 learned-monitor rollout is rerun
with the identical frozen AC9 rollout code and the prespecified step
``dtau* = dtau * min(1, rho_target / rho_max)`` with ``rho_target = 0.5``.
The prediction is completion; the cost is reported relative to manual
``g=r`` at ``dtau = 0.04``.

The protocol must be created by ``register_expAC19.py`` before this runner is
invoked. ``--dry-run`` executes a two-monitor, two-orbit subset into a
temporary directory without touching ``results/expAC19``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from systems import Kepler2D  # noqa: E402
from expAC9_end_to_end_monitor_generalisation import (  # noqa: E402
    Monitor,
    rotate_state,
    run_rollout,
)

AC9_DIR = os.path.join(ROOT, "results", "expAC9")
DEFAULT_OUT = os.path.join(ROOT, "results", "expAC19")

SYMPLECTIC_J = torch.tensor(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
)


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
    """Atomic JSON write with retries: cloud-synchronised folders can hold a
    transient lock on the destination and raise ``PermissionError`` on
    ``os.replace``. The payload is unchanged by retrying."""
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


def verify_protocol(protocol_path):
    protocol = load_json(protocol_path)
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


# ---------------------------------------------------------------------------
# analytic Kepler orbit
# ---------------------------------------------------------------------------
def state_from_eccentric_anomaly(semi_major_axis, eccentricity, anomaly):
    """Same parameterisation as AC9 ``kepler_state`` but indexed by the
    eccentric anomaly, which avoids solving Kepler's equation and resolves
    periapsis uniformly for e -> 1."""
    anomaly = np.asarray(anomaly, dtype=np.float64)
    root = math.sqrt(1.0 - eccentricity**2)
    denominator = 1.0 - eccentricity * np.cos(anomaly)
    scale = math.sqrt(semi_major_axis)
    qx = semi_major_axis * (eccentricity - np.cos(anomaly))
    qy = -semi_major_axis * root * np.sin(anomaly)
    px = np.sin(anomaly) / (scale * denominator)
    py = -root * np.cos(anomaly) / (scale * denominator)
    return np.stack([qx, qy, px, py], axis=-1)


def solve_kepler_equation(mean_anomaly, eccentricity):
    """Robust scalar solve of E - e sin E = M by bisection on [M-1, M+1]."""
    mean_anomaly = float(mean_anomaly)
    lower, upper = mean_anomaly - 1.0, mean_anomaly + 1.0

    def residual(value):
        return value - eccentricity * math.sin(value) - mean_anomaly

    if residual(lower) > 0 or residual(upper) < 0:
        raise RuntimeError("Kepler bracket invalid")
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def radius_at_physical_time(scenario, physical_time):
    """Radius on the analytic reference orbit of an AC9 scenario at the
    physical time ``physical_time`` after its registered initial state."""
    a = scenario["semi_major_axis"]
    e = scenario["eccentricity"]
    mean_motion = a ** -1.5
    sign = -1.0 if scenario["momentum_direction"] == "reversed" else 1.0
    mean_anomaly = scenario["initial_mean_anomaly"] + sign * mean_motion * (
        physical_time
    )
    mean_anomaly = math.fmod(mean_anomaly, 2.0 * math.pi)
    if mean_anomaly < 0.0:
        mean_anomaly += 2.0 * math.pi
    anomaly = solve_kepler_equation(mean_anomaly, e)
    return float(a * (1.0 - e * math.cos(anomaly))), anomaly


def transform_states(states, orientation, direction):
    transformed = np.stack(
        [rotate_state(state, orientation) for state in states]
    )
    if direction == "reversed":
        transformed[:, 2:] *= -1.0
    return transformed


# ---------------------------------------------------------------------------
# contraction diagnostic
# ---------------------------------------------------------------------------
def manual_radius_monitor(z):
    return torch.linalg.vector_norm(z[:, :2], dim=1)


def batched_hessian(function, z):
    """Hessian of a batched scalar function, returned as ``(N, 4, 4)``."""
    z = z.detach().requires_grad_(True)
    value = function(z).sum()
    gradient = torch.autograd.grad(value, z, create_graph=True)[0]
    rows = []
    for component in range(z.shape[1]):
        row = torch.autograd.grad(
            gradient[:, component].sum(), z, retain_graph=True
        )[0]
        rows.append(row)
    return torch.stack(rows, dim=1).detach()


def orbit_diagnostic(monitor, system, states, dtau, batch=64):
    """Return per-state radius, monitor value, spectral radius and spectral
    norm of ``(dtau/2) J D2_zz K`` along the supplied orbit states."""
    states_t = torch.tensor(states, dtype=torch.float64)
    p0 = -system.H_torch(states_t[:1]).detach()

    def extended(z):
        return monitor(z) * (system.H_torch(z) + p0)

    radii, monitor_values, spectral_radii, spectral_norms = [], [], [], []
    for start in range(0, states_t.shape[0], batch):
        chunk = states_t[start:start + batch]
        hessian = batched_hessian(extended, chunk)
        operator = 0.5 * dtau * torch.matmul(SYMPLECTIC_J, hessian)
        eigenvalues = torch.linalg.eigvals(operator)
        spectral_radii.append(eigenvalues.abs().max(dim=1).values)
        spectral_norms.append(torch.linalg.matrix_norm(operator, ord=2))
        with torch.no_grad():
            monitor_values.append(monitor(chunk))
        radii.append(torch.linalg.vector_norm(chunk[:, :2], dim=1))
    return {
        "radius": torch.cat(radii).numpy(),
        "monitor": torch.cat(monitor_values).numpy(),
        "spectral_radius": torch.cat(spectral_radii).numpy(),
        "spectral_norm": torch.cat(spectral_norms).numpy(),
    }


def summarise_orbit(diagnostic, anomalies, semi_major_axis, eccentricity,
                    training_radius_minimum):
    radius = diagnostic["radius"]
    monitor = diagnostic["monitor"]
    rho = diagnostic["spectral_radius"]
    periapsis = int(np.argmin(radius))
    # physical-time weights: dt = a^{3/2} (1 - e cos E) dE on a uniform E grid
    weights = (1.0 - eccentricity * np.cos(anomalies))
    weights = weights / weights.sum()
    return {
        "radius_minimum": float(radius[periapsis]),
        "radius_maximum": float(radius.max()),
        "monitor_at_periapsis": float(monitor[periapsis]),
        "monitor_over_radius_at_periapsis": float(
            monitor[periapsis] / radius[periapsis]
        ),
        "monitor_minimum": float(monitor.min()),
        "monitor_maximum": float(monitor.max()),
        "spectral_radius_maximum": float(rho.max()),
        "spectral_radius_at_periapsis": float(rho[periapsis]),
        "spectral_radius_argmax_radius": float(radius[int(np.argmax(rho))]),
        "spectral_norm_maximum": float(diagnostic["spectral_norm"].max()),
        "orbit_time_fraction_below_training_radius": float(
            weights[radius < training_radius_minimum].sum()
        ),
        "states_below_training_radius": int(
            (radius < training_radius_minimum).sum()
        ),
        "sampled_states": int(radius.size),
    }


def load_monitor(training_record):
    monitor = Monitor(training_record["architecture"])
    package = torch.load(
        os.path.join(ROOT, training_record["checkpoint_path"]),
        map_location="cpu",
        weights_only=True,
    )
    monitor.load_state_dict(package["state_dict"])
    monitor.eval()
    return monitor


def orbit_families(ac9_protocol):
    evaluation = ac9_protocol["evaluation"]
    families = []
    for semi_major_axis in evaluation["semi_major_axes"]:
        for eccentricity in evaluation["eccentricities"]:
            for angle in evaluation["orientations_radians"]:
                for direction in evaluation["momentum_directions"]:
                    families.append(
                        {
                            "semi_major_axis": semi_major_axis,
                            "eccentricity": eccentricity,
                            "orientation_radians": angle,
                            "momentum_direction": direction,
                        }
                    )
    return families


def scenario_key(record):
    return (
        record["architecture"],
        record["checkpoint_seed"],
        record["semi_major_axis"],
        record["eccentricity"],
        record["initial_mean_anomaly"],
        record["orientation_radians"],
        record["momentum_direction"],
    )


def family_key(record):
    return (
        record["semi_major_axis"],
        record["eccentricity"],
        record["orientation_radians"],
        record["momentum_direction"],
    )


# ---------------------------------------------------------------------------
# phase A
# ---------------------------------------------------------------------------
def run_phase_a(protocol, ac9_protocol, ac9_training, ac9_raw, out_dir,
                monitor_limit=None, family_limit=None):
    config = protocol["phase_a"]
    dtau = config["dtau"]
    anomalies = np.linspace(
        0.0, 2.0 * math.pi, config["states_per_orbit"], endpoint=False
    )
    monitors = [("manual_r", None, manual_radius_monitor)]
    for record in ac9_training["records"]:
        monitors.append(
            (record["architecture"], record["checkpoint_seed"],
             load_monitor(record))
        )
    if monitor_limit:
        monitors = monitors[:monitor_limit]
    families = orbit_families(ac9_protocol)
    if family_limit:
        families = [
            family for family in families if family["eccentricity"] >= 0.9
        ][:family_limit]
    raw_path = os.path.join(out_dir, "raw_results.json")
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "phase": "A_diagnostic",
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "dtau": dtau,
        "states_per_orbit": config["states_per_orbit"],
        "training_radius_minimum": config["training_radius_minimum"],
        "orbit_records": [],
        "failure_locations": [],
    }
    if os.path.exists(raw_path):
        artifact = load_json(raw_path)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError("raw artifact protocol mismatch")
    existing = {
        (record["architecture"], record["checkpoint_seed"],
         *family_key(record))
        for record in artifact["orbit_records"]
    }
    for architecture, seed, monitor in monitors:
        for family in families:
            key = (architecture, seed, *family_key(family))
            if key in existing:
                continue
            system = Kepler2D(family["eccentricity"])
            base = state_from_eccentric_anomaly(
                family["semi_major_axis"], family["eccentricity"], anomalies
            )
            states = transform_states(
                base, family["orientation_radians"],
                family["momentum_direction"],
            )
            started = time.perf_counter()
            diagnostic = orbit_diagnostic(monitor, system, states, dtau)
            summary = summarise_orbit(
                diagnostic, anomalies, family["semi_major_axis"],
                family["eccentricity"], config["training_radius_minimum"],
            )
            record = {
                "architecture": architecture,
                "checkpoint_seed": seed,
                **family,
                **summary,
                "predicted_fixed_point_failure":
                    summary["spectral_radius_maximum"]
                    >= config["contraction_threshold"],
                "wall_seconds": time.perf_counter() - started,
            }
            if (
                config["store_profiles"]
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
            artifact["orbit_records"].append(record)
            atomic_json(raw_path, artifact)
            print(
                f"{architecture} seed={seed} a={family['semi_major_axis']} "
                f"e={family['eccentricity']} "
                f"angle={family['orientation_radians']:.3f} "
                f"{family['momentum_direction']}: "
                f"rho_max={summary['spectral_radius_maximum']:.3f} "
                f"g/r(peri)={summary['monitor_over_radius_at_periapsis']:.2f}",
                flush=True,
            )
    if not artifact["failure_locations"]:
        for record in ac9_raw["records"]:
            if record["completion_status"] == "complete":
                continue
            radius, anomaly = radius_at_physical_time(
                record, record["physical_time"]
            )
            artifact["failure_locations"].append(
                {
                    **{k: record[k] for k in (
                        "architecture", "checkpoint_seed",
                        "semi_major_axis", "eccentricity",
                        "initial_mean_anomaly", "orientation_radians",
                        "momentum_direction", "completion_status",
                        "physical_time", "steps",
                    )},
                    "failure_radius": radius,
                    "failure_eccentric_anomaly": anomaly,
                    "failure_radius_over_periapsis": radius / (
                        record["semi_major_axis"]
                        * (1.0 - record["eccentricity"])
                    ),
                    "orbit_period": 2.0 * math.pi
                    * record["semi_major_axis"] ** 1.5,
                }
            )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(raw_path, artifact)
    return artifact


# ---------------------------------------------------------------------------
# phase B
# ---------------------------------------------------------------------------
def run_phase_b(protocol, ac9_protocol, ac9_training, ac9_raw, phase_a,
                out_dir, rollout_limit=None):
    config = protocol["phase_b"]
    dtau = protocol["phase_a"]["dtau"]
    rho_by_key = {
        (record["architecture"], record["checkpoint_seed"],
         *family_key(record)): record["spectral_radius_maximum"]
        for record in phase_a["orbit_records"]
    }
    monitors = {
        (record["architecture"], record["checkpoint_seed"]):
            load_monitor(record)
        for record in ac9_training["records"]
    }
    failed = [
        record for record in ac9_raw["records"]
        if record["completion_status"] != "complete"
        and record["architecture"] in ("raw", "invariant")
    ]
    if rollout_limit:
        failed = failed[:rollout_limit]
    intervention_path = os.path.join(out_dir, "intervention.json")
    artifact = {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "phase": "B_intervention",
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": config["rule"],
        "rho_target": config["rho_target"],
        "records": [],
    }
    if os.path.exists(intervention_path):
        artifact = load_json(intervention_path)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError("intervention artifact protocol mismatch")
    existing = {scenario_key(record) for record in artifact["records"]}
    evaluation = dict(ac9_protocol["evaluation"])
    evaluation["maximum_steps"] = config["maximum_steps"]
    for original in failed:
        key = scenario_key(original)
        if key in existing:
            continue
        rho_max = rho_by_key[
            (original["architecture"], original["checkpoint_seed"],
             *family_key(original))
        ]
        factor = min(1.0, config["rho_target"] / rho_max)
        evaluation["dtau"] = dtau * factor
        scenario = {
            k: original[k] for k in (
                "semi_major_axis", "eccentricity", "initial_mean_anomaly",
                "orientation_radians", "momentum_direction",
            )
        }
        record = run_rollout(
            monitors[(original["architecture"], original["checkpoint_seed"])],
            original["architecture"],
            original["checkpoint_seed"],
            scenario,
            evaluation,
        )
        record.update(
            original_completion_status=original["completion_status"],
            original_dtau=dtau,
            original_steps=original["steps"],
            original_vector_field_evaluations=
                original["vector_field_evaluations"],
            spectral_radius_maximum_at_original_dtau=rho_max,
            step_factor=factor,
            intervention_dtau=evaluation["dtau"],
            predicted_spectral_radius_maximum=rho_max * factor,
            rule_prescribes_change=factor < 1.0,
        )
        artifact["records"].append(record)
        atomic_json(intervention_path, artifact)
        print(
            f"[B] {original['architecture']} seed={original['checkpoint_seed']} "
            f"a={scenario['semi_major_axis']} e={scenario['eccentricity']} "
            f"M={scenario['initial_mean_anomaly']:.3f} "
            f"angle={scenario['orientation_radians']:.3f} "
            f"{scenario['momentum_direction']}: dtau*={evaluation['dtau']:.5f} "
            f"-> {record['completion_status']} steps={record['steps']}",
            flush=True,
        )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(intervention_path, artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-dir", default=None)
    parser.add_argument("--skip-phase-b", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    ac9_protocol = load_json(os.path.join(AC9_DIR, "preregistration.json"))
    ac9_training = load_json(os.path.join(AC9_DIR, "training.json"))
    ac9_raw = load_json(os.path.join(AC9_DIR, "raw_results.json"))
    if args.dry_run:
        out_dir = args.dry_run_dir or os.path.join(
            os.environ.get("TEMP", "/tmp"), "expAC19_dry_run"
        )
        os.makedirs(out_dir, exist_ok=True)
        protocol = {
            "experiment": "expAC19_dry_run",
            "integrity": {"protocol_sha256": "dry-run"},
            "phase_a": {
                "dtau": 0.04, "states_per_orbit": 90,
                "training_radius_minimum": 0.1,
                "contraction_threshold": 1.0, "store_profiles": True,
                "profile_stride": 2,
            },
            "phase_b": {
                "rule": "dtau* = dtau * min(1, rho_target / rho_max)",
                "rho_target": 0.5, "maximum_steps": 2_000_000,
            },
        }
        phase_a = run_phase_a(
            protocol, ac9_protocol, ac9_training, ac9_raw, out_dir,
            monitor_limit=2, family_limit=2,
        )
        # restrict to failures whose family was diagnosed in the subset and
        # write the subset plus a mock protocol so analyze_expAC19.py can be
        # exercised with AC19_OUT pointing at this directory
        diagnosed = {
            (r["architecture"], r["checkpoint_seed"], *family_key(r))
            for r in phase_a["orbit_records"]
        }
        ac9_raw = dict(ac9_raw)
        ac9_raw["records"] = [
            r for r in ac9_raw["records"]
            if (r["architecture"], r["checkpoint_seed"], *family_key(r))
            in diagnosed
        ]
        atomic_json(os.path.join(out_dir, "ac9_subset.json"), ac9_raw)
        failed_learned = sum(
            r["completion_status"] != "complete"
            and r["architecture"] != "manual_r" for r in ac9_raw["records"]
        )
        phase_a["failure_locations"] = [
            loc for loc in phase_a["failure_locations"]
            if (loc["architecture"], loc["checkpoint_seed"], *family_key(loc))
            in diagnosed
        ]
        atomic_json(os.path.join(out_dir, "raw_results.json"), phase_a)
        protocol["acceptance"] = {
            "expected_orbit_records": len(phase_a["orbit_records"]),
            "expected_failed_rollouts": failed_learned,
            "expected_ac9_rollouts": len(ac9_raw["records"]),
        }
        atomic_json(os.path.join(out_dir, "preregistration.json"), protocol)
        if not args.skip_phase_b:
            run_phase_b(
                protocol, ac9_protocol, ac9_training, ac9_raw, phase_a,
                out_dir,
            )
        print(f"dry run written to {out_dir}")
        return
    out_dir = DEFAULT_OUT
    protocol = verify_protocol(os.path.join(out_dir, "preregistration.json"))
    phase_a = run_phase_a(
        protocol, ac9_protocol, ac9_training, ac9_raw, out_dir
    )
    if not args.skip_phase_b:
        run_phase_b(
            protocol, ac9_protocol, ac9_training, ac9_raw, phase_a, out_dir
        )


if __name__ == "__main__":
    main()
