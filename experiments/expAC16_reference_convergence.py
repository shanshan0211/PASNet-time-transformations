"""Run sealed AC16 convergence checks for the references used by AC13."""

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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "expAC16")
PROTOCOL_PATH = os.path.join(OUT, "preregistration.json")
RAW_PATH = os.path.join(OUT, "raw_results.json")
SUMMARY_PATH = os.path.join(OUT, "summary.json")
sys.path.insert(0, os.path.join(ROOT, "src"))

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
        raise RuntimeError("seal AC16 with register_expAC16.py before execution")
    with open(PROTOCOL_PATH, encoding="utf-8") as handle:
        protocol = json.load(handle)
    if canonical_protocol_hash(protocol) != protocol["integrity"]["protocol_sha256"]:
        raise RuntimeError("AC16 protocol hash mismatch")
    for source in protocol["frozen_sources"]:
        if sha256_file(os.path.join(ROOT, source["path"])) != source["sha256"]:
            raise RuntimeError(f"frozen source changed: {source['path']}")
    for artifact in protocol["frozen_ac13_artifacts"]:
        if sha256_file(os.path.join(ROOT, artifact["path"])) != artifact["sha256"]:
            raise RuntimeError(f"AC13 artifact changed: {artifact['path']}")
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


def initial_state(system_name, energy, seed, *, kappa=None):
    system = build_system(system_name, kappa=kappa)
    q, p = system.initial_condition(e0=energy, seed=seed)
    return system, np.concatenate([q, p]).astype(np.float64)


def energy(system_name, system, state):
    d = dimension(system_name)
    if system_name == "modal":
        return float(system.energy(state[:d], state[d:]))
    return float(np.asarray(system.H(state[:d], state[d:])).squeeze())


def force(system_name, system, q):
    if system_name == "modal":
        return -(system.omegas**2 * q + system.beta * q**3)
    return -system.grad_x(q)


def velocity_verlet_trajectory(system_name, system, state0, horizon, dt):
    """Return every endpoint, including a shortened final step."""
    d = dimension(system_name)
    q = state0[:d].copy()
    p = state0[d:].copy()
    n_steps = int(math.ceil(horizon / dt))
    times = np.empty(n_steps + 1, dtype=np.float64)
    states = np.empty((n_steps + 1, state0.size), dtype=np.float64)
    times[0], states[0] = 0.0, state0
    physical_time = 0.0
    for step in range(n_steps):
        h = min(dt, horizon - physical_time)
        p_half = p + 0.5 * h * force(system_name, system, q)
        q = q + h * p_half
        p = p_half + 0.5 * h * force(system_name, system, q)
        physical_time += h
        times[step + 1] = physical_time
        states[step + 1, :d] = q
        states[step + 1, d:] = p
    return times, states


def interpolated_state_discrepancy(
    coarse_times, coarse_states, fine_times, fine_states, chunk_size=16384
):
    """Compare a linearly interpolated coarse reference on the fine grid."""
    maximum = 0.0
    final = None
    for start in range(0, len(fine_times), chunk_size):
        stop = min(start + chunk_size, len(fine_times))
        query = fine_times[start:stop]
        upper = np.searchsorted(coarse_times, query, side="right")
        upper = np.clip(upper, 1, len(coarse_times) - 1)
        lower = upper - 1
        denominator = coarse_times[upper] - coarse_times[lower]
        weight = (query - coarse_times[lower]) / denominator
        interpolated = coarse_states[lower] + weight[:, None] * (
            coarse_states[upper] - coarse_states[lower]
        )
        norms = np.linalg.norm(interpolated - fine_states[start:stop], axis=1)
        maximum = max(maximum, float(np.max(norms)))
        if stop == len(fine_times):
            final = float(norms[-1])
    return {"maximum": maximum, "final": final}


def observed_order(coarse_difference, fine_difference):
    if coarse_difference <= 0.0 or fine_difference <= 0.0:
        return None
    value = math.log(coarse_difference / fine_difference, 2.0)
    return value if math.isfinite(value) else None


def effective_order(value):
    """Bound extrapolation sensitivity when empirical order is irregular."""
    if value is None or value <= 0.0:
        return 0.5, "fallback_nonpositive_or_undefined"
    if value > 4.0:
        return 4.0, "observed_clipped_above_at_four"
    return value, "observed"


def richardson_uncertainty(level_difference, order, *, fine_side):
    used_order, rule = effective_order(order)
    divisor = (
        2.0**used_order - 1.0
        if fine_side
        else 1.0 - 2.0 ** (-used_order)
    )
    return {
        "absolute": float(level_difference / divisor),
        "effective_order": used_order,
        "order_rule": rule,
    }


def scenarios(protocol, system_name):
    config = protocol["evaluation"][system_name]
    result = []
    if system_name == "modal":
        for shell in config["energy_shells"]:
            for seed in config["initial_condition_seeds"]:
                for kappa in config["kappas"]:
                    for direction in config["momentum_directions"]:
                        result.append(
                            {
                                "system": system_name,
                                "energy_shell": shell,
                                "initial_condition_seed": seed,
                                "kappa": kappa,
                                "momentum_direction": direction,
                            }
                        )
    else:
        for shell in config["energy_shells"]:
            for seed in config["initial_condition_seeds"]:
                for direction in config["momentum_directions"]:
                    result.append(
                        {
                            "system": system_name,
                            "energy_shell": shell,
                            "initial_condition_seed": seed,
                            "kappa": None,
                            "momentum_direction": direction,
                        }
                    )
    return result


def scenario_key(record):
    return (
        record["system"],
        record["energy_shell"],
        record["initial_condition_seed"],
        record.get("kappa"),
        record["momentum_direction"],
    )


def energy_metrics(system_name, system, states, initial_energy):
    values = np.asarray([energy(system_name, system, state) for state in states])
    errors = np.abs(values - initial_energy)
    return {
        "maximum_absolute_drift": float(np.max(errors)),
        "final_absolute_drift": float(errors[-1]),
    }


def run_scenario(protocol, scenario):
    config = protocol["evaluation"][scenario["system"]]
    system, state0 = initial_state(
        scenario["system"],
        scenario["energy_shell"],
        scenario["initial_condition_seed"],
        kappa=scenario.get("kappa"),
    )
    if scenario["momentum_direction"] == "reversed":
        state0[dimension(scenario["system"]) :] *= -1
    horizon = (
        config["slow_periods"] * 2.0 * math.pi
        if scenario["system"] == "modal"
        else config["physical_horizon"]
    )
    trajectories = {}
    initial_energy = energy(scenario["system"], system, state0)
    started = time.perf_counter()
    for dt in protocol["reference"]["dt_values"]:
        times, states = velocity_verlet_trajectory(
            scenario["system"], system, state0, horizon, dt
        )
        trajectories[dt] = (times, states)

    dt_coarse, dt_ac13, dt_fine = protocol["reference"]["dt_values"]
    coarse_ac13 = interpolated_state_discrepancy(
        *trajectories[dt_coarse], *trajectories[dt_ac13]
    )
    ac13_fine = interpolated_state_discrepancy(
        *trajectories[dt_ac13], *trajectories[dt_fine]
    )
    state_order = {
        field: observed_order(coarse_ac13[field], ac13_fine[field])
        for field in ["maximum", "final"]
    }
    state_uncertainty = {
        "ac13_dt": {
            field: richardson_uncertainty(
                ac13_fine[field], state_order[field], fine_side=False
            )
            for field in ["maximum", "final"]
        },
        "finest_dt": {
            field: richardson_uncertainty(
                ac13_fine[field], state_order[field], fine_side=True
            )
            for field in ["maximum", "final"]
        },
    }
    level_energy = {
        str(dt): energy_metrics(
            scenario["system"], system, trajectories[dt][1], initial_energy
        )
        for dt in protocol["reference"]["dt_values"]
    }
    energy_order = {}
    energy_uncertainty = {}
    for field in ["maximum_absolute_drift", "final_absolute_drift"]:
        order = observed_order(
            level_energy[str(dt_coarse)][field],
            level_energy[str(dt_ac13)][field],
        )
        energy_order[field] = order
        energy_uncertainty[field] = richardson_uncertainty(
            level_energy[str(dt_fine)][field], order, fine_side=True
        )
    return {
        **scenario,
        "completion_status": "complete",
        "horizon": horizon,
        "dt_values": protocol["reference"]["dt_values"],
        "steps_by_dt": {
            str(dt): len(trajectories[dt][0]) - 1
            for dt in protocol["reference"]["dt_values"]
        },
        "state_level_differences": {
            f"{dt_coarse}_vs_{dt_ac13}": coarse_ac13,
            f"{dt_ac13}_vs_{dt_fine}": ac13_fine,
        },
        "state_observed_order": state_order,
        "state_reference_uncertainty": state_uncertainty,
        "energy_error_by_dt": level_energy,
        "energy_observed_order": energy_order,
        "energy_reference_uncertainty_at_finest_dt": energy_uncertainty,
        "wall_seconds": time.perf_counter() - started,
    }


def summarize(protocol, raw):
    records = raw["records"]
    finite_uncertainty = all(
        record["state_reference_uncertainty"]["ac13_dt"][field]["absolute"] > 0
        and math.isfinite(
            record["state_reference_uncertainty"]["ac13_dt"][field]["absolute"]
        )
        for record in records
        for field in ["maximum", "final"]
    )
    acceptance = {
        "all_registered_scenarios_retained":
            len(records) == protocol["evaluation"]["total_scenarios"],
        "all_scenarios_complete":
            all(record["completion_status"] == "complete" for record in records),
        "all_dt_levels_retained": all(
            record["dt_values"] == protocol["reference"]["dt_values"]
            for record in records
        ),
        "nonzero_finite_state_uncertainty": finite_uncertainty,
        "ac13_artifact_hashes_valid": True,
        "source_hashes_valid": True,
    }
    acceptance["artifact_gate_passed"] = all(acceptance.values())
    return {
        "schema_version": "1.0",
        "experiment": protocol["experiment"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol["integrity"]["protocol_sha256"],
        "counts": {"registered": protocol["evaluation"]["total_scenarios"],
                   "completed": len(records)},
        "acceptance": acceptance,
        "raw_results_sha256": sha256_file(RAW_PATH),
        "claim_limits": protocol["claim_limits"],
    }


def run_full():
    protocol = verify_protocol()
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
        },
        "records": [],
    }
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact["protocol_sha256"] != protocol["integrity"]["protocol_sha256"]:
            raise RuntimeError("raw artifact protocol mismatch")
    existing = {scenario_key(record) for record in artifact["records"]}
    for system_name in ["modal", "duffing"]:
        for scenario in scenarios(protocol, system_name):
            if scenario_key(scenario) in existing:
                continue
            record = run_scenario(protocol, scenario)
            artifact["records"].append(record)
            existing.add(scenario_key(record))
            atomic_json(RAW_PATH, artifact)
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RAW_PATH, artifact)
    summary = summarize(protocol, artifact)
    atomic_json(SUMMARY_PATH, summary)
    if not summary["acceptance"]["artifact_gate_passed"]:
        raise SystemExit("AC16 artifact gate failed")


def smoke():
    for system_name in ["modal", "duffing"]:
        system, state = initial_state(system_name, 1.0, 0, kappa=10.0)
        times, states = velocity_verlet_trajectory(
            system_name, system, state, 0.01, 0.001
        )
        assert len(times) == 11
        assert np.isfinite(states).all()
    print("AC16 smoke passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true", help="run a non-writing short check"
    )
    arguments = parser.parse_args()
    if arguments.smoke:
        smoke()
    else:
        run_full()


if __name__ == "__main__":
    main()
