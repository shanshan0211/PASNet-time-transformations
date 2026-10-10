"""Symplectic integrators in modal coordinates, batch-vectorized.

Velocity Verlet (kick-drift-kick leapfrog) for separable
H(u, v) = 1/2 v^T v + V(u),  V(u) = 1/2 sum_i om2_i u_i^2 + (beta/4) sum_i u_i^4.

An optional smooth perturbation force models the gradient error of a learned
Hamiltonian H_theta = H + eps*H1: the integrator conserves (a shadow of)
H_theta, and the drift of the TRUE energy H is what we measure.
"""
import numpy as np


def leapfrog_modal(om2, beta, dt, n_steps, u0, v0, energy_fn,
                   extra_force=None, n_records=2000, explode_at=1e6):
    """Velocity Verlet, vectorized over a batch of states.

    Parameters
    ----------
    om2 : (n,) array, squared eigen-frequencies.
    beta : float, quartic nonlinearity strength.
    dt : (B, 1) per-row step sizes.
    n_steps : int or (B,) per-row step counts; a row freezes once i > its own
        count (so rows can share one physical horizon with different dt).
    u0, v0 : (B, n) batch of initial states.
    energy_fn : callable(u, v) -> (B,), energy to RECORD (the true energy).
    extra_force : callable(u) -> (B, n), perturbation force added to -grad V.
    Returns
    -------
    ts : (n_rec,) step indices of records.
    Es : (n_rec, B) recorded energies (NaN for exploded rows).
    exploded : (B,) bool mask.
    """
    u = np.array(u0, dtype=np.float64)
    v = np.array(v0, dtype=np.float64)
    B = u.shape[0]
    dt = np.asarray(dt, dtype=np.float64).reshape(B, 1)
    if np.isscalar(n_steps):
        n_steps = np.full(B, int(n_steps), dtype=np.int64)
    else:
        n_steps = np.asarray(n_steps, dtype=np.int64)
    max_steps = int(n_steps.max())

    def force(x):
        f = -om2 * x
        if beta != 0.0:
            f = f - beta * x ** 3
        if extra_force is not None:
            f = f + extra_force(x)
        return np.nan_to_num(f, nan=0.0)  # keep exploded rows from poisoning

    record_every = max(1, max_steps // n_records)
    amp0 = np.max(np.abs(u), axis=1) + 1e-30
    exploded = np.zeros(B, dtype=bool)

    ts, Es = [0], [energy_fn(u, v).copy()]

    f = force(u)
    v = v + 0.5 * dt * f  # force() returns -grad V, the physical force
    for i in range(1, max_steps + 1):
        active = ((i <= n_steps) & ~exploded).astype(np.float64)[:, None]
        d = dt * active
        u += d * v
        f = force(u)
        v += d * f
        if i % record_every == 0 or i == max_steps:
            ts.append(i)
            # recover integer-step momentum for recording; use the row's own
            # dt (not the active-masked d) so frozen rows record correctly too
            v_sync = v - 0.5 * dt * f
            Es.append(energy_fn(u, v_sync).copy())
        if i % 512 == 0:
            blown = (np.nanmax(np.abs(u), axis=1) > explode_at * amp0) & ~exploded
            if np.any(blown):
                exploded |= blown
                u[blown] = np.nan
                v[blown] = np.nan
            if bool(np.all(exploded)):
                break
    return np.asarray(ts, dtype=np.float64), np.asarray(Es), exploded
