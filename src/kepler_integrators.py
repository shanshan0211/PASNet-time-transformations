"""Validated low-level integrators for the planar Kepler Hamiltonian.

The canonical Hamiltonian is

    H(q, p) = |p|^2 / 2 - 1 / |q|,

so ``grad_q H = +q / |q|^3`` and Hamilton's equation gives
``dp/dt = -q / |q|^3``.  Keeping this sign convention in one module avoids
the duplicated sign errors that previously invalidated the direct-midpoint
controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MidpointInfo:
    converged: bool
    iterations: int
    residual: float


def kepler_energy(q: np.ndarray, p: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(0.5 * p @ p - 1.0 / np.linalg.norm(q))


def kepler_gradient(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(grad_q H, grad_p H)`` for the attractive Kepler problem."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    r = np.linalg.norm(q)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("Kepler gradient is undefined at r <= 0")
    return q / r**3, p.copy()


def kepler_vector_field(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Hamilton's equations ``(dq/dt, dp/dt)``."""
    grad_q, grad_p = kepler_gradient(q, p)
    return grad_p, -grad_q


def implicit_midpoint_step(
    q: np.ndarray,
    p: np.ndarray,
    dt: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-13,
    require_convergence: bool = True,
) -> tuple[np.ndarray, np.ndarray, MidpointInfo]:
    """Advance one fixed-point implicit-midpoint step.

    The returned residual is the infinity norm of the fixed-point update.
    A non-converged nonlinear solve is never silently treated as a valid
    symplectic step when ``require_convergence`` is true.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    dq, dp = kepler_vector_field(q, p)
    qn, pn = q + dt * dq, p + dt * dp
    residual = np.inf

    for iteration in range(1, max_iter + 1):
        qm = 0.5 * (q + qn)
        pm = 0.5 * (p + pn)
        dq, dp = kepler_vector_field(qm, pm)
        q_next = q + dt * dq
        p_next = p + dt * dp
        residual = float(
            max(np.max(np.abs(q_next - qn)), np.max(np.abs(p_next - pn)))
        )
        qn, pn = q_next, p_next
        if residual < tol:
            return qn, pn, MidpointInfo(True, iteration, residual)

    info = MidpointInfo(False, max_iter, residual)
    if require_convergence:
        raise RuntimeError(
            f"implicit midpoint did not converge in {max_iter} iterations "
            f"(residual={residual:.3e}, dt={dt:.3e})"
        )
    return qn, pn, info
