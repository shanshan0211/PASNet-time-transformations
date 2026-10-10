"""Detached conditioning diagnostics for planar Kepler monitor steps."""

from __future__ import annotations

import math
from typing import Callable

import torch


Tensor = torch.Tensor
ScalarField = Callable[[Tensor], Tensor]


def kepler_shell_scaling(
    semi_major_axis: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``S=diag(a,a,a^-1/2,a^-1/2)`` and its inverse."""
    if not math.isfinite(semi_major_axis) or semi_major_axis <= 0.0:
        raise ValueError("semi_major_axis must be positive and finite")
    a = float(semi_major_axis)
    diagonal = torch.tensor(
        [a, a, a ** -0.5, a ** -0.5], dtype=dtype, device=device
    )
    return torch.diag(diagonal), torch.diag(1.0 / diagonal)


def scaled_step_size(step_size: float, semi_major_axis: float) -> float:
    """Return orbital-time-normalized ``h a^-3/2``.

    This quantity is not the coefficient in the similarity-transformed
    endpoint Jacobian.  It remains available only as a descriptive time-scale
    diagnostic for compatibility with the historical AC11 code.
    """
    return float(step_size) * float(semi_major_axis) ** -1.5


def canonical_symplectic_matrix(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    identity = torch.eye(2, dtype=dtype, device=device)
    zero = torch.zeros((2, 2), dtype=dtype, device=device)
    return torch.cat(
        [torch.cat([zero, identity], dim=1), torch.cat([-identity, zero], dim=1)],
        dim=0,
    )


def _scalar(function: ScalarField, flat_state: Tensor) -> Tensor:
    value = function(flat_state.reshape(1, 4))
    if value.numel() != 1:
        raise ValueError("monitor and Hamiltonian must return one scalar per state")
    return value.reshape(())


def _gradient_hessian(
    function: ScalarField, flat_state: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    value = _scalar(function, flat_state)
    gradient = torch.autograd.grad(value, flat_state, create_graph=True)[0]
    hessian = torch.autograd.functional.hessian(
        lambda value_: _scalar(function, value_),
        flat_state,
        create_graph=False,
        vectorize=True,
    )
    return value, gradient, hessian


def _number(value: Tensor) -> float:
    return float(value.detach().cpu())


def _matrix(value: Tensor) -> list[list[float]]:
    return value.detach().cpu().numpy().tolist()


def nondimensional_endpoint_diagnostics(
    d2_k: Tensor,
    step_size: float,
    semi_major_axis: float,
) -> dict:
    """Compute the correctly transformed endpoint-Jacobian diagnostics.

    With ``z=S z_tilde`` and residual
    ``R(z_+)=z_+-z-h J grad K((z+z_+)/2)``, define
    ``R_tilde(z_tilde_+)=S^-1 R(S z_tilde_+)``.  The chain rule gives

    ``D R_tilde = S^-1 (D R) S
                 = I-(h/2) S^-1 J D2K S``.

    The coefficient is the original fictitious step ``h``.  Multiplying it by
    an additional ``a^-3/2`` is incorrect because the state-coordinate
    similarity transform does not reparameterize time.
    """
    if d2_k.shape != (4, 4):
        raise ValueError("D2K must be a 4x4 planar-state Hessian")
    if d2_k.dtype != torch.float64:
        raise TypeError("endpoint diagnostics require float64 input")
    if not math.isfinite(step_size) or step_size < 0.0:
        raise ValueError("step_size must be nonnegative and finite")
    scaling, inverse_scaling = kepler_shell_scaling(
        semi_major_axis, dtype=d2_k.dtype, device=d2_k.device
    )
    j = canonical_symplectic_matrix(dtype=d2_k.dtype, device=d2_k.device)
    scaled_operator = inverse_scaling @ j @ d2_k @ scaling
    scaled_norm = torch.linalg.matrix_norm(scaled_operator, ord=2)
    coefficient = 0.5 * float(step_size)
    endpoint_jacobian = (
        torch.eye(4, dtype=d2_k.dtype, device=d2_k.device)
        - coefficient * scaled_operator
    )
    singular_values = torch.linalg.svdvals(endpoint_jacobian)
    return {
        "operator_definition": "S^-1 J D2zzK S",
        "endpoint_jacobian_formula":
            "I-(h/2) S^-1 J D2zzK S",
        "step_coefficient": float(step_size),
        "scaled_J_D2zzK_norm_2": _number(scaled_norm),
        "contraction_q": _number(coefficient * scaled_norm),
        "scaled_endpoint_jacobian_singular_values": [
            float(value) for value in singular_values.detach().cpu()
        ],
        "scaled_endpoint_jacobian_sigma_min": _number(
            torch.min(singular_values)
        ),
        "scaled_endpoint_jacobian_sigma_max": _number(
            torch.max(singular_values)
        ),
        "scaled_endpoint_jacobian_condition_2": _number(
            torch.max(singular_values) / torch.min(singular_values)
        ),
    }


def conditioning_diagnostics(
    midpoint: Tensor,
    monitor: ScalarField,
    hamiltonian: ScalarField,
    p0: Tensor | float,
    step_size: float,
    semi_major_axis: float,
) -> dict:
    """Evaluate scaled and raw diagnostics, returning graph-free Python data.

    Primary scaled quantities use
    ``A~=S^-1 J D2K S``, ``q=(h/2)||A~||_2``, and
    ``D R~=I-(h/2)A~``.  There is no additional time-rescaling factor in this
    state-coordinate similarity transform.
    Raw-coordinate quantities are retained as explicitly secondary metrics.
    """
    if midpoint.numel() != 4:
        raise ValueError("conditioning diagnostics require one planar state")
    if midpoint.dtype != torch.float64:
        raise TypeError("conditioning diagnostics require float64 input")
    flat = midpoint.detach().reshape(4).clone().requires_grad_(True)
    p0_tensor = torch.as_tensor(p0, dtype=flat.dtype, device=flat.device).reshape(())

    g_value, grad_g, hessian_g = _gradient_hessian(monitor, flat)
    h_value, grad_h, hessian_h = _gradient_hessian(hamiltonian, flat)
    defect = h_value + p0_tensor
    d2_k = (
        g_value * hessian_h
        + torch.outer(grad_g, grad_h)
        + torch.outer(grad_h, grad_g)
        + defect * hessian_g
    )
    j = canonical_symplectic_matrix(dtype=flat.dtype, device=flat.device)
    raw_operator = j @ d2_k
    scaling, _ = kepler_shell_scaling(
        semi_major_axis, dtype=flat.dtype, device=flat.device
    )
    corrected = nondimensional_endpoint_diagnostics(
        d2_k, step_size, semi_major_axis
    )
    raw_norm = torch.linalg.matrix_norm(raw_operator, ord=2)
    raw_endpoint_jacobian = (
        torch.eye(4, dtype=flat.dtype, device=flat.device)
        - 0.5 * float(step_size) * raw_operator
    )
    raw_singular_values = torch.linalg.svdvals(raw_endpoint_jacobian)
    scaled_grad_g = scaling.T @ grad_g
    scaled_hessian_g = scaling.T @ hessian_g @ scaling
    scaled_hessian_h = scaling.T @ hessian_h @ scaling

    # Every returned object is detached here; no autograd graph escapes.
    return {
        "scaling": {
            "semi_major_axis": float(semi_major_axis),
            "S_diagonal": [float(v) for v in torch.diag(scaling).detach().cpu()],
            "step_coefficient": float(step_size),
            "operator_definition": "S^-1 J D2zzK S",
            "endpoint_jacobian_formula":
                "I-(h/2) S^-1 J D2zzK S",
            "matrix_norm": "spectral_2_norm",
        },
        "g": _number(g_value),
        "H": _number(h_value),
        "off_surface_defect_abs": abs(_number(defect)),
        "grad_g_scaled_2": _number(torch.linalg.vector_norm(scaled_grad_g)),
        "hessian_g_scaled_2": _number(
            torch.linalg.matrix_norm(scaled_hessian_g, ord=2)
        ),
        "hessian_H_scaled_2": _number(
            torch.linalg.matrix_norm(scaled_hessian_h, ord=2)
        ),
        "D2zzK": _matrix(d2_k),
        "scaled_J_D2zzK_norm_2":
            corrected["scaled_J_D2zzK_norm_2"],
        "contraction_q": corrected["contraction_q"],
        "scaled_endpoint_jacobian_singular_values":
            corrected["scaled_endpoint_jacobian_singular_values"],
        "scaled_endpoint_jacobian_sigma_min":
            corrected["scaled_endpoint_jacobian_sigma_min"],
        "scaled_endpoint_jacobian_sigma_max":
            corrected["scaled_endpoint_jacobian_sigma_max"],
        "scaled_endpoint_jacobian_condition_2":
            corrected["scaled_endpoint_jacobian_condition_2"],
        "raw_secondary": {
            "grad_g_2": _number(torch.linalg.vector_norm(grad_g)),
            "hessian_g_2": _number(
                torch.linalg.matrix_norm(hessian_g, ord=2)
            ),
            "grad_H_2": _number(torch.linalg.vector_norm(grad_h)),
            "hessian_H_2": _number(
                torch.linalg.matrix_norm(hessian_h, ord=2)
            ),
            "D2zzK_2": _number(torch.linalg.matrix_norm(d2_k, ord=2)),
            "J_D2zzK_norm_2": _number(raw_norm),
            "endpoint_jacobian_singular_values": [
                float(value) for value in raw_singular_values.detach().cpu()
            ],
            "endpoint_jacobian_condition_2": _number(
                torch.max(raw_singular_values)
                / torch.min(raw_singular_values)
            ),
            "off_surface_defect_signed": _number(defect),
            "off_surface_defect_relative_to_abs_H": abs(_number(defect))
            / max(abs(_number(h_value)), torch.finfo(flat.dtype).tiny),
        },
    }
