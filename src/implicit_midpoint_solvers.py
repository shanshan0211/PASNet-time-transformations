"""Residual-safe endpoint solvers for implicit midpoint.

The historical AC8/AC9 runners stopped fixed-point iteration on an update
norm.  Solvers in this module always evaluate the defining endpoint residual
at the endpoint they return and never report convergence unless that residual
is below tolerance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import torch


Tensor = torch.Tensor
VectorField = Callable[[Tensor], Tensor]


def _inf_norm(value: Tensor) -> float:
    return float(torch.max(torch.abs(value.detach())).cpu())


def endpoint_residual(
    state: Tensor,
    endpoint: Tensor,
    step_size: float,
    vector_field: VectorField,
) -> Tensor:
    """Return ``R_z = z_+ - z - h f((z + z_+)/2)``."""
    return endpoint - state - step_size * vector_field(0.5 * (state + endpoint))


@dataclass
class SolverDiagnostics:
    solver: str
    converged: bool = False
    iterations: int = 0
    vector_field_evaluations: int = 0
    jacobian_evaluations: int = 0
    backtracks: int = 0
    stopping_update_inf: float | None = None
    returned_endpoint_residual_inf: float | None = None
    failure_reason: str | None = None
    update_history_inf: list[float] = field(default_factory=list)
    residual_history_inf: list[float] = field(default_factory=list)
    accepted_damping_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ImplicitMidpointSolverError(RuntimeError):
    """Base failure carrying the final iterate and structured diagnostics."""

    def __init__(
        self,
        message: str,
        diagnostics: SolverDiagnostics,
        endpoint: Tensor,
    ):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.endpoint = endpoint


class NonFiniteIterateError(ImplicitMidpointSolverError):
    pass


class IterationLimitError(ImplicitMidpointSolverError):
    pass


class SingularJacobianError(ImplicitMidpointSolverError):
    pass


class LineSearchError(ImplicitMidpointSolverError):
    pass


def fixed_point_endpoint(
    state: Tensor,
    step_size: float,
    vector_field: VectorField,
    *,
    tolerance: float = 1e-12,
    maximum_iterations: int = 100,
    initial_endpoint: Tensor | None = None,
) -> tuple[Tensor, SolverDiagnostics]:
    """Solve by fixed point, accepting only a verified endpoint residual."""
    diagnostics = SolverDiagnostics(solver="fixed_point")
    if initial_endpoint is None:
        trial = state + step_size * vector_field(state)
        diagnostics.vector_field_evaluations += 1
    else:
        trial = initial_endpoint

    for iteration in range(1, maximum_iterations + 1):
        proposal = state + step_size * vector_field(0.5 * (state + trial))
        diagnostics.vector_field_evaluations += 1
        update_inf = _inf_norm(proposal - trial)
        diagnostics.update_history_inf.append(update_inf)
        diagnostics.stopping_update_inf = update_inf
        diagnostics.iterations = iteration
        trial = proposal
        if not bool(torch.isfinite(trial.detach()).all()):
            diagnostics.failure_reason = "nonfinite_iterate"
            raise NonFiniteIterateError(
                "nonfinite fixed-point iterate",
                diagnostics,
                trial,
            )

        # The update norm is only a cheap verification trigger.  Acceptance
        # always requires this additional evaluation of the defining residual
        # at the exact endpoint that will be returned.
        if update_inf < tolerance:
            residual = endpoint_residual(
                state, trial, step_size, vector_field
            )
            diagnostics.vector_field_evaluations += 1
            residual_inf = _inf_norm(residual)
            diagnostics.residual_history_inf.append(residual_inf)
            diagnostics.returned_endpoint_residual_inf = residual_inf
            if residual_inf < tolerance:
                diagnostics.converged = True
                return trial, diagnostics

    final_residual = endpoint_residual(state, trial, step_size, vector_field)
    diagnostics.vector_field_evaluations += 1
    diagnostics.returned_endpoint_residual_inf = _inf_norm(final_residual)
    diagnostics.residual_history_inf.append(
        diagnostics.returned_endpoint_residual_inf
    )
    diagnostics.failure_reason = "iteration_limit"
    raise IterationLimitError(
        f"fixed point did not reach endpoint residual tolerance "
        f"{tolerance:.3e} in {maximum_iterations} iterations",
        diagnostics,
        trial,
    )


def _residual_and_jacobian(
    state: Tensor,
    trial: Tensor,
    step_size: float,
    vector_field: VectorField,
) -> tuple[Tensor, Tensor]:
    shape = trial.shape
    flat_trial = trial.reshape(-1)

    def flat_residual(value: Tensor) -> Tensor:
        endpoint = value.reshape(shape)
        return endpoint_residual(
            state, endpoint, step_size, vector_field
        ).reshape(-1)

    residual = flat_residual(flat_trial)
    jacobian = torch.autograd.functional.jacobian(
        flat_residual,
        flat_trial,
        create_graph=torch.is_grad_enabled(),
        vectorize=True,
    )
    return residual, jacobian


def monotone_residual_newton_endpoint(
    state: Tensor,
    step_size: float,
    vector_field: VectorField,
    *,
    tolerance: float = 1e-12,
    maximum_iterations: int = 12,
    maximum_backtracks: int = 8,
    minimum_damping: float = 0.00390625,
    initial_endpoint: Tensor | None = None,
) -> tuple[Tensor, SolverDiagnostics]:
    """Newton solve with strict residual-decrease backtracking."""
    diagnostics = SolverDiagnostics(solver="monotone_residual_newton")
    if initial_endpoint is None:
        trial = state + step_size * vector_field(state)
        diagnostics.vector_field_evaluations += 1
    else:
        trial = initial_endpoint

    for iteration in range(1, maximum_iterations + 1):
        residual, jacobian = _residual_and_jacobian(
            state, trial, step_size, vector_field
        )
        diagnostics.vector_field_evaluations += 1
        diagnostics.jacobian_evaluations += 1
        diagnostics.iterations = iteration
        residual_inf = _inf_norm(residual)
        diagnostics.residual_history_inf.append(residual_inf)
        diagnostics.returned_endpoint_residual_inf = residual_inf
        if residual_inf < tolerance:
            diagnostics.converged = True
            return trial, diagnostics
        try:
            direction = torch.linalg.solve(jacobian, -residual).reshape_as(trial)
        except torch.linalg.LinAlgError as error:
            diagnostics.failure_reason = "singular_endpoint_jacobian"
            raise SingularJacobianError(
                f"singular Newton endpoint Jacobian: {error}",
                diagnostics,
                trial,
            ) from error

        accepted = False
        damping = 1.0
        for _ in range(maximum_backtracks + 1):
            candidate = trial + damping * direction
            candidate_residual = endpoint_residual(
                state, candidate, step_size, vector_field
            )
            diagnostics.vector_field_evaluations += 1
            candidate_inf = _inf_norm(candidate_residual)
            if bool(torch.isfinite(candidate.detach()).all()) and (
                candidate_inf < residual_inf
            ):
                diagnostics.stopping_update_inf = _inf_norm(candidate - trial)
                diagnostics.update_history_inf.append(
                    diagnostics.stopping_update_inf
                )
                diagnostics.accepted_damping_history.append(damping)
                trial = candidate
                diagnostics.returned_endpoint_residual_inf = candidate_inf
                accepted = True
                break
            damping *= 0.5
            diagnostics.backtracks += 1
        if not accepted or damping < minimum_damping:
            diagnostics.failure_reason = "monotone_line_search_failed"
            raise LineSearchError(
                f"Newton line search failed at residual {residual_inf:.3e}",
                diagnostics,
                trial,
            )

    # Explicitly evaluate at the final iterate even though the previous loop
    # already measured candidates.
    final_residual = endpoint_residual(state, trial, step_size, vector_field)
    diagnostics.vector_field_evaluations += 1
    diagnostics.returned_endpoint_residual_inf = _inf_norm(final_residual)
    diagnostics.failure_reason = "iteration_limit"
    raise IterationLimitError(
        f"Newton did not reach endpoint residual tolerance "
        f"{tolerance:.3e} in {maximum_iterations} iterations",
        diagnostics,
        trial,
    )
