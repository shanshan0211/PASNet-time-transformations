"""Stiff coupled oscillator with analytically controllable stiffness ratio.

H(q, p) = 1/2 p^T p + 1/2 q^T K q + (beta/4) * sum_i u_i^4,  u = Q^T q

eig-frequencies omega_i are log-uniform in [omega_min, omega_max],
stiffness ratio kappa = omega_max / omega_min.
All integration is done in modal coordinates (u, v) = (Q^T q, Q^T p),
where the linear part is diagonal (orthogonal change of coordinates is
symplectic, so no structure is lost).
"""
import numpy as np


class ModalOscillator:
    def __init__(self, n_modes=8, omega_min=1.0, kappa=1.0, beta=0.0,
                 seed=0, orthogonal_mixing=True):
        rng = np.random.default_rng(seed)
        self.n = n_modes
        self.beta = beta
        self.omega_min = omega_min
        self.kappa = kappa
        omegas = np.exp(np.linspace(np.log(omega_min),
                                    np.log(omega_min * kappa), n_modes))
        self.omegas = omegas
        self.omega_max = omegas[-1]
        if orthogonal_mixing and n_modes > 1:
            Q, _ = np.linalg.qr(rng.standard_normal((n_modes, n_modes)))
        else:
            Q = np.eye(n_modes)
        self.Q = Q  # modal matrix: q = Q u

    def initial_condition(self, e0=1.0, slow_fraction=0.9, seed=1):
        """Total energy e0; slow_fraction of it goes to the lower half of modes."""
        rng = np.random.default_rng(seed)
        n_slow = self.n // 2
        weights = np.concatenate([
            np.full(n_slow, slow_fraction / n_slow),
            np.full(self.n - n_slow, (1 - slow_fraction) / (self.n - n_slow)),
        ])
        phases = rng.uniform(0, 2 * np.pi, self.n)
        amps = np.sqrt(2 * e0 * weights) / self.omegas
        u0 = amps * np.cos(phases)
        v0 = -amps * self.omegas * np.sin(phases)
        return u0, v0

    def energy_linear_part(self, u, v):
        """The exactly-conserved energy of the beta=0 system."""
        om2 = self.omegas ** 2
        return 0.5 * np.sum(v ** 2 + om2 * u ** 2, axis=-1)

    def energy(self, u, v):
        e = self.energy_linear_part(u, v)
        if self.beta != 0.0:
            e = e + (self.beta / 4.0) * np.sum(u ** 4, axis=-1)
        return e

    def H_torch(self, z):
        """Differentiable modal Hamiltonian for batched ``[u, v]`` tensors."""
        import torch

        u, v = z[..., :self.n], z[..., self.n:]
        om2 = torch.as_tensor(
            self.omegas ** 2, dtype=z.dtype, device=z.device
        )
        energy = 0.5 * torch.sum(v ** 2 + om2 * u ** 2, dim=-1)
        if self.beta != 0.0:
            energy = energy + (self.beta / 4.0) * torch.sum(u ** 4, dim=-1)
        return energy


class Kepler2D:
    """Planar Kepler problem  H = |p|^2/2 - 1/|q|,  q, p in R^2.

    Eccentricity e; semi-major axis a = 1; gravitational parameter mu = 1.
    Period T = 2 pi. Periapsis r_min = 1 - e, apoapsis r_max = 1 + e.
    Near periapsis the local timescale collapses (~ r^{3/2}), which is where
    fixed-step explicit integrators fail and Sundman time transformation
    (g ~ r) regularizes.
    """

    def __init__(self, eccentricity=0.9):
        self.e = eccentricity
        self.energy_exact = -0.5  # a = 1

    def initial_condition(self):
        """Start at apoapsis: q = (1+e, 0), p = (0, sqrt((1-e)/(1+e)))."""
        q0 = np.array([1.0 + self.e, 0.0])
        p0 = np.array([0.0, np.sqrt((1.0 - self.e) / (1.0 + self.e))])
        return q0, p0

    def H(self, q, p):
        r = np.linalg.norm(q, axis=-1)
        return 0.5 * np.sum(p ** 2, axis=-1) - 1.0 / r

    def H_torch(self, z):
        """Differentiable Kepler Hamiltonian for batched ``[q, p]`` tensors.

        Keep the full computation in Torch. Converting ``z`` to NumPy here
        would sever the autograd graph and remove the physical Hamiltonian
        vector field from extended-phase-space rollouts.
        """
        import torch

        q, p = z[..., :2], z[..., 2:]
        r = torch.linalg.vector_norm(q, dim=-1)
        return 0.5 * torch.sum(p ** 2, dim=-1) - 1.0 / r

    def vector_field_torch(self, z):
        """Analytic Hamiltonian vector field ``(dq/dt, dp/dt)`` in Torch."""
        import torch

        q, p = z[..., :2], z[..., 2:]
        r = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        return torch.cat([p, -q / r ** 3], dim=-1)

    def grad_q(self, q):
        r = np.linalg.norm(q, axis=-1, keepdims=True)
        return q / r ** 3

    def period(self):
        return 2 * np.pi


class Duffing1D:
    """Hardening Duffing oscillator  H = p^2/2 + x^2/2 + beta x^4/4.

    The instantaneous frequency grows with amplitude, so large-|x| regions
    need finer steps -- an INTERMEDIATE regime between the uniform stiff
    oscillator (no state dependence) and Kepler (localized singularity).
    """

    def __init__(self, beta=0.5):
        self.beta = beta

    def initial_condition(self, e0=1.0, seed=1):
        # turning point: solve 0.5 A^2 + beta A^4/4 = e0  (y=A^2):
        y = (-1.0 + np.sqrt(1.0 + 4.0 * self.beta * e0)) / self.beta
        A = np.sqrt(max(y, 1e-12))
        rng = np.random.default_rng(seed)
        # random phase via uniform angle
        phi = rng.uniform(0, 2 * np.pi)
        # elliptic-orbit parametrization (approximate): on the energy shell
        # use canonical coordinates with the turning-point amplitude A
        x0 = A * np.cos(phi)
        # p chosen so H = e0: p^2/2 = e0 - V(x0)
        V = 0.5 * x0 ** 2 + 0.25 * self.beta * x0 ** 4
        p0 = np.sqrt(max(2 * (e0 - V), 0.0)) * np.sign(rng.standard_normal())
        return np.array([x0]), np.array([p0])

    def H(self, x, p):
        return 0.5 * p ** 2 + 0.5 * x ** 2 + 0.25 * self.beta * x ** 4

    def H_torch(self, z):
        """Differentiable Duffing Hamiltonian for batched ``[x, p]`` tensors."""
        x, p = z[..., 0], z[..., 1]
        return 0.5 * p ** 2 + 0.5 * x ** 2 + 0.25 * self.beta * x ** 4

    def grad_x(self, x):
        return x + self.beta * x ** 3

    def omega_at_amplitude(self, A):
        # first-order action-angle approximation
        return np.sqrt(1.0 + 1.5 * self.beta * A ** 2)
