"""Stability probe for learned Hamiltonians (paper Algorithm 1).

Given a *learned* Hamiltonian H_theta(x, p) and a quadratic reference
    H0(x, p) = 1/2 |p|^2 + 1/2 x^T K x      (normal modes supplied),
together with a reference ORBIT of H0 (the trajectory whose stability is in
question), this module answers:

    "Is the dynamics of H_theta stable along this orbit, and if it fails,
     through which channel?"

Theory (results/stage5_evidence_report.md, sections 6.4-6.7): write
H_theta = H0 + m * g where m is the rms relative gradient error.  Instability
sets in when the error opens an escape channel whose 1-D effective potential
barrier drops to the energy the orbit carries along that direction.  The
UNIFIED DIRECTIONAL-MIN statistic

    Psi = min over directions u of  eps_crit(u),

scans two physically motivated direction families and takes the minimum:
  * slow modal directions  (Class S: global slow-mode barrier crossing,
                            the directional-min lemma, escape w_max ~ 1);
  * bond-dipole / coordinate directions (Class L: local barrier crossing).
For each direction u we build the 1-D slice potential
    V_u(y) = 1/2 A(u) y^2 + eps * g_u(y),   A(u) = u^T K u,
with g_u the normalised error potential along the ray, and find the first eps
whose escape barrier falls to  E_avail(u) = 1/2 A(u) y_turn(u)^2, where
y_turn(u) = max_t |u . x_orbit(t)| is the orbit's excursion along u.

Verdict:  ratio = m / Psi.  Calibration on 10 real trained HNNs (Exp V2):
measured thresholds = 0.71 * Psi with zero spread, so
    ratio < 0.7    -> predicted STABLE
    0.7 .. 1.4     -> MARGINAL (within the O(1) constant of the theory)
    ratio > 1.4    -> predicted UNSTABLE
The argmin direction identifies the failure channel ("slow" / "local").

Typical usage
-------------
>>> probe = StabilityProbe.from_chain(n=16, n_slow=4, e0=1.0, seed=999)
>>> result = probe.assess(H_theta)          # H_theta: batched torch callable
>>> print(result.verdict, result.ratio, result.channel)

Scope: stiff near-quadratic systems whose reference dynamics is well described
by supplied normal modes (chains, lattices, modal oscillators).  Not intended
for strongly nonlinear references (e.g. Kepler) where no modal decomposition
exists.
"""
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# reference orbit of the quadratic system
# ---------------------------------------------------------------------------
@dataclass
class ReferenceOrbit:
    """Exact orbit of H0: x(t) = modes @ (amp * cos(om t + ph)), v0 = 0."""
    omegas: np.ndarray          # (n,) modal frequencies
    modes: np.ndarray           # (n, n) modal matrix, x = modes @ q
    amp: np.ndarray             # (n,) modal amplitudes
    phases: np.ndarray          # (n,)

    def states(self, ts):
        ts = np.atleast_1d(ts)
        q = self.amp * np.cos(np.outer(ts, self.omegas) + self.phases)
        p = -self.amp * np.sin(np.outer(ts, self.omegas) + self.phases) * self.omegas
        return q @ self.modes.T, p @ self.modes.T

    def excursion(self, u, t_max=4.0e4, dt_sample=0.5):
        """max_t |u . x(t)| on a dense grid (sup over the quasi-periodic orbit)."""
        ts = np.arange(0.0, t_max, dt_sample)
        x_t, _ = self.states(ts)
        return float(np.abs(x_t @ u).max())


# ---------------------------------------------------------------------------
# probe result
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    m: float                    # rms relative gradient error of H_theta
    psi: float                  # unified directional-min threshold
    ratio: float                # m / psi
    verdict: str                # "STABLE" / "MARGINAL" / "UNSTABLE"
    channel: str                # "slow" | "local" | None
    detail: dict = field(default_factory=dict)

    def __str__(self):
        return (f"m={self.m:.3e}  Psi={self.psi:.3e}  m/Psi={self.ratio:.2f}  "
                f"-> {self.verdict} (channel={self.channel}, "
                f"detail={self.detail})")


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------
class StabilityProbe:
    CAL_LO, CAL_HI = 0.7, 1.4   # calibrated margin band (Exp V2)

    def __init__(self, omegas, modes, orbit, n_slow, bond_matrix=None,
                 n_slow_dirs=300, seed=3, x_range=12.0, n_grid=2401):
        """
        omegas      : (n,) modal frequencies of the quadratic reference.
        modes       : (n, n) orthogonal modal matrix (x = modes @ q).
        orbit       : ReferenceOrbit on which stability is assessed.
        n_slow      : number of slow modes (energy-carrying low-frequency part).
        bond_matrix : optional (n_bond, n) bond-difference matrix D; its rows
                      (normalised) provide the Class-L dipole directions.
                      If None, coordinate axes are used as local directions.
        """
        self.om = np.asarray(omegas, dtype=float)
        self.modes = np.asarray(modes, dtype=float)
        self.n = len(self.om)
        self.K = self.modes @ np.diag(self.om ** 2) @ self.modes.T
        self.orbit = orbit
        self.n_slow = n_slow
        self.D = None if bond_matrix is None else np.asarray(bond_matrix, dtype=float)
        self.n_slow_dirs = n_slow_dirs
        self.seed = seed
        self.x_grid = np.linspace(-x_range, x_range, n_grid)

    # ---------------- convenience constructors ----------------
    @classmethod
    def from_chain(cls, n, n_slow, e0=1.0, seed=999, **kw):
        """Fixed-fixed harmonic chain (the testbed of Exp R/T/U/V/W)."""
        k = np.arange(1, n + 1)
        om = 2.0 * np.sin(k * np.pi / (2 * (n + 1)))
        j = np.arange(1, n + 1)
        modes = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(j, k) * np.pi / (n + 1))
        D = np.zeros((n + 1, n)); D[0, 0] = 1.0; D[n, n - 1] = -1.0
        for jj in range(1, n):
            D[jj, jj - 1] = -1.0; D[jj, jj] = 1.0
        rng = np.random.default_rng(seed)
        amp = np.zeros(n); amp[:n_slow] = np.sqrt(2.0 * e0 / n_slow) / om[:n_slow]
        ph = rng.uniform(0, 2 * np.pi, n)
        orbit = ReferenceOrbit(om, modes, amp, ph)
        return cls(om, modes, orbit, n_slow, bond_matrix=D, **kw)

    # ---------------- direction set ----------------
    def _directions(self):
        rng = np.random.default_rng(self.seed)
        dirs = []
        slow = rng.standard_normal((self.n_slow_dirs, self.n_slow))
        slow /= np.linalg.norm(slow, axis=1, keepdims=True)
        slow = np.vstack([slow, np.eye(self.n_slow)])
        for nv in slow:
            q = np.zeros(self.n); q[:self.n_slow] = nv
            u = self.modes @ q
            dirs.append(("slow", u / np.linalg.norm(u),
                         dict(w_max=float(np.max(nv ** 2)))))
        if self.D is not None:
            for j, e in enumerate(self.D):
                nrm = np.linalg.norm(e)
                if nrm > 0:
                    dirs.append(("local", e / nrm, dict(bond=int(j))))
        else:
            for i in range(self.n):
                dirs.append(("local", np.eye(self.n)[i], dict(coord=int(i))))
        return dirs

    # ---------------- error magnitude ----------------
    def error_magnitude(self, grad_err_fn, n_states=300, t_max=200.0, seed=7):
        """m = rms(|grad_x H_err|)/rms(|K x|) over orbit states.
        grad_err_fn(x_batch, p_batch) -> grad_x (H_theta - H0) AT THE ORBIT
        (x, p) points (the error generally depends on p)."""
        rng = np.random.default_rng(seed)
        x, p = self.orbit.states(rng.uniform(0, t_max, n_states))
        err = grad_err_fn(x, p)
        Kx = x @ self.K.T
        return float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(Kx ** 2)))

    # ---------------- core: unified directional-min ----------------
    def _eps_crit_along(self, V_true, g_u, E_avail):
        """First eps whose escape barrier of V_true + eps*g_u drops to E_avail."""
        xg = self.x_grid

        def barrier(eps):
            V = V_true + eps * g_u
            i0 = int(np.argmin(np.abs(xg)))
            lo_i, hi_i = max(0, i0 - 60), min(len(V), i0 + 61)
            iwell = lo_i + int(np.argmin(V[lo_i:hi_i]))
            Vwell = V[iwell]
            left = V[:iwell].max() - Vwell if iwell > 0 else np.inf
            right = V[iwell + 1:].max() - Vwell if iwell < len(V) - 1 else np.inf
            return min(left, right)

        lo, hi = 1e-4, 1e3
        if barrier(hi) > E_avail:
            return np.inf
        for _ in range(26):
            mid = np.sqrt(lo * hi)
            if barrier(mid) <= E_avail:
                hi = mid
            else:
                lo = mid
        return float(hi)

    def assess(self, H_potential_fn, grad_err_fn=None, m=None, n_states=300):
        """Run the probe.

        H_potential_fn(x_batch) -> H_theta(x, p=0) - H_theta(0, 0)   (batched,
            numpy in/out; potential on the p=0 slice, constant removed).
        grad_err_fn(x_batch, p_batch) -> grad_x (H_theta - H0) at the orbit
            states; used for m.  If m is supplied directly, not needed.
        """
        if m is None:
            assert grad_err_fn is not None, "need grad_err_fn or explicit m"
            m = self.error_magnitude(grad_err_fn, n_states=n_states)
        assert m > 0, "error magnitude must be positive"

        dirs = self._directions()
        ts = np.arange(0.0, 4.0e4, 0.5)
        x_t, _ = self.orbit.states(ts)
        U = np.stack([u for _, u, _ in dirs])
        excursions = np.abs(x_t @ U.T).max(axis=0)
        best = dict(psi=np.inf, channel=None, meta={})
        table = []
        for (typ, u, meta), y_turn in zip(dirs, excursions):
            A = float(u @ self.K @ u)
            E_avail = 0.5 * A * y_turn ** 2
            x_phys = self.x_grid[:, None] * u[None, :]
            Hv = H_potential_fn(x_phys)
            V_true = 0.5 * A * self.x_grid ** 2
            g_u = (np.asarray(Hv) - V_true) / m
            eps_c = self._eps_crit_along(V_true, g_u, E_avail)
            table.append((eps_c, typ, meta))
            if eps_c < best["psi"]:
                best = dict(psi=eps_c, channel=typ, meta=meta)
        table.sort(key=lambda r: r[0])

        ratio = m / best["psi"] if np.isfinite(best["psi"]) else 0.0
        if ratio < self.CAL_LO:
            verdict = "STABLE"
        elif ratio > self.CAL_HI:
            verdict = "UNSTABLE"
        else:
            verdict = "MARGINAL"
        return ProbeResult(
            m=m, psi=best["psi"], ratio=ratio, verdict=verdict,
            channel=best["channel"],
            detail=dict(winner=best["meta"],
                        top_channels=[dict(eps=float(e), type=t, meta=mt)
                                      for e, t, mt in table[:5]]))


# ---------------------------------------------------------------------------
# demo: probe a synthetic learned H with a bond-local cubic error (Class L)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n, n_slow = 16, 4
    probe = StabilityProbe.from_chain(n, n_slow, e0=1.0)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n + 1)
    eps_true = 0.2

    def H_err_potential(x):
        Dq = x @ probe.D.T
        return eps_true * np.sum(b * Dq ** 3, axis=1) / 3.0

    def grad_err(x, p=None):
        Dq = x @ probe.D.T
        return eps_true * (probe.D.T @ (b * Dq ** 2).T).T

    H0_pot = lambda x: 0.5 * np.einsum('bi,ij,bj->b', x, probe.K, x)
    res = probe.assess(lambda x: H0_pot(x) + H_err_potential(x),
                       grad_err_fn=grad_err)
    print("demo (bond-cubic error, expect channel=local):", res)
