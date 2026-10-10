# Proposition 5: Valid Continuous Statement and Retracted Discrete Bound

> **Status (2026-08-15):** only the continuous-flow proposition below is
> currently supported. The earlier “corrected” document still overclaimed a
> time-independent discretisation term and a non-accumulating solver-residual
> term. Those discrete claims are retracted pending a separate proof.

## Setup

Let

\[
K_\theta(z,p_0)=g_\theta(z)\bigl(H_\theta(z)+p_0\bigr),\qquad g_\theta(z)>0,
\]

and consider its exact continuous Hamiltonian flow on the zero level set
\(K_\theta=0\). Assume the trajectory remains in a compact set
\(\Omega\) over physical time \(0\le t\le T\), and define

\[
\epsilon_H=
\sup_{z\in\Omega}\|\nabla H_\theta(z)-\nabla H(z)\|,\qquad
C_1=\sup_{z\in\Omega}\|\nabla H(z)\|.
\]

## Proposition: continuous true-energy variation

Along the exact transformed flow,

\[
\boxed{|H(z(T))-H(z(0))|\le C_1\epsilon_H T.}
\]

This bound is independent of the positive time transformation \(g_\theta\).

### Proof

On \(K_\theta=0\), positivity of \(g_\theta\) gives
\(H_\theta+p_0=0\). Hamilton’s equation in transformed time reduces to

\[
\frac{dz}{d\tau}=g_\theta J\nabla H_\theta,\qquad
\frac{dt}{d\tau}=g_\theta,
\]

and therefore \(dz/dt=J\nabla H_\theta\). Hence

\[
\frac{dH}{dt}
=\nabla H^\top J\nabla H_\theta
=\nabla H^\top J(\nabla H_\theta-\nabla H),
\]

because \(\nabla H^\top J\nabla H=0\). Cauchy–Schwarz gives
\(|dH/dt|\le C_1\epsilon_H\); integration over \([0,T]\) proves the claim.

## What the proposition does and does not imply

Supported:

- any positive \(g_\theta\) is a time reparameterisation of the continuous
  \(H_\theta\) flow on \(K_\theta=0\);
- when \(H_\theta=H\), the exact continuous transformed flow conserves the
  true energy for every positive \(g_\theta\);
- no additive “time-transformation approximation error”
  \(C_2\epsilon_g\omega_{\max}T/g_{\min}\) belongs in this continuous bound.

Not established:

- a discrete bound of the form
  \(C_3\Delta\tau^2/g_{\min}+C_4\delta_{\rm fp}/g_{\min}\) with no time or
  step-count dependence;
- an unconditional, all-time backward-error bound;
- a Grönwall “crossover” formula for energy error;
- that the learned monitor reduces any bound relative to a manual monitor;
- that a time transformation is mathematically necessary for convergence on
  a Kepler orbit with finite \(r_{\min}>0\).

For a discrete result, the proof must state the analytic domain, step-size
restriction, transformed-time horizon, bounds on derivatives of
\(K_\theta\), and how per-step nonlinear-solver defects accumulate. Until
then, discretisation and solver effects are measured empirically with
refinement and residual gates.

## Corrected numerical evidence

The historical known-\(H\) zero-drift result was generated after converting
the state to NumPy inside the Hamiltonian evaluation, which severed
automatic differentiation. The historical direct-midpoint plateau used the
wrong sign for \(\nabla_q(-1/r)\). Both are invalidated.

With corrected float64 implementations on Kepler \(e=0.9\), one period:

- direct implicit midpoint state errors are
  \(1.1368\times10^{-1}\), \(2.6800\times10^{-2}\), and
  \(6.6078\times10^{-3}\) for
  \(\Delta t=0.01,0.005,0.0025\), with observed orders 2.085 and 2.020;
- learned-\(g_\theta\), true-\(H\) maximum energy errors are
  \(5.4116\times10^{-3}\), \(1.3476\times10^{-3}\), and
  \(3.3662\times10^{-4}\) for
  \(\Delta\tau=0.04,0.02,0.01\).

Artifacts:

- `../results/expAC0/correctness_smoke.json`
- `../results/expAC1/knownH_corrected.json`
- `../tests/test_kepler_correctness.py`

These data support second-order refinement, not machine-precision
conservation or structural necessity.
