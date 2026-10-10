# Monitor derivatives and implicit-midpoint conditioning

Status: derivation for internal and independent review. This note does not
claim a necessary condition, global convergence theorem, or experimentally
established conditioning mechanism.

## Setup

Let \(z\in\mathbb R^{2d}\), let \(J\) be the canonical symplectic matrix, and
define

\[
K(z,p_0)=g(z)\bigl(H(z)+p_0\bigr).
\]

The full canonical state is \(x=(z,t,p_0)\). Because \(p_0\) is constant and
\(t\) is recovered after the physical-state solve, the implementation uses
the reduced endpoint equation

\[
R_z(z_+;z)
:=z_+-z-hJ\nabla_zK\!\left(\frac{z+z_+}{2},p_0\right)=0,
\qquad h=\Delta\tau .
\]

Its endpoint Jacobian is

\[
D_{z_+}R_z
=I-\frac h2JD^2_{zz}K\!\left(\frac{z+z_+}{2},p_0\right).
\]

Thus reduced fixed-point contraction and Newton conditioning depend on the
\(z\)-Hessian of the extended Hamiltonian, not only on the physical
Hamiltonian.

## Reduced and full Hessian structure

The product rule gives

\[
\begin{aligned}
D^2_{zz}K
={}&gD^2H
 +\nabla g\otimes\nabla H
 +\nabla H\otimes\nabla g\\
& +(H+p_0)D^2g .
\end{aligned}
\]

This identity separates physical curvature, two mixed monitor/physics terms,
and an off-constraint term. Along the exact continuous trajectory initialized
with \(p_0=-H(z_0)\), \(H+p_0=0\), so the last term vanishes. It need not
vanish at an arithmetic midpoint, during a nonlinear iteration, or after
discrete energy error.

For the full state \(x=(z,t,p_0)\),

\[
\nabla_x^2K=
\begin{bmatrix}
D^2_{zz}K&0&\nabla g\\
0&0&0\\
\nabla g^\top&0&0
\end{bmatrix}.
\]

The full implicit-midpoint endpoint Jacobian is

\[
D_{x_+}\mathcal R=
\begin{bmatrix}
I-\frac h2JD^2_{zz}K&0&-\frac h2J\nabla g\\
-\frac h2\nabla g^\top&1&0\\
0&0&1
\end{bmatrix}.
\]

This block-triangular form shows that reduced-Jacobian nonsingularity
determines full endpoint-Jacobian nonsingularity. The displayed
\(D^2_{zz}K\) must not be called the full extended Hessian.

## A sufficient reduced fixed-point condition

For

\[
\Phi(x)=z+hJ\nabla_zK\!\left(\frac{z+x}{2},p_0\right),
\]

let \(V\) be a closed complete endpoint set satisfying
\(\Phi(V)\subseteq V\), with the initial iterate in \(V\). Define

\[
M=\left\{\frac{z+x}{2}:x\in V\right\},
\qquad
L_K=\sup_{y\in M}\lVert JD^2_{zz}K(y)\rVert .
\]

Then

\[
\lVert D\Phi(x)\rVert\le\frac h2L_K .
\]

Hence

\[
q:=\frac h2L_K<1
\]

is sufficient for a unique reduced fixed point in \(V\) and convergence from
starts in \(V\). It is not necessary and does not imply global trajectory
stability.

Using a subordinate norm,

\[
\begin{aligned}
L_K\le \lVert J\rVert\sup_M\bigl(
&|g|\,\lVert D^2H\rVert
+2\lVert\nabla g\rVert\lVert\nabla H\rVert\\
&+|H+p_0|\,\lVert D^2g\rVert
\bigr).
\end{aligned}
\]

This inequality motivates—but does not validate—conditioning diagnostics.
AC8 did not record these derivative norms or Jacobian spectra.

## Newton interpretation

Newton solves

\[
\left(I-\frac h2JD^2_{zz}K\right)\Delta=-R_z.
\]

If \(q<1\), then in the same subordinate norm

\[
\lVert(DR_z)^{-1}\rVert\le\frac1{1-q},
\qquad
\kappa(DR_z)\le\frac{1+q}{1-q}.
\]

Newton may converge when fixed point is not contractive. Conversely, the AC8
solver is Newton with monotone residual backtracking and a finite backtracking
cap, not a generic globally convergent damped-Newton method. It may fail
because of Jacobian scaling or conditioning, an initial guess outside the
local basin, multiple-root geometry, or globalization settings. AC8 solver
transitions do not identify which explanation applies.

## Symmetry and scaling

Features such as

\[
\left(r,r^{-1},\lVert p\rVert^2,(q^\top p)^2,L^2\right)
\]

make the scalar \(g\) invariant under planar rotations and even under
momentum reversal. Its gradient and transformed vector field are equivariant
under rotations (and compatible with momentum reversal), not componentwise
invariant. These features do not bound \(\lVert\nabla g\rVert\),
\(\lVert D^2g\rVert\), or a Newton condition number.

Condition numbers and singular values depend on coordinate scaling. Variables
must be nondimensionalized, and the norm and scaling matrices fixed, before
comparing reduced or full Jacobians across monitors or orbits.

For the Kepler shell scaling
\[
z=S\widetilde z,\qquad
S=\operatorname{diag}(a,a,a^{-1/2},a^{-1/2}),
\]
the dimensionless residual is defined by
\(\widetilde R(\widetilde z_+)=S^{-1}R(S\widetilde z_+)\).  Therefore the
chain rule gives
\[
D_{\widetilde z_+}\widetilde R
=S^{-1}(D_{z_+}R)S
=I-\frac h2S^{-1}JD^2_{zz}KS.
\]
The corresponding fixed-point derivative is
\((h/2)S^{-1}JD^2_{zz}KS\).  The coefficient remains the original
fictitious-time step \(h\): a state-coordinate similarity transform does not
also reparameterize time.  In particular, inserting an additional
\(a^{-3/2}\) factor is erroneous.  The orbital-time-normalized quantity
\(ha^{-3/2}\) may be reported descriptively, but it is not the coefficient of
this transformed endpoint Jacobian.

## Stopping quantity and symplecticity

The fixed-point implementation tests
\(\lVert\Phi(x_k)-x_k\rVert_\infty\), assigns
\(x_{k+1}=\Phi(x_k)\), and returns \(x_{k+1}\) without recomputing
\(R_z(x_{k+1})\). The recorded value is therefore a stopping update, not a
verified endpoint residual.

Exact fixed-step implicit midpoint is symplectic when its defining equations
are solved exactly. Finite stopping tolerances define an approximate map. A
small stopping update alone does not prove symplecticity; controlling a
symplecticity defect would additionally require derivative control of the
solve error and uniform invertibility assumptions.

## Prospective conditioning audit

For each attempted step, a future audit should record:

- \(\lVert\nabla g\rVert\) and \(\lVert D^2g\rVert\);
- \(|H+p_0|\) at converged midpoints and failed iterates;
- \(\lVert JD^2_{zz}K\rVert\) and \(h\lVert JD^2_{zz}K\rVert/2\);
- scaled singular values and condition numbers of \(D_{z_+}R_z\);
- nonlinear residual histories, accepted damping factors, and failure reason;
- the chosen nondimensionalization, scaling matrices, and norm.

These data could test whether fixed-point failures concentrate at larger
contraction proxies and whether Newton failures concentrate at smaller scaled
singular values. Until then, conditioning is a falsifiable diagnostic
hypothesis, not an established AC8 mechanism.
