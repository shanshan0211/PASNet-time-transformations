"""PASNet: Poincare-Adaptive Symplectic Network.

Learns an extended-phase-space Hamiltonian
    K_theta(z, p0) = g_theta(z) * (H_theta(z) + p0),   z = (q, p)
where
  * H_theta  : learned physical Hamiltonian (Stage A, supervised on derivatives)
  * g_theta  : learned time-transformation function dt = g_theta(z) d-tau,
               bounded in (0, g_max) (Stage B, trained for long-horizon
               structure preservation at low cost)

Rollout integrates K with the implicit midpoint rule (symplectic for general
non-separable K) at UNIFORM d-tau; the physical step dt = g_theta d-tau adapts
automatically. p0 stays at -H(z0) since dK/dtau = 0 along the flow.
"""
import torch
import torch.nn as nn


def make_mlp(in_dim, hidden, n_hidden, out_dim=1):
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class PASNet(nn.Module):
    def __init__(self, dim, hidden_H=128, n_hidden_H=3,
                 hidden_g=64, n_hidden_g=2, g_max=2.0,
                 h_features=None, g_features=None):
        """h_features / g_features: callable(z) -> feature tensor appended
        to the respective network input. g_features lets one supply the
        coordinate radius r (a pure coordinate function, not a physical
        prior) to ease learning the r-dependence of g on singular problems."""
        super().__init__()
        self.dim = dim
        self.g_max = g_max
        self.h_features = h_features
        self.g_features = g_features
        in_H = 2 * dim + (len(h_features(torch.zeros(1, 2 * dim))[0])
                          if h_features else 0)
        in_g = 2 * dim + (len(g_features(torch.zeros(1, 2 * dim))[0])
                          if g_features else 0)
        self.H_net = make_mlp(in_H, hidden_H, n_hidden_H)
        self.g_net = make_mlp(in_g, hidden_g, n_hidden_g)
        # cold-start safety: initialize g small (~0.1*g_max) so that untrained
        # rollouts are stable (small effective physical steps); g can grow
        # during Stage-B training.
        last = self.g_net[-1]
        nn.init.normal_(last.weight, std=0.1)
        nn.init.constant_(last.bias, -2.2)

    def H(self, z):
        x = z
        if self.h_features is not None:
            x = torch.cat([z, self.h_features(z)], dim=1)
        return self.H_net(x).squeeze(-1)

    def g(self, z):
        x = z
        if self.g_features is not None:
            x = torch.cat([z, self.g_features(z)], dim=1)
        return self.g_max * torch.sigmoid(self.g_net(x).squeeze(-1))

    def K(self, z, p0):
        return self.g(z) * (self.H(z) + p0)

    def J_grad_H(self, z):
        if not z.requires_grad:
            z = z.requires_grad_(True)
        h = self.H(z).sum()
        grad = torch.autograd.grad(h, z, create_graph=True)[0]
        d = self.dim
        return torch.cat([grad[:, d:], -grad[:, :d]], dim=1)

    def J_grad_K(self, z, p0):
        if not z.requires_grad:
            z = z.requires_grad_(True)
        k = self.K(z, p0).sum()
        grad = torch.autograd.grad(k, z, create_graph=True)[0]
        d = self.dim
        return torch.cat([grad[:, d:], -grad[:, :d]], dim=1)

    @torch.no_grad()
    def g_numpy(self, z_np):
        import numpy as np
        z = torch.as_tensor(z_np, dtype=torch.float32)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.g(z).numpy()


def midpoint_rollout(model, z0, p0, dtau, n_steps, n_fp_iter=5,
                     record_every=1, detach_every=0):
    """Implicit midpoint on K with uniform tau steps.

    z0: (1, 2d) tensor. p0: scalar tensor (constant along the flow).
    detach_every > 0 enables truncated backpropagation through time (TBPTT):
    state is detached every detach_every steps, bounding the graph depth for
    long rollouts (losses are step-local statistics, so training signals are
    preserved). Returns lists of (t, z) tensors.
    """
    z = z0
    t = torch.zeros((), dtype=z0.dtype)
    ts, zs = [t], [z]
    for i in range(n_steps):
        f0 = model.J_grad_K(z, p0)
        z_new = z + dtau * f0
        for _ in range(n_fp_iter):
            zm = 0.5 * (z + z_new)
            f = model.J_grad_K(zm, p0)
            z_new = z + dtau * f
        t = t + dtau * model.g(0.5 * (z + z_new)).squeeze()
        z = z_new
        if detach_every and (i + 1) % detach_every == 0:
            z = z.detach()
            t = t.detach()
        if (i + 1) % record_every == 0 or i == n_steps - 1:
            ts.append(t)
            zs.append(z)
    return ts, zs
