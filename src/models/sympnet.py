"""G-SympNet baseline (Jin et al., 2020): composition of symplectic shear maps.

p-update layer:  p <- p + K^T (a * act(K q + b))      (= p - dt * grad V_eff(q))
q-update layer:  q <- q + K^T (a * act(K p + b))

Each map is exactly symplectic since its Jacobian is block-triangular with a
symmetric off-diagonal block  K^T diag(a * act') K.
The composition learns the fixed-dt flow map of the Hamiltonian system.
"""
import torch
import torch.nn as nn


class ShearLayer(nn.Module):
    def __init__(self, dim, width, update):
        super().__init__()
        assert update in ("p", "q")
        self.update = update
        self.K = nn.Parameter(torch.zeros(width, dim))
        self.b = nn.Parameter(torch.zeros(width))
        self.a = nn.Parameter(torch.zeros(width))  # zero init => identity map
        nn.init.normal_(self.K, std=0.01)

    def _g(self, x):
        h = torch.tanh(x @ self.K.T + self.b)
        return (h * self.a) @ self.K

    def forward(self, q, p):
        if self.update == "p":
            return q, p + self._g(q)
        return q + self._g(p), p


class GSympNet(nn.Module):
    def __init__(self, dim, width=32, n_layers=6):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            ShearLayer(dim, width, "p" if i % 2 == 0 else "q")
            for i in range(n_layers))

    def forward(self, z):
        """One application of the learned dt-flow map. z: (B, 2d)."""
        q, p = z[:, : self.dim], z[:, self.dim:]
        for layer in self.layers:
            q, p = layer(q, p)
        return torch.cat([q, p], dim=1)

    @torch.no_grad()
    def rollout(self, z0, n_steps):
        """z0: (2d,) numpy. Returns (n_steps+1, 2d) numpy trajectory."""
        import numpy as np
        z = torch.as_tensor(z0, dtype=torch.float32).unsqueeze(0)
        traj = [z.squeeze(0).numpy()]
        for _ in range(n_steps):
            z = self.forward(z)
            traj.append(z.squeeze(0).numpy())
        return np.stack(traj)
