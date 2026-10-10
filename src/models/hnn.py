"""Hamiltonian Neural Network (Greydanus et al., 2019) baseline.

Scalar MLP H_theta(z), z = (u, v) in modal coordinates; dynamics recovered
via autograd: dz/dt = J grad H = (dH/dv, -dH/du).
"""
import torch
import torch.nn as nn


class HNN(nn.Module):
    def __init__(self, dim, hidden=128, n_hidden=3):
        super().__init__()
        self.dim = dim
        layers = [nn.Linear(2 * dim, hidden), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)

    def time_derivative(self, z):
        """z: (B, 2d) tensor with requires_grad; returns dz/dt = J grad H."""
        h = self.forward(z).sum()
        grad = torch.autograd.grad(h, z, create_graph=True)[0]
        d = self.dim
        return torch.cat([grad[:, d:], -grad[:, :d]], dim=1)
