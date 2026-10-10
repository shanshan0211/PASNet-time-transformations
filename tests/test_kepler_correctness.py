"""Regression tests for the corrected known-H Kepler implementation."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from kepler_integrators import implicit_midpoint_step, kepler_gradient  # noqa: E402
from systems import Duffing1D, Kepler2D, ModalOscillator  # noqa: E402


class KeplerCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.system = Kepler2D(0.9)
        self.q0, self.p0 = self.system.initial_condition()

    def test_torch_hamiltonian_gradient_matches_analytic_gradient(self):
        z = torch.tensor(
            [
                [1.2, -0.4, 0.3, -0.2],
                [-0.7, 0.9, -0.1, 0.5],
                [0.2, 0.4, 1.1, -0.8],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        grad = torch.autograd.grad(self.system.H_torch(z).sum(), z)[0]

        q = z.detach().numpy()[:, :2]
        p = z.detach().numpy()[:, 2:]
        expected_q = q / np.linalg.norm(q, axis=1, keepdims=True) ** 3
        expected = np.concatenate([expected_q, p], axis=1)
        np.testing.assert_allclose(
            grad.detach().numpy(), expected, rtol=1e-12, atol=1e-12
        )

    def test_extended_hamiltonian_vector_field_is_not_severed(self):
        z = torch.tensor(
            [[1.3, 0.2, -0.1, 0.4]], dtype=torch.float64, requires_grad=True
        )
        p0 = -self.system.H_torch(z.detach())
        g = torch.ones(z.shape[0], dtype=z.dtype)
        k = g * (self.system.H_torch(z) + p0)
        grad_k = torch.autograd.grad(k.sum(), z)[0]
        j_grad_k = torch.cat([grad_k[:, 2:], -grad_k[:, :2]], dim=1)

        expected = self.system.vector_field_torch(z.detach())
        torch.testing.assert_close(j_grad_k, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(float(torch.linalg.vector_norm(j_grad_k)), 0.1)

    def test_midpoint_step_has_attractive_force_and_converges(self):
        q1, p1, info = implicit_midpoint_step(
            self.q0, self.p0, 1e-3, max_iter=100, tol=1e-14
        )
        self.assertTrue(info.converged)
        self.assertLess(info.residual, 1e-14)
        self.assertLess(p1[0], 0.0, "gravity at apoapsis must point toward origin")
        self.assertLess(q1[0], self.q0[0])
        self.assertFalse(np.allclose(q1, self.q0))

    def test_midpoint_one_period_state_error_is_second_order(self):
        errors = []
        for dt in (0.01, 0.005, 0.0025):
            q, p = self.q0.copy(), self.p0.copy()
            t = 0.0
            period = 2.0 * math.pi
            while t < period:
                h = min(dt, period - t)
                q, p, _ = implicit_midpoint_step(
                    q, p, h, max_iter=100, tol=1e-13
                )
                t += h
            errors.append(
                float(
                    np.linalg.norm(
                        np.concatenate([q - self.q0, p - self.p0])
                    )
                )
            )

        self.assertGreater(errors[0] / errors[1], 3.0)
        self.assertGreater(errors[1] / errors[2], 3.0)

    def test_numpy_gradient_sign_matches_system_definition(self):
        grad_q, grad_p = kepler_gradient(self.q0, self.p0)
        np.testing.assert_allclose(grad_q, self.system.grad_q(self.q0))
        np.testing.assert_allclose(grad_p, self.p0)
        self.assertGreater(grad_q[0], 0.0)

    def test_other_known_hamiltonians_keep_autograd_graph(self):
        modal = ModalOscillator(
            n_modes=2, omega_min=1.0, kappa=3.0, beta=0.25,
            orthogonal_mixing=False,
        )
        z_modal = torch.tensor(
            [[0.2, -0.1, 0.4, 0.3]], dtype=torch.float64, requires_grad=True
        )
        grad_modal = torch.autograd.grad(modal.H_torch(z_modal).sum(), z_modal)[0]
        self.assertGreater(float(torch.linalg.vector_norm(grad_modal)), 0.0)

        duffing = Duffing1D(beta=0.1)
        z_duffing = torch.tensor(
            [[0.5, -0.2]], dtype=torch.float64, requires_grad=True
        )
        grad_duffing = torch.autograd.grad(
            duffing.H_torch(z_duffing).sum(), z_duffing
        )[0]
        expected = torch.tensor(
            [[0.5 + 0.1 * 0.5**3, -0.2]], dtype=torch.float64
        )
        torch.testing.assert_close(
            grad_duffing, expected, rtol=1e-12, atol=1e-12
        )

    def test_experiments_do_not_reintroduce_known_h_graph_breaks(self):
        experiments = Path(HERE).parent / "experiments"
        forbidden = {
            r"torch\.as_tensor\(\s*kep\.H": "Kepler H converted from NumPy",
            r"Hv_np\s*=\s*kep\.H": "detached Kepler H intermediate",
            r"torch\.as_tensor\(\s*fpu\.energy": "FPU H converted from NumPy",
            r"torch\.as_tensor\(\s*np\.array\(\s*\[\s*duffing_energy":
                "Duffing H converted from NumPy",
            r"dH_dq\s*=\s*-q\s*/\s*r\*\*3": "wrong Kepler gradient sign",
        }
        failures = []
        for script in experiments.glob("*.py"):
            text = script.read_text(encoding="utf-8")
            for pattern, reason in forbidden.items():
                if re.search(pattern, text, flags=re.MULTILINE):
                    failures.append(f"{script.name}: {reason}")
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
