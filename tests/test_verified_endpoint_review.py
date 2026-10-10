from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new"))
sys.path.insert(0, str(ROOT / "src"))
from replay_verified_endpoints import verified_step
from implicit_midpoint_solvers import fixed_point_endpoint, IterationLimitError


class VerifiedReviewTests(unittest.TestCase):
    def test_independent_acceptance_and_iteration_limit_match_verified_solver(self):
        torch.set_default_dtype(torch.float64)
        z = torch.tensor([[1.0], [2.0], [1e-13]])
        h = torch.tensor([0.1, 0.4, 1.0])
        rates = torch.tensor([0.3, 1.1, 4.0])
        field = lambda value, ids: rates[ids, None] * value
        out, ok, _, its, _, _, residual = verified_step(z, h, field)
        self.assertEqual(ok.tolist(), [True, True, False])
        for i in (0, 1):
            scalar, diag = fixed_point_endpoint(z[i:i+1], float(h[i]), lambda value: rates[i] * value)
            self.assertTrue(torch.equal(scalar, out[i:i+1]))
            self.assertEqual(int(its[i]), diag.iterations)
            self.assertEqual(float(residual[i]), diag.returned_endpoint_residual_inf)
        with self.assertRaises(IterationLimitError):
            fixed_point_endpoint(z[2:3], 1.0, lambda value: 4.0 * value)
        self.assertGreater(float(residual[2]), 1e-12)


if __name__ == "__main__":
    unittest.main()
