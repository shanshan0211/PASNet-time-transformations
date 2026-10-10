"""Test replay independence, actual returned residuals and manifest coverage."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "new"))
import review_returned_endpoints as review


class ReturnedEndpointReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)

    def test_independent_rows_match_frozen_scalar_solver(self):
        states = torch.tensor([[1.0], [2.0], [-1.0]])
        rates = torch.tensor([0.3, 1.1, 8.0])
        h = torch.tensor([0.1, 0.4, 0.5])
        field = lambda z, ids: rates[ids, None] * z
        endpoints, ok, bad, its, calls, update, residual = review.batch_step(states, h, field)
        self.assertEqual(ok.tolist(), [True, True, False])
        self.assertFalse(bool(bad.any()))
        for i in (0, 1):
            scalar, stopping, iterations, evaluations = review.ac9.transformed_step(
                states[i:i+1], float(h[i]), lambda z: rates[i] * z,
                {"tolerance": 1e-12, "maximum_iterations": 100})
            self.assertTrue(torch.equal(scalar, endpoints[i:i+1]))
            self.assertEqual(int(its[i]), iterations)
            self.assertEqual(int(calls[i]), evaluations)
            self.assertEqual(float(update[i]), stopping)
        actual = endpoints - states - h[:, None] * field((states + endpoints) * 0.5, torch.arange(3))
        self.assertTrue(torch.equal(residual, actual.abs().amax(dim=1)))
        self.assertGreater(float(residual[2]), 1e-12)

    def test_update_acceptance_does_not_enforce_returned_residual(self):
        # Expanding map: a small update need not bound the returned residual.
        state = torch.tensor([[1e-13]])
        field = lambda z, ids: 4.0 * z
        _, ok, _, its, _, update, residual = review.batch_step(state, torch.tensor([1.0]), field)
        self.assertTrue(bool(ok[0]))
        self.assertEqual(int(its[0]), 1)
        self.assertLess(float(update[0]), 1e-12)
        self.assertGreater(float(residual[0]), float(update[0]))
        self.assertGreater(float(residual[0]), 1e-12)

    def test_manifest_covers_all_new_deployments_without_manual_duplicates(self):
        groups = review.groups()
        self.assertEqual(len(groups), 25)
        self.assertEqual(sum(len(g["records"]) for g in groups if g["study"] == "AC19"), 252)
        self.assertEqual(sum(len(g["records"]) for g in groups if g["study"] == "AC20"), 1080)
        for g in groups:
            keys = [tuple(r[k] for k in review.SCENARIO) for r in g["records"]]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue((ROOT / g["checkpoint"]).is_file())


if __name__ == "__main__":
    unittest.main()
