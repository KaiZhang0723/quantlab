"""Tests for portfolio.markowitz."""

from __future__ import annotations

import unittest

import numpy as np

from quantlab.portfolio.markowitz import mean_variance_optimal


class MarkowitzTests(unittest.TestCase):
    def test_two_asset_min_variance_closed_form(self) -> None:
        # For two assets the global min-variance long-only portfolio has
        # weights w_a = (sigma_b^2 - rho * sigma_a * sigma_b) / (sigma_a^2 + sigma_b^2 - 2*rho*sigma_a*sigma_b)
        sigma_a, sigma_b, rho = 0.10, 0.20, 0.0
        cov = np.array([[sigma_a**2, rho * sigma_a * sigma_b],
                        [rho * sigma_a * sigma_b, sigma_b**2]])
        mu = np.array([0.05, 0.10])
        res = mean_variance_optimal(mu, cov, risk_aversion=0.0, long_only=True)
        denom = sigma_a**2 + sigma_b**2 - 2 * rho * sigma_a * sigma_b
        expected_wa = (sigma_b**2 - rho * sigma_a * sigma_b) / denom
        self.assertTrue(res.success)
        self.assertAlmostEqual(res.weights[0], expected_wa, places=4)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=8)

    def test_weights_sum_to_one_and_non_negative_long_only(self) -> None:
        rng = np.random.default_rng(0)
        n = 5
        A = rng.normal(size=(n, n))
        cov = A @ A.T / n + np.eye(n) * 1e-3
        mu = rng.normal(size=n) * 0.05
        res = mean_variance_optimal(mu, cov, risk_aversion=1.0, long_only=True)
        self.assertAlmostEqual(res.weights.sum(), 1.0, places=6)
        self.assertTrue((res.weights >= -1e-8).all())

    def test_long_short_allows_negatives(self) -> None:
        # With 3 assets and one having strongly negative mu, the long-short
        # max-utility weights should short that asset.
        cov = np.eye(3) * 0.04
        mu = np.array([0.20, 0.10, -0.30])
        res = mean_variance_optimal(mu, cov, risk_aversion=5.0, long_only=False)
        self.assertLess(res.weights[2], -1e-6)

    def test_invalid_inputs_raise(self) -> None:
        cov = np.eye(2)
        with self.assertRaises(ValueError):
            mean_variance_optimal(np.array([0.1, 0.2]), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            mean_variance_optimal(np.array([0.1, 0.2]), cov, risk_aversion=-1.0)
        non_sym = np.array([[1.0, 0.0], [0.5, 1.0]])
        with self.assertRaises(ValueError):
            mean_variance_optimal(np.array([0.1, 0.2]), non_sym)

    def test_sharpe_zero_when_vol_zero(self) -> None:
        # Degenerate but well-defined: zero covariance => zero vol.
        cov = np.zeros((2, 2))
        mu = np.array([0.05, 0.10])
        res = mean_variance_optimal(mu, cov, risk_aversion=0.0)
        self.assertEqual(res.sharpe, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
