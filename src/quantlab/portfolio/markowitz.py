"""Mean-variance portfolio optimisation (Markowitz, 1952).

Solves ``minimise w'Sigma w - lambda * mu' w`` subject to ``sum(w) = 1`` and
optional ``w >= 0`` (long-only). Uses :func:`scipy.optimize.minimize` with
SLSQP. The 2-asset closed-form sanity test lives in the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class MarkowitzResult:
    """Result of a mean-variance optimisation."""

    weights: np.ndarray
    expected_return: float
    expected_vol: float
    sharpe: float
    success: bool
    message: str


def mean_variance_optimal(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> MarkowitzResult:
    """Minimum-variance / max-utility portfolio under the standard constraints.

    Args:
        mu: Vector of expected returns, shape ``(n,)``.
        cov: Covariance matrix, shape ``(n, n)``. Must be positive semi-definite.
        risk_aversion: Trade-off between return and variance; ``0`` ⇒ pure min-var.
        long_only: If True, constrain ``w_i >= 0``.

    Returns:
        :class:`MarkowitzResult` with the optimal weight vector and summary stats.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    n = mu.size
    if cov.shape != (n, n):
        raise ValueError(f"cov must be ({n},{n}), got {cov.shape}")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("cov must be symmetric")

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w - risk_aversion * mu @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2 * cov @ w - risk_aversion * mu

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    # Long-only is bounded to [0, 1]; long-short allows up to 2x leverage
    # in either direction, which is enough to express realistic dollar-
    # neutral or 130/30 books while still keeping the optimisation bounded.
    bounds = [(0.0, 1.0)] * n if long_only else [(-2.0, 2.0)] * n
    w0 = np.full(n, 1.0 / n)

    res = minimize(objective, w0, jac=gradient, method="SLSQP",
                   bounds=bounds, constraints=constraints, options={"ftol": 1e-10, "maxiter": 500})

    w = res.x
    er = float(mu @ w)
    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = er / vol if vol > 0 else 0.0
    return MarkowitzResult(weights=w, expected_return=er, expected_vol=vol,
                           sharpe=sharpe, success=bool(res.success), message=str(res.message))
