"""Parallel Monte Carlo VaR / CVaR (mirrors the HW3 π estimator pattern).

Each worker simulates a slice of paths and returns a partial loss vector;
the main process concatenates and computes empirical quantiles. The
implementation deliberately handles the ``N % n_workers != 0`` case that the
prof flagged in Ed forum #66.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from quantlab._logging import get_logger
from quantlab.exceptions import InsufficientHistoryError

log = get_logger(__name__)


@dataclass
class VaRResult:
    """Container returned by :func:`monte_carlo_var`."""

    var: float
    cvar: float
    confidence: float
    n_paths: int
    horizon_days: int


def _split_evenly(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` non-zero chunks (may differ by one).

    Mirrors HW3's safer ``samples_per_process`` pattern: never emits a 0.
    """
    if parts <= 0:
        raise ValueError("parts must be >= 1")
    if total < parts:
        return [1] * total
    base, rem = divmod(total, parts)
    return [base + 1 if i < rem else base for i in range(parts)]


def _simulate_chunk(args: tuple[int, float, float, int, int]) -> np.ndarray:
    """Worker: simulate a chunk of geometric Brownian motion terminal returns."""
    n_paths, mu, sigma, horizon, seed = args
    rng = np.random.default_rng(seed)
    # Terminal log return = sum of horizon iid Normal(mu - 0.5*sigma^2, sigma) per day.
    drift = (mu - 0.5 * sigma * sigma)
    increments = rng.normal(loc=drift, scale=sigma, size=(n_paths, horizon))
    return np.exp(increments.sum(axis=1)) - 1.0


def monte_carlo_var(
    mu: float,
    sigma: float,
    horizon_days: int,
    n_paths: int,
    confidence: float = 0.99,
    n_workers: int = 1,
    seed: int = 0,
) -> VaRResult:
    """Estimate VaR / CVaR by parallel Monte Carlo on a GBM return process.

    Args:
        mu: Daily log-return mean.
        sigma: Daily log-return std.
        horizon_days: Forecast horizon in trading days.
        n_paths: Total number of simulated paths.
        confidence: VaR confidence level in (0, 1) (e.g. 0.99 for 99%).
        n_workers: Number of parallel processes; ``1`` runs serially.
        seed: Base RNG seed; each worker derives a distinct sub-seed.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    chunks = _split_evenly(n_paths, max(n_workers, 1))
    args = [(c, mu, sigma, horizon_days, seed + i) for i, c in enumerate(chunks)]
    log.info("Monte Carlo VaR: %d paths split into %d chunks", n_paths, len(chunks))

    if n_workers <= 1:
        partials = [_simulate_chunk(a) for a in args]
    else:
        with Pool(processes=n_workers) as pool:
            partials = pool.map(_simulate_chunk, args)

    returns = np.concatenate(partials)
    losses = -returns
    var = float(np.quantile(losses, confidence))
    cvar = float(losses[losses >= var].mean())
    return VaRResult(var=var, cvar=cvar, confidence=confidence, n_paths=int(returns.size),
                     horizon_days=horizon_days)


def historical_simulation(
    returns: Sequence[float],
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> VaRResult:
    """Non-parametric VaR / CVaR from an empirical return distribution.

    Args:
        returns: One-period (typically daily) log returns.
        confidence: VaR confidence level in (0, 1).
        horizon_days: Forecast horizon. When ``> 1``, returns are aggregated
            into ``horizon_days``-day rolling sums (correct for log returns)
            before taking the loss quantile.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    arr = np.asarray(returns, dtype=float)
    if arr.size < horizon_days + 1:
        raise InsufficientHistoryError(
            f"historical_simulation requires >= {horizon_days + 1} returns")
    if horizon_days == 1:
        agg = arr
    else:
        # Rolling sum of log returns gives the H-period log return.
        kernel = np.ones(horizon_days)
        agg = np.convolve(arr, kernel, mode="valid")
    losses = -agg
    var = float(np.quantile(losses, confidence))
    cvar = float(losses[losses >= var].mean())
    return VaRResult(var=var, cvar=cvar, confidence=confidence,
                     n_paths=int(agg.size), horizon_days=horizon_days)
