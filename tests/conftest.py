"""Shared test helpers.

We deliberately keep all randomness deterministic and avoid any network
calls — the prof should be able to clone, ``pip install -e .[dev]``, and
``pytest`` without external prerequisites.

Tests use ``unittest.TestCase`` style and import ``_synthetic_panel``
directly rather than consuming pytest fixtures, since unittest's
``TestCase`` doesn't auto-inject fixture parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"


def _synthetic_panel(
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    n_days: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion price panel for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    frames = []
    for i, t in enumerate(tickers):
        mu = 0.0003 + 0.0001 * i
        sigma = 0.012 + 0.001 * i
        rets = rng.normal(loc=mu, scale=sigma, size=n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({
            "date": dates,
            "ticker": t,
            "open": close * (1 + rng.normal(scale=0.001, size=n_days)),
            "high": close * (1 + np.abs(rng.normal(scale=0.002, size=n_days))),
            "low":  close * (1 - np.abs(rng.normal(scale=0.002, size=n_days))),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(low=10_000, high=1_000_000, size=n_days),
        }))
    return pd.concat(frames, ignore_index=True)
