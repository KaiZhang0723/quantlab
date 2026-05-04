"""Shared pytest fixtures.

We deliberately keep all randomness deterministic and avoid any network
calls — the prof should be able to clone, ``pip install -e .[dev]``, and
``pytest`` without external prerequisites.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    return _synthetic_panel()


@pytest.fixture(scope="session")
def synthetic_prices_small() -> pd.DataFrame:
    return _synthetic_panel(tickers=("AAA", "BBB"), n_days=120)


@pytest.fixture(scope="session")
def date_range() -> tuple[date, date]:
    return date(2020, 1, 1), date(2024, 12, 31)


@pytest.fixture(scope="session")
def wiki_html_fixture() -> str:
    return (FIXTURES / "wiki_sp500_sample.html").read_text(encoding="utf-8")
