"""Vectorised rolling and cumulative analytics on price / return series.

These are the building blocks the backtester and reporting code consume.
Pure-functional and numpy-only so they're trivial to unit-test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.exceptions import InsufficientHistoryError

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns ``log(p_t / p_{t-1})`` of a price series.

    The first observation is dropped because it has no predecessor.

    Raises:
        ValueError: If any price is non-positive (``log`` would be ill-defined).
    """
    if len(prices) < 2:
        raise InsufficientHistoryError("log_returns requires at least 2 prices")
    if (prices <= 0).any():
        raise ValueError("log_returns requires strictly positive prices")
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative growth factor ``exp(cumsum(returns))``.

    Note: the resulting series is the equity *after* each period, so
    ``equity[0] = exp(returns[0])``, not ``1.0``. Drawdowns measured from
    this curve are computed against the running max of the post-trade
    equity, which is the standard convention.
    """
    if returns.empty:
        return returns.copy()
    return np.exp(returns.cumsum())


def rolling_volatility(returns: pd.Series, window: int = 20, annualised: bool = True) -> pd.Series:
    """Rolling sample standard deviation of returns.

    Args:
        returns: Daily log or simple returns.
        window: Trailing window length in observations.
        annualised: If True, multiply by ``sqrt(TRADING_DAYS_PER_YEAR)``.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    vol = returns.rolling(window).std(ddof=1)
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualised else vol


def rolling_sharpe(returns: pd.Series, window: int = 60, rf: float = 0.0) -> pd.Series:
    """Rolling annualised Sharpe ratio of ``returns`` over ``window``."""
    if window < 2:
        raise ValueError("window must be >= 2")
    excess = returns - rf / TRADING_DAYS_PER_YEAR
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown = current equity / running max - 1.

    ``equity`` is typically the cumulative return curve (starts at 1.0).
    """
    if equity.empty:
        return equity.copy()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown of an equity curve, as a non-positive scalar."""
    dd = drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0
