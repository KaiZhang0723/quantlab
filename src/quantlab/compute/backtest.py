"""Parallel backtester for per-ticker trading strategies.

A strategy is any callable ``(prices_df) -> positions_series`` that maps a
single ticker's price history to a series of positions in {-1, 0, +1}. The
backtester applies one strategy per ticker in parallel via
``multiprocessing.Pool`` (W8 / HW3 pattern).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd

from quantlab._logging import get_logger
from quantlab.compute.rolling import (
    TRADING_DAYS_PER_YEAR,
    cumulative_returns,
    log_returns,
    max_drawdown,
)
from quantlab.exceptions import InsufficientHistoryError

log = get_logger(__name__)

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class TickerPnL:
    """Per-ticker backtest output."""

    ticker: str
    equity_curve: pd.Series
    sharpe: float
    annual_return: float
    annual_vol: float
    max_drawdown: float


@dataclass
class BacktestResult:
    """Aggregate backtest output across all tickers."""

    per_ticker: dict[str, TickerPnL]
    portfolio_equity: pd.Series

    def metrics_table(self) -> pd.DataFrame:
        """Tidy DataFrame summarising per-ticker metrics."""
        rows = [
            {
                "ticker": t,
                "sharpe": p.sharpe,
                "annual_return": p.annual_return,
                "annual_vol": p.annual_vol,
                "max_drawdown": p.max_drawdown,
            }
            for t, p in self.per_ticker.items()
        ]
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def momentum_strategy(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Per-ticker 12-1 *time-series* momentum signal.

    Long (1) when the trailing ``lookback``-day return excluding the last
    ``skip`` days is positive; flat (0) otherwise. This is a single-asset
    time-series signal, not a cross-sectional rank across tickers.
    """
    close = prices["close"]
    if len(close) < lookback + 1:
        return pd.Series(0, index=close.index, dtype=int)
    signal = close.shift(skip) / close.shift(lookback) - 1.0
    return (signal > 0).astype(int).fillna(0)


def _backtest_one(args: tuple[str, pd.DataFrame, Strategy]) -> TickerPnL:
    """Compute per-ticker PnL with the no-look-ahead alignment convention.

    Position decided at the close of day ``t-1`` earns the return from
    ``t-1`` to ``t``. We therefore lag the strategy's positions by one
    period before aligning with the daily log returns.
    """
    ticker, prices, strategy = args
    if len(prices) < 5:
        raise InsufficientHistoryError(f"{ticker}: need >= 5 rows for backtest")
    positions = strategy(prices).reindex(prices.index).fillna(0).astype(float)
    rets = log_returns(prices["close"])
    lagged = positions.shift(1).reindex(rets.index).fillna(0)
    pnl = lagged * rets
    equity = cumulative_returns(pnl)
    annual_return = float(pnl.mean() * TRADING_DAYS_PER_YEAR)
    annual_vol = float(pnl.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else 0.0
    mdd = max_drawdown(equity)
    return TickerPnL(ticker=ticker, equity_curve=equity, sharpe=sharpe,
                     annual_return=annual_return, annual_vol=annual_vol, max_drawdown=mdd)


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy | None = None,
    n_workers: int = 1,
) -> BacktestResult:
    """Run a per-ticker backtest in parallel.

    Args:
        prices: Long-format price panel with ``date``, ``ticker``, ``close``.
        strategy: Callable mapping a single ticker's frame to a position series.
            Defaults to :func:`momentum_strategy`.
        n_workers: Number of parallel processes.
    """
    if prices.empty:
        raise InsufficientHistoryError("price panel is empty")
    strategy = strategy or momentum_strategy
    grouped = [
        (str(t), sub.set_index("date").sort_index(), strategy)
        for t, sub in prices.groupby("ticker")
    ]
    log.info("Backtest: %d tickers, %d workers", len(grouped), n_workers)

    if n_workers <= 1:
        results = [_backtest_one(g) for g in grouped]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_backtest_one, grouped)

    per_ticker = {r.ticker: r for r in results}
    equity_frame = pd.DataFrame({t: p.equity_curve for t, p in per_ticker.items()}).ffill()
    portfolio = equity_frame.mean(axis=1)
    return BacktestResult(per_ticker=per_ticker, portfolio_equity=portfolio)
