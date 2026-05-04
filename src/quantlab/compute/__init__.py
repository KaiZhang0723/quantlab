"""Computation subpackage: rolling stats, parallel backtesting, Monte Carlo VaR, DP, MapReduce."""

from __future__ import annotations

from quantlab.compute.backtest import BacktestResult, run_backtest
from quantlab.compute.montecarlo import historical_simulation, monte_carlo_var
from quantlab.compute.optimal_execution import max_profit_with_fee, max_profit_with_k_trades
from quantlab.compute.rolling import (
    cumulative_returns,
    drawdown_series,
    log_returns,
    rolling_sharpe,
    rolling_volatility,
)

__all__ = [
    "BacktestResult",
    "run_backtest",
    "historical_simulation",
    "monte_carlo_var",
    "max_profit_with_fee",
    "max_profit_with_k_trades",
    "cumulative_returns",
    "drawdown_series",
    "log_returns",
    "rolling_sharpe",
    "rolling_volatility",
]
