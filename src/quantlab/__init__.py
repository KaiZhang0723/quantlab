"""quantlab: reproducible equity-analysis pipeline for Yahoo Finance data.

A Python package for fetching equity prices, computing streaming and rolling
risk metrics, running parallel backtests, and benchmarking simple ML forecasts.

The top-level namespace re-exports the most common entry points so users can
do ``from quantlab import YFinanceSource`` instead of digging through subpackages.
"""

from __future__ import annotations

__version__ = "0.1.0"

from quantlab.compute.backtest import momentum_strategy, run_backtest
from quantlab.data.cache import CSVCache
from quantlab.data.wiki_constituents import WikipediaConstituents
from quantlab.data.yfinance_source import YFinanceSource
from quantlab.exceptions import (
    DataSourceError,
    InsufficientHistoryError,
    MissingPriceDataError,
    QuantLabError,
)
from quantlab.streaming.median import RunningMedian
from quantlab.streaming.topk import TopK
from quantlab.streaming.welford import OnlineMoments

__all__ = [
    "__version__",
    "CSVCache",
    "WikipediaConstituents",
    "YFinanceSource",
    "DataSourceError",
    "InsufficientHistoryError",
    "MissingPriceDataError",
    "QuantLabError",
    "RunningMedian",
    "TopK",
    "OnlineMoments",
    "momentum_strategy",
    "run_backtest",
]
