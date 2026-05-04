"""Generate the committed sample artefacts under ``reports/``.

By default the script reads ``reports/sample_prices.csv`` (real Yahoo
Finance data fetched via ``quantlab fetch``) if present, otherwise falls
back to a deterministic synthetic GBM panel so the script needs no
network access. Pass ``--prices PATH`` to point at a different CSV, or
``--use-synthetic`` to force the synthetic fallback.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from quantlab._logging import configure, get_logger
from quantlab.compute.backtest import momentum_strategy, run_backtest
from quantlab.compute.montecarlo import historical_simulation, monte_carlo_var
from quantlab.compute.optimal_execution import max_profit_with_fee
from quantlab.compute.rolling import log_returns
from quantlab.data.cache import CSVCache
from quantlab.data.yfinance_source import YFinanceSource
from quantlab.models.evaluation import classification_metrics
from quantlab.models.features import build_features
from quantlab.models.forecaster import ReturnForecaster
from quantlab.viz.correlation import plot_correlation_heatmap
from quantlab.viz.drawdown import plot_drawdown
from quantlab.viz.returns import plot_cumulative_returns

log = get_logger("quantlab.reports")

REPORTS = Path(__file__).resolve().parent.parent / "reports"
DEFAULT_CSV = REPORTS / "sample_prices.csv"
TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "WMT")
DEFAULT_START = date(2019, 1, 1)
DEFAULT_END = date(2024, 12, 31)


def synthetic_panel(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion panel that mimics 2019–2024 daily data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    frames = []
    for i, t in enumerate(TICKERS):
        mu = 0.0004 + 0.0001 * (i % 3)
        sigma = 0.011 + 0.001 * (i % 5)
        rets = rng.normal(loc=mu, scale=sigma, size=n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "adj_close": close,
            "volume": rng.integers(10_000, 5_000_000, n_days),
        }))
    return pd.concat(frames, ignore_index=True)


def fetch_real_panel(start: date = DEFAULT_START, end: date = DEFAULT_END) -> pd.DataFrame:
    """Fetch real Yahoo Finance data via ``YFinanceSource`` (cached on disk)."""
    src = CSVCache(YFinanceSource(), root=REPORTS.parent / ".cache" / "yfinance")
    return src.fetch(TICKERS, start, end)


def load_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Resolve which price panel to use; return ``(panel, label)``."""
    if args.use_synthetic:
        return synthetic_panel(), "synthetic GBM panel (forced)"
    if args.fetch:
        log.info("fetching real Yahoo Finance data for %d tickers", len(TICKERS))
        panel = fetch_real_panel()
        REPORTS.mkdir(parents=True, exist_ok=True)
        panel.to_csv(DEFAULT_CSV, index=False)
        return panel, f"real Yahoo Finance ({DEFAULT_CSV})"
    src_path = args.prices or DEFAULT_CSV
    if src_path.exists():
        panel = pd.read_csv(src_path, parse_dates=["date"])
        return panel, f"loaded from {src_path}"
    log.info("no CSV at %s; falling back to synthetic panel", src_path)
    return synthetic_panel(), "synthetic GBM panel (no CSV found)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--prices", type=Path, default=None,
                   help="Path to a long-format prices CSV.")
    g.add_argument("--fetch", action="store_true",
                   help="Fetch fresh data via yfinance and save to reports/sample_prices.csv.")
    g.add_argument("--use-synthetic", action="store_true",
                   help="Force the deterministic synthetic panel.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    configure()
    REPORTS.mkdir(parents=True, exist_ok=True)

    panel, label = load_panel(args)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    log.info("price panel: %s, %d rows, %d tickers, %d dates",
             label, len(panel), panel["ticker"].nunique(), panel["date"].nunique())
    if not args.fetch:
        panel.to_csv(DEFAULT_CSV, index=False)

    bt = run_backtest(panel, strategy=lambda d: momentum_strategy(d, lookback=252, skip=21),
                      n_workers=1)
    metrics = bt.metrics_table()
    metrics.to_csv(REPORTS / "backtest_metrics.csv", index=False)

    fig = plot_cumulative_returns(bt.portfolio_equity.rename("portfolio"),
                                  title="Equal-weight 12-1 momentum portfolio")
    fig.savefig(REPORTS / "cumulative_returns.png", dpi=120)

    fig = plot_drawdown(bt.portfolio_equity, title="Portfolio drawdown")
    fig.savefig(REPORTS / "drawdown.png", dpi=120)

    rets_wide = pd.DataFrame({
        t: log_returns(sub.set_index("date")["close"])
        for t, sub in panel.groupby("ticker")
    }).dropna()
    fig = plot_correlation_heatmap(rets_wide, title="Daily return correlations")
    fig.savefig(REPORTS / "correlation_heatmap.png", dpi=120)

    apple = panel[panel["ticker"] == "AAPL"].set_index("date").sort_index()
    feats = build_features(apple)
    fc = ReturnForecaster(task="classification", n_splits=5)
    fit = fc.fit(feats)
    proba = fc.predict_proba(feats)
    cls_metrics = classification_metrics(feats["y_next_dir"], proba)

    rets = log_returns(apple["close"]).dropna()
    mc = monte_carlo_var(mu=float(rets.mean()), sigma=float(rets.std(ddof=1)),
                         horizon_days=10, n_paths=20_000, confidence=0.99,
                         n_workers=1, seed=0)
    hist = historical_simulation(rets.values, confidence=0.99)

    upper_bound = max_profit_with_fee(apple["close"].tolist(), fee=0.001)

    summary = {
        "data_source": label,
        "n_tickers": int(panel["ticker"].nunique()),
        "n_dates": int(panel["date"].nunique()),
        "date_range": [str(panel["date"].min()), str(panel["date"].max())],
        "portfolio": {
            "sharpe": float(metrics["sharpe"].mean()),
            "annual_return": float(metrics["annual_return"].mean()),
            "annual_vol": float(metrics["annual_vol"].mean()),
            "max_drawdown": float(metrics["max_drawdown"].mean()),
        },
        "aapl_forecast": {
            "task": "classification",
            "cv_auc_mean": fit.cv_mean,
            "in_sample": cls_metrics,
        },
        "aapl_var_99": {
            "monte_carlo": {"var": mc.var, "cvar": mc.cvar, "n_paths": mc.n_paths},
            "historical":  {"var": hist.var, "cvar": hist.cvar, "n_paths": hist.n_paths},
        },
        "aapl_perfect_foresight_profit_per_share_with_fee_0.001": upper_bound,
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote reports to %s", REPORTS)


if __name__ == "__main__":
    run(_parse_args())
