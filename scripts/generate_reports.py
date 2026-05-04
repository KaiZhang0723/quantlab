"""Generate the committed sample artefacts under ``reports/``.

Runs end-to-end on a deterministic synthetic price panel so the script
needs no network access. Real Yahoo Finance data can be substituted by
running the CLI ``quantlab fetch`` first and pointing this script at the
resulting CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from quantlab._logging import configure
from quantlab.compute.backtest import momentum_strategy, run_backtest
from quantlab.compute.montecarlo import historical_simulation, monte_carlo_var
from quantlab.compute.optimal_execution import max_profit_with_fee
from quantlab.compute.rolling import log_returns
from quantlab.models.evaluation import classification_metrics
from quantlab.models.features import build_features
from quantlab.models.forecaster import ReturnForecaster
from quantlab.viz.correlation import plot_correlation_heatmap
from quantlab.viz.drawdown import plot_drawdown
from quantlab.viz.returns import plot_cumulative_returns

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "WMT")


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


def run() -> None:
    configure()
    REPORTS.mkdir(parents=True, exist_ok=True)
    panel = synthetic_panel()
    panel.to_csv(REPORTS / "sample_prices.csv", index=False)

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
        "n_tickers": len(TICKERS),
        "n_dates": int(panel["date"].nunique()),
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


if __name__ == "__main__":
    run()
