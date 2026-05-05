"""Command-line interface for quantlab.

Sub-commands::

    quantlab fetch     --tickers AAPL MSFT --start 2020-01-01 --end 2024-12-31 --out prices.csv
    quantlab backtest  --prices prices.csv --workers 4 --out reports/metrics.csv
    quantlab forecast  --prices prices.csv --task classification --out reports/forecast.csv
    quantlab var       --prices prices.csv --confidence 0.99 --paths 50000
    quantlab sql       --prices prices.csv --query rank --limit 10

A YAML config file (``--config configs/default.yaml``) supplies the
``tickers``, ``start``, and ``end`` defaults consumed by ``fetch``. Other
subcommands read all parameters from CLI flags directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from quantlab._logging import configure, get_logger
from quantlab.compute.backtest import momentum_strategy, run_backtest
from quantlab.compute.montecarlo import historical_simulation, monte_carlo_var
from quantlab.compute.rolling import log_returns
from quantlab.data.cache import CSVCache
from quantlab.data.sql_layer import SQLAnalytics
from quantlab.data.yfinance_source import YFinanceSource
from quantlab.exceptions import ConfigError, QuantLabError
from quantlab.models.evaluation import classification_metrics, regression_metrics
from quantlab.models.features import build_features
from quantlab.models.forecaster import ReturnForecaster

log = get_logger("quantlab.cli")


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab", description="Reproducible equity-analysis pipeline.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING)")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download price history into a CSV.")
    fetch.add_argument("--tickers", nargs="+")
    fetch.add_argument("--start", type=date.fromisoformat)
    fetch.add_argument("--end", type=date.fromisoformat)
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--cache-dir", type=Path, default=Path(".cache"))

    bt = sub.add_parser("backtest", help="Run a parallel momentum backtest.")
    bt.add_argument("--prices", type=Path, required=True)
    bt.add_argument("--workers", type=int, default=1)
    bt.add_argument("--out", type=Path, required=True)
    bt.add_argument("--lookback", type=int, default=252)
    bt.add_argument("--skip", type=int, default=21)

    fc = sub.add_parser("forecast", help="Train a single-ticker forecaster.")
    fc.add_argument("--prices", type=Path, required=True)
    fc.add_argument("--ticker", required=True)
    fc.add_argument("--task", choices=("regression", "classification"), default="regression")
    fc.add_argument("--out", type=Path, required=True)

    var = sub.add_parser("var", help="Estimate VaR / CVaR (Monte Carlo + historical).")
    var.add_argument("--prices", type=Path, required=True)
    var.add_argument("--ticker", required=True)
    var.add_argument("--confidence", type=float, default=0.99)
    var.add_argument("--paths", type=int, default=50_000)
    var.add_argument("--horizon", type=int, default=10)
    var.add_argument("--workers", type=int, default=1)

    sql = sub.add_parser("sql", help="Run analytical SQL window-function queries on a price panel.")
    sql.add_argument("--prices", type=Path, required=True)
    sql.add_argument("--query", choices=("rank", "rolling_volume", "momentum"), default="rank")
    sql.add_argument("--window", type=int, default=20, help="Window for rolling_volume.")
    sql.add_argument("--lookback", type=int, default=252, help="Lookback for momentum.")
    sql.add_argument("--limit", type=int, default=20, help="Rows to print.")

    return parser


def _load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "ticker" not in df.columns:
        raise ConfigError("price file must contain 'ticker' column")
    return df


def _cmd_fetch(args: argparse.Namespace, cfg: dict) -> int:
    tickers = args.tickers or cfg.get("tickers")
    start = args.start or (date.fromisoformat(cfg["start"]) if "start" in cfg else None)
    end = args.end or (date.fromisoformat(cfg["end"]) if "end" in cfg else None)
    if not tickers or start is None or end is None:
        raise ConfigError("fetch requires --tickers, --start, --end (or values in --config)")
    src = CSVCache(YFinanceSource(), root=args.cache_dir)
    df = src.fetch(tickers, start, end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %d rows -> %s", len(df), args.out)
    return 0


def _cmd_backtest(args: argparse.Namespace, _cfg: dict) -> int:
    prices = _load_prices(args.prices)
    # ``functools.partial`` of a top-level function is pickleable, so it
    # works under multiprocessing's ``spawn`` start method (default on
    # macOS / Windows). A nested ``def strategy(...)`` would not.
    strategy = partial(momentum_strategy, lookback=args.lookback, skip=args.skip)
    result = run_backtest(prices, strategy=strategy, n_workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.metrics_table().to_csv(args.out, index=False)
    log.info("backtest metrics -> %s", args.out)
    return 0


def _cmd_forecast(args: argparse.Namespace, _cfg: dict) -> int:
    prices = _load_prices(args.prices)
    sub = prices[prices["ticker"] == args.ticker].set_index("date").sort_index()
    if sub.empty:
        raise ConfigError(f"no rows for ticker {args.ticker}")
    feats = build_features(sub)
    fc = ReturnForecaster(task=args.task)
    res = fc.fit(feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "ticker": args.ticker,
        "task": args.task,
        "cv_mean": res.cv_mean,
        "cv_scores": res.cv_scores.tolist(),
        "n_obs": res.n_obs,
        "n_features": res.n_features,
    }
    if args.task == "regression":
        preds = fc.predict(feats)
        summary.update(regression_metrics(feats["y_next_ret"], preds))
    else:
        proba = fc.predict_proba(feats)
        summary.update(classification_metrics(feats["y_next_dir"], proba))
    args.out.write_text(json.dumps(summary, indent=2))
    log.info("forecast summary -> %s", args.out)
    return 0


def _cmd_var(args: argparse.Namespace, _cfg: dict) -> int:
    prices = _load_prices(args.prices)
    sub = prices[prices["ticker"] == args.ticker].sort_values("date")
    if sub.empty:
        raise ConfigError(f"no rows for ticker {args.ticker}")
    rets = log_returns(sub.set_index("date")["close"]).dropna()
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))
    mc = monte_carlo_var(mu=mu, sigma=sigma, horizon_days=args.horizon,
                         n_paths=args.paths, confidence=args.confidence,
                         n_workers=args.workers)
    hist = historical_simulation(rets.values, confidence=args.confidence,
                                 horizon_days=args.horizon)
    print(f"Monte Carlo ({args.horizon}-day): VaR={mc.var:.4%}  CVaR={mc.cvar:.4%}  paths={mc.n_paths}")
    print(f"Historical  ({args.horizon}-day): VaR={hist.var:.4%}  CVaR={hist.cvar:.4%}  obs={hist.n_paths}")
    return 0


def _cmd_sql(args: argparse.Namespace, _cfg: dict) -> int:
    prices = _load_prices(args.prices)
    with SQLAnalytics(prices) as sql:
        if args.query == "rank":
            df = sql.cross_sectional_rank()
        elif args.query == "rolling_volume":
            df = sql.rolling_avg_volume(window=args.window)
        else:
            df = sql.momentum_signal(lookback=args.lookback)
    print(df.head(args.limit).to_string(index=False))
    return 0


_COMMANDS = {
    "fetch": _cmd_fetch,
    "backtest": _cmd_backtest,
    "forecast": _cmd_forecast,
    "var": _cmd_var,
    "sql": _cmd_sql,
}


def _resolve_log_level(name: str) -> int:
    """Map an upper- or lower-case level name to a ``logging`` constant."""
    level = getattr(logging, name.upper(), None)
    if not isinstance(level, int):
        raise ConfigError(f"unknown log level: {name!r}")
    return level


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        configure(level=_resolve_log_level(args.log_level))
        cfg = load_config(args.config)
        handler = _COMMANDS.get(args.command)
        if handler is None:
            # Practically unreachable given argparse's ``required=True``,
            # but raising lets the surrounding try/except convert this to
            # the package's standard exit code 2 instead of argparse's
            # direct ``sys.exit``.
            raise ConfigError(f"unknown command {args.command}")
        return handler(args, cfg)
    except QuantLabError as exc:
        log.error("quantlab error: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
