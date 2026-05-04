"""Tests for the CLI entry point.

We invoke ``quantlab.cli.main`` directly with stub arguments to keep tests
in-process and deterministic; the underlying yfinance source is replaced via
monkeypatching.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

from quantlab.cli import _build_parser, load_config, main
from quantlab.exceptions import ConfigError


def _toy_panel(n_days: int = 400, tickers=("AAA", "BBB")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    frames = []
    for i, t in enumerate(tickers):
        rets = rng.normal(0.0003 + 0.0001 * i, 0.012, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "open": close,
                                    "high": close * 1.01, "low": close * 0.99,
                                    "close": close, "adj_close": close,
                                    "volume": rng.integers(1, 1_000_000, n_days)}))
    return pd.concat(frames, ignore_index=True)


class ParserTests(unittest.TestCase):
    def test_parser_requires_subcommand(self) -> None:
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])


class LoadConfigTests(unittest.TestCase):
    def test_returns_empty_when_path_is_none(self) -> None:
        self.assertEqual(load_config(None), {})

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(Path("/no/such/file.yaml"))

    def test_invalid_yaml_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text(":\n :\n   not: valid: yaml: ::")
            with self.assertRaises(ConfigError):
                load_config(p)


class CommandTests(unittest.TestCase):
    def test_backtest_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            prices_csv = Path(tmp) / "prices.csv"
            out_csv = Path(tmp) / "metrics.csv"
            _toy_panel().to_csv(prices_csv, index=False)
            rc = main(["backtest", "--prices", str(prices_csv),
                       "--workers", "1", "--out", str(out_csv),
                       "--lookback", "60", "--skip", "5"])
            self.assertEqual(rc, 0)
            self.assertTrue(out_csv.exists())

    def test_forecast_writes_json(self) -> None:
        with TemporaryDirectory() as tmp:
            prices_csv = Path(tmp) / "prices.csv"
            out_json = Path(tmp) / "fc.json"
            _toy_panel().to_csv(prices_csv, index=False)
            rc = main(["forecast", "--prices", str(prices_csv),
                       "--ticker", "AAA", "--task", "regression",
                       "--out", str(out_json)])
            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text())
            self.assertEqual(payload["ticker"], "AAA")
            self.assertIn("cv_mean", payload)

    def test_forecast_unknown_ticker_returns_error(self) -> None:
        with TemporaryDirectory() as tmp:
            prices_csv = Path(tmp) / "prices.csv"
            _toy_panel().to_csv(prices_csv, index=False)
            rc = main(["forecast", "--prices", str(prices_csv),
                       "--ticker", "ZZZ", "--out", str(Path(tmp) / "fc.json")])
            self.assertEqual(rc, 2)

    def test_var_runs_and_prints(self) -> None:
        with TemporaryDirectory() as tmp:
            prices_csv = Path(tmp) / "prices.csv"
            _toy_panel(n_days=300).to_csv(prices_csv, index=False)
            rc = main(["var", "--prices", str(prices_csv), "--ticker", "AAA",
                       "--paths", "1000", "--horizon", "5", "--workers", "1"])
            self.assertEqual(rc, 0)

    def test_fetch_uses_injected_source(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "prices.csv"
            cache = Path(tmp) / ".cache"
            with mock.patch("quantlab.cli.YFinanceSource") as mock_yf:
                instance = mock.MagicMock()
                instance.fetch.return_value = _toy_panel(n_days=20)
                mock_yf.return_value = instance
                rc = main(["fetch", "--tickers", "AAA", "BBB",
                           "--start", "2024-01-01", "--end", "2024-01-31",
                           "--out", str(out), "--cache-dir", str(cache)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
