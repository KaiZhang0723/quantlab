"""Tests for the data subpackage."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.base import EXPECTED_COLUMNS, PriceQuery, PriceSource
from quantlab.data.cache import CSVCache
from quantlab.data.sql_layer import SQLAnalytics
from quantlab.data.wiki_constituents import WikipediaConstituents
from quantlab.data.yfinance_source import YFinanceSource
from quantlab.exceptions import DataSourceError, MissingPriceDataError

FIXTURE_HTML = (Path(__file__).parent / "fixtures" / "wiki_sp500_sample.html").read_text()


def _wide_yfinance_response(tickers: list[str], n_days: int = 10) -> pd.DataFrame:
    """Mimic yfinance's MultiIndex wide format."""
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    data = np.random.default_rng(0).normal(loc=100, scale=1, size=(n_days, len(cols)))
    df = pd.DataFrame(data, index=idx, columns=cols)
    df.index.name = "Date"
    return df


class PriceQueryTests(unittest.TestCase):
    def test_requires_tickers(self) -> None:
        with self.assertRaises(ValueError):
            PriceQuery(tickers=(), start=date(2020, 1, 1), end=date(2020, 12, 31))

    def test_rejects_inverted_dates(self) -> None:
        with self.assertRaises(ValueError):
            PriceQuery(tickers=("A",), start=date(2024, 1, 1), end=date(2023, 1, 1))


class PriceSourceValidateTests(unittest.TestCase):
    def test_validate_passes_with_full_schema(self) -> None:
        df = pd.DataFrame({c: [1] for c in EXPECTED_COLUMNS})
        out = PriceSource.validate(df)
        self.assertIs(out, df)

    def test_validate_raises_on_missing_columns(self) -> None:
        with self.assertRaises(ValueError):
            PriceSource.validate(pd.DataFrame({"date": [1], "close": [2]}))


class YFinanceSourceTests(unittest.TestCase):
    def test_normalises_multiindex_response(self) -> None:
        tickers = ["AAA", "BBB"]
        src = YFinanceSource(downloader=lambda **kw: _wide_yfinance_response(tickers))
        df = src.fetch(tickers, date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(set(EXPECTED_COLUMNS), set(df.columns))
        self.assertEqual(set(tickers), set(df["ticker"].unique()))

    def test_handles_single_index_response(self) -> None:
        idx = pd.bdate_range("2024-01-02", periods=5)
        wide = pd.DataFrame(
            np.random.default_rng(0).normal(loc=100, scale=1, size=(5, 6)),
            index=idx,
            columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )
        wide.index.name = "Date"
        src = YFinanceSource(downloader=lambda **kw: wide)
        df = src.fetch(["SOLO"], date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual(["SOLO"], df["ticker"].unique().tolist())

    def test_empty_response_raises(self) -> None:
        src = YFinanceSource(downloader=lambda **kw: pd.DataFrame())
        with self.assertRaises(MissingPriceDataError):
            src.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 31))

    def test_downloader_error_wrapped(self) -> None:
        def bad(**_kw):
            raise RuntimeError("boom")
        src = YFinanceSource(downloader=bad)
        with self.assertRaises(DataSourceError):
            src.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 31))

    def test_rejects_empty_tickers(self) -> None:
        src = YFinanceSource(downloader=lambda **kw: pd.DataFrame())
        with self.assertRaises(ValueError):
            src.fetch([], date(2024, 1, 1), date(2024, 1, 31))

    def test_rejects_single_index_response_for_multi_ticker_request(self) -> None:
        # yfinance occasionally returns a flat-column frame when only one of
        # the requested tickers survives. We refuse to guess which one.
        idx = pd.bdate_range("2024-01-02", periods=5)
        flat = pd.DataFrame(
            np.random.default_rng(0).normal(loc=100, scale=1, size=(5, 6)),
            index=idx,
            columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        )
        flat.index.name = "Date"
        src = YFinanceSource(downloader=lambda **kw: flat)
        with self.assertRaises(MissingPriceDataError):
            src.fetch(["AAA", "BBB", "CCC"], date(2024, 1, 1), date(2024, 1, 31))


class WikipediaConstituentsTests(unittest.TestCase):
    def test_parse_extracts_rows(self) -> None:
        df = WikipediaConstituents.parse(FIXTURE_HTML)
        self.assertGreaterEqual(len(df), 5)
        self.assertEqual({"symbol", "security", "sector", "sub_industry"}, set(df.columns))
        self.assertIn("AAPL", df["symbol"].tolist())

    def test_parse_normalises_dot_to_dash(self) -> None:
        df = WikipediaConstituents.parse(FIXTURE_HTML)
        self.assertIn("BRK-B", df["symbol"].tolist())

    def test_parse_rejects_html_without_table(self) -> None:
        with self.assertRaises(DataSourceError):
            WikipediaConstituents.parse("<html><body>nope</body></html>")

    def test_fetch_uses_injected_fetcher(self) -> None:
        wiki = WikipediaConstituents(fetcher=lambda url: FIXTURE_HTML)
        df = wiki.fetch("https://example.com/whatever")
        self.assertGreater(len(df), 0)

    def test_fetcher_failure_wrapped(self) -> None:
        def bad(_url):
            raise RuntimeError("network down")
        with self.assertRaises(DataSourceError):
            WikipediaConstituents(fetcher=bad).fetch("http://example.com")


class _CountingSource(PriceSource):
    """Test stub that records how many times ``fetch`` is called."""

    def __init__(self, dates: pd.DatetimeIndex) -> None:
        self.calls = 0
        self._dates = dates

    def fetch(self, tickers, start, end):
        self.calls += 1
        rows = []
        for t in tickers:
            rows.append(pd.DataFrame({
                "date": self._dates, "ticker": t, "open": 1.0, "high": 1.0,
                "low": 1.0, "close": 1.0, "adj_close": 1.0, "volume": 100,
            }))
        return pd.concat(rows, ignore_index=True)


class CSVCacheTests(unittest.TestCase):
    def test_cache_round_trip_serves_repeat_request(self) -> None:
        # Request matches cached range exactly, so the second call is a hit.
        import tempfile
        ts = pd.bdate_range("2024-01-02", periods=10)
        start, end = date(2024, 1, 2), date(2024, 1, 15)
        with tempfile.TemporaryDirectory() as tmp:
            src = _CountingSource(ts)
            cache = CSVCache(src, root=Path(tmp), ttl_seconds=3600)
            df1 = cache.fetch(["AAA", "BBB"], start, end)
            df2 = cache.fetch(["AAA", "BBB"], start, end)
            self.assertEqual(src.calls, 1)
            self.assertEqual(len(df1), len(df2))

    def test_cache_misses_when_request_exceeds_cached_range(self) -> None:
        # Cached file covers Jan 2024 only; a follow-up request for the
        # full year should NOT be a hit (B5 fix). Otherwise the user
        # silently receives an incomplete date range.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            narrow = pd.bdate_range("2024-01-02", periods=10)  # 10 January days
            src = _CountingSource(narrow)
            cache = CSVCache(src, root=Path(tmp), ttl_seconds=3600)
            cache.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 31))
            self.assertEqual(src.calls, 1)
            # Now request a wider range; the cache must miss and refetch.
            cache.fetch(["AAA"], date(2024, 1, 1), date(2024, 12, 31))
            self.assertEqual(src.calls, 2)


class SQLAnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ts = pd.bdate_range("2024-01-02", periods=30)
        # Distinct close-price paths so cross_sectional_rank produces distinct ranks.
        cls.df = pd.concat([
            pd.DataFrame({"date": ts, "ticker": "AAA", "close": np.linspace(100, 130, 30),
                          "volume": np.arange(30) + 1, "open": 100.0, "high": 130.0,
                          "low": 90.0, "adj_close": 100.0}),
            pd.DataFrame({"date": ts, "ticker": "BBB", "close": np.linspace(150, 200, 30),
                          "volume": np.arange(30) * 2 + 1, "open": 150.0, "high": 200.0,
                          "low": 140.0, "adj_close": 150.0}),
        ], ignore_index=True)

    def test_rolling_avg_volume_window_function(self) -> None:
        with SQLAnalytics(self.df) as sql:
            out = sql.rolling_avg_volume(window=5)
        self.assertIn("avg_volume_5d", out.columns)
        self.assertEqual(len(out), len(self.df))

    def test_cross_sectional_rank_per_date(self) -> None:
        with SQLAnalytics(self.df) as sql:
            ranks = sql.cross_sectional_rank()
        groups = ranks.groupby("date")["rank_close"].apply(lambda s: sorted(s.tolist()))
        for day in groups:
            self.assertEqual(day, list(range(1, len(day) + 1)))

    def test_momentum_signal(self) -> None:
        with SQLAnalytics(self.df) as sql:
            mom = sql.momentum_signal(lookback=5)
        self.assertIn("momentum_5d", mom.columns)

    def test_rejects_empty_panel(self) -> None:
        with self.assertRaises(ValueError):
            SQLAnalytics(pd.DataFrame())

    def test_rejects_invalid_window(self) -> None:
        with SQLAnalytics(self.df) as sql:
            with self.assertRaises(ValueError):
                sql.rolling_avg_volume(window=0)
            with self.assertRaises(ValueError):
                sql.momentum_signal(lookback=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
