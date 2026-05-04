"""In-memory SQLite analytics layer.

Loads a long-format price panel into a temporary SQLite database and exposes
a few canned analytical queries that demonstrate W9–W10 SQL material —
notably window functions (``OVER (PARTITION BY ... ORDER BY ...)``) for
rolling rank and running totals. Per Ed forum #82, SQLite is not required
for the project, so this module is a pedagogical layer over CSV data, not the
source of truth.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

import pandas as pd


PRICE_TABLE = "prices"


class SQLAnalytics:
    """Wrap a pandas price panel with a transient in-memory SQLite database.

    The connection is opened in the constructor and closed by :meth:`close` or
    by using the instance as a context manager.

    Args:
        prices: Long-format DataFrame with at minimum ``date``, ``ticker``,
            ``close``, ``volume``.
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        if prices.empty:
            raise ValueError("SQLAnalytics requires a non-empty price panel")
        self._conn = sqlite3.connect(":memory:")
        self._load(prices)

    def __enter__(self) -> "SQLAnalytics":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _load(self, prices: pd.DataFrame) -> None:
        df = prices.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_sql(PRICE_TABLE, self._conn, index=False, if_exists="replace")
        with self._cursor() as cur:
            cur.execute(f"CREATE INDEX idx_prices_ticker_date ON {PRICE_TABLE}(ticker, date)")

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    def query(self, sql: str) -> pd.DataFrame:
        """Execute an arbitrary SQL string and return the result as a DataFrame."""
        return pd.read_sql_query(sql, self._conn)

    def rolling_avg_volume(self, window: int = 20) -> pd.DataFrame:
        """Per-ticker rolling average volume using a SQL window function."""
        if window < 1:
            raise ValueError("window must be >= 1")
        sql = f"""
            SELECT
                date,
                ticker,
                volume,
                AVG(volume) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                    ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                ) AS avg_volume_{window}d
            FROM {PRICE_TABLE}
            ORDER BY ticker, date
        """
        return self.query(sql)

    def cross_sectional_rank(self) -> pd.DataFrame:
        """For each date, rank tickers by close price (descending)."""
        sql = f"""
            SELECT
                date,
                ticker,
                close,
                RANK() OVER (PARTITION BY date ORDER BY close DESC) AS rank_close
            FROM {PRICE_TABLE}
            ORDER BY date, rank_close
        """
        return self.query(sql)

    def momentum_signal(self, lookback: int = 252) -> pd.DataFrame:
        """12-month (configurable) momentum: close / lagged close - 1.

        Uses a window function with ``LAG`` to retrieve the prior close.
        """
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        sql = f"""
            SELECT
                date,
                ticker,
                close,
                LAG(close, {lookback}) OVER (PARTITION BY ticker ORDER BY date) AS lagged_close,
                close / NULLIF(LAG(close, {lookback}) OVER (PARTITION BY ticker ORDER BY date), 0) - 1
                    AS momentum_{lookback}d
            FROM {PRICE_TABLE}
            ORDER BY ticker, date
        """
        return self.query(sql)
