"""On-disk CSV cache for price panels.

Wraps any :class:`PriceSource` with a CSV-backed cache that serves repeat
requests from disk. The TTL semantics borrow from HW2's caching mechanism,
simplified for the file-system case (one CSV per ticker).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from quantlab._logging import get_logger
from quantlab.data.base import PriceSource

log = get_logger(__name__)


class CSVCache(PriceSource):
    """File-system cache that wraps an upstream :class:`PriceSource`.

    Each ticker is stored as ``<root>/<ticker>.csv``. A row's age is computed
    from the file's mtime; if older than ``ttl_seconds`` the cache misses and
    refetches from the upstream source.

    Args:
        upstream: The wrapped source to call on cache misses.
        root: Directory holding the cached CSV files.
        ttl_seconds: Cache lifetime per ticker. ``None`` disables expiry.
    """

    def __init__(self, upstream: PriceSource, root: Path, ttl_seconds: int | None = 24 * 3600) -> None:
        self._upstream = upstream
        self._root = Path(root)
        self._ttl = ttl_seconds
        self._root.mkdir(parents=True, exist_ok=True)

    def fetch(self, tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
        ticker_list = list(tickers)
        cached, missing = self._partition(ticker_list, start, end)

        frames: list[pd.DataFrame] = []
        for tkr in cached:
            frames.append(self._read(tkr))
            log.debug("cache hit: %s", tkr)

        if missing:
            log.info("cache miss for %d tickers", len(missing))
            fresh = self._upstream.fetch(missing, start, end)
            for tkr, sub in fresh.groupby("ticker"):
                self._write(str(tkr), sub)
                frames.append(sub)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        mask = (merged["date"] >= pd.Timestamp(start)) & (merged["date"] <= pd.Timestamp(end))
        return merged.loc[mask].sort_values(["ticker", "date"]).reset_index(drop=True)

    def _partition(self, tickers: list[str], start: date, end: date) -> tuple[list[str], list[str]]:
        hits, misses = [], []
        for tkr in tickers:
            path = self._path(tkr)
            if path.exists() and not self._expired(path) and self._covers(path, start, end):
                hits.append(tkr)
            else:
                misses.append(tkr)
        return hits, misses

    def _expired(self, path: Path) -> bool:
        if self._ttl is None:
            return False
        return (time.time() - path.stat().st_mtime) > self._ttl

    @staticmethod
    def _covers(path: Path, start: date, end: date) -> bool:
        """True if the cached file's date range envelopes ``[start, end]``."""
        try:
            dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"]
        except Exception:
            return False
        if dates.empty:
            return False
        return dates.min() <= pd.Timestamp(start) and dates.max() >= pd.Timestamp(end)

    def _path(self, ticker: str) -> Path:
        return self._root / f"{ticker}.csv"

    def _read(self, ticker: str) -> pd.DataFrame:
        df = pd.read_csv(self._path(ticker), parse_dates=["date"])
        df["ticker"] = ticker
        return df

    def _write(self, ticker: str, df: pd.DataFrame) -> None:
        out = df.drop(columns=["ticker"], errors="ignore")
        out.to_csv(self._path(ticker), index=False)
