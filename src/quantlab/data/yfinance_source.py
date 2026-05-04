"""Primary price source backed by the ``yfinance`` library.

We intentionally do NOT make network calls in the test suite; tests inject a
fake downloader via the ``downloader`` constructor argument so CI is
deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date

import pandas as pd

from quantlab._logging import get_logger
from quantlab.data.base import EXPECTED_COLUMNS, PriceSource
from quantlab.exceptions import DataSourceError, MissingPriceDataError

log = get_logger(__name__)

Downloader = Callable[..., pd.DataFrame]


def _default_downloader() -> Downloader:
    """Lazy-import yfinance so the test path needs no network library."""
    import yfinance as yf

    return yf.download


class YFinanceSource(PriceSource):
    """Fetch historical OHLCV data via the ``yfinance`` Yahoo Finance client.

    Args:
        downloader: Callable conforming to ``yfinance.download``. Defaults to the
            real client; tests pass a stub returning a fixture DataFrame.
        auto_adjust: If ``True``, request split/dividend-adjusted prices.

    Example:
        >>> src = YFinanceSource(downloader=lambda **kw: _fake_yf_response())
        >>> isinstance(src, YFinanceSource)
        True
    """

    def __init__(self, downloader: Downloader | None = None, auto_adjust: bool = False) -> None:
        self._downloader = downloader or _default_downloader()
        self._auto_adjust = auto_adjust

    def fetch(self, tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
        ticker_list = list(tickers)
        if not ticker_list:
            raise ValueError("at least one ticker required")
        log.info("fetching %d tickers from yfinance: %s", len(ticker_list), ticker_list[:5])
        try:
            raw = self._downloader(
                tickers=ticker_list,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=self._auto_adjust,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            raise DataSourceError(f"yfinance download failed: {exc}") from exc

        if raw is None or len(raw) == 0:
            raise MissingPriceDataError(f"no data returned for {ticker_list}")

        return self._normalise(raw, ticker_list)

    @staticmethod
    def _normalise(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Reshape yfinance's wide multi-index frame into the long schema."""
        frames: list[pd.DataFrame] = []

        if isinstance(raw.columns, pd.MultiIndex):
            for tkr in tickers:
                if tkr not in raw.columns.get_level_values(0):
                    log.warning("ticker %s missing from yfinance response", tkr)
                    continue
                sub = raw[tkr].copy()
                sub["ticker"] = tkr
                frames.append(sub)
        else:
            sub = raw.copy()
            sub["ticker"] = tickers[0]
            frames.append(sub)

        if not frames:
            raise MissingPriceDataError(f"no usable rows for {tickers}")

        long = pd.concat(frames).reset_index()
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        long = long.rename(columns={"index": "date"})

        if "adj_close" not in long.columns and "close" in long.columns:
            long["adj_close"] = long["close"]

        for col in EXPECTED_COLUMNS:
            if col not in long.columns:
                long[col] = pd.NA

        long = long[list(EXPECTED_COLUMNS)].dropna(subset=["close"])
        long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
        long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
        return PriceSource.validate(long)
