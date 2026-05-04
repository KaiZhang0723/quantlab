"""Abstract interface for price data sources.

Concrete implementations live in this subpackage. Tests rely on the ABC so
they can swap real network sources for in-memory fakes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

EXPECTED_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class PriceQuery:
    """Immutable query parameters passed to a :class:`PriceSource`."""

    tickers: tuple[str, ...]
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("PriceQuery requires at least one ticker")
        if self.end < self.start:
            raise ValueError(f"end {self.end} precedes start {self.start}")


class PriceSource(ABC):
    """Abstract base class for any source that returns OHLCV price data.

    Implementations must return a long-format DataFrame with columns
    listed in :data:`EXPECTED_COLUMNS`. The ``date`` column is timezone-naive
    and the rows are sorted by (ticker, date) ascending.
    """

    @abstractmethod
    def fetch(self, tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
        """Return OHLCV history for the supplied tickers / date range."""

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Validate that ``df`` matches the expected schema; return it unchanged.

        Raises:
            ValueError: If columns are missing or out of order.
        """
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"price frame missing columns: {missing}")
        return df
