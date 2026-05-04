"""Typed exception hierarchy for quantlab.

All package-raised errors derive from :class:`QuantLabError` so callers can
catch the entire family with one ``except`` clause.
"""

from __future__ import annotations


class QuantLabError(Exception):
    """Base class for all quantlab errors."""


class DataSourceError(QuantLabError):
    """Raised when a price data source fails (network, parse, schema)."""


class MissingPriceDataError(DataSourceError):
    """Raised when a requested ticker / date range yields no rows."""


class InsufficientHistoryError(QuantLabError):
    """Raised when an analytic requires more observations than were supplied."""


class ConfigError(QuantLabError):
    """Raised when a YAML or CLI configuration is malformed."""
