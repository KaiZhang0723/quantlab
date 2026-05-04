"""Dynamic-programming benchmarks for trading on a known price path.

Given a *historical* price path, what is the maximum achievable profit under
perfect foresight? The answer is a useful upper bound for ML-based forecasts.
Both classics from W7D1 are here:

* :func:`max_profit_with_fee` — at most one position at a time, transaction
  cost ``fee`` per round-trip.
* :func:`max_profit_with_k_trades` — at most ``k`` round-trips, no fee.

These are the textbook "Best Time to Buy and Sell Stock IV" / "with Fee"
LeetCode problems, which are pure DP.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from quantlab.exceptions import InsufficientHistoryError


def max_profit_with_fee(prices: Sequence[float], fee: float = 0.0) -> float:
    """Maximum profit over a price path with a per-trade ``fee``.

    Two-state DP: ``cash`` (no position) and ``hold`` (long one share).
    Runs in O(n) time and O(1) space.

    Example:
        >>> max_profit_with_fee([1, 3, 2, 8, 4, 9], fee=2)
        8.0
    """
    if len(prices) == 0:
        raise InsufficientHistoryError("max_profit_with_fee requires at least 1 price")
    if fee < 0:
        raise ValueError("fee must be non-negative")
    cash = 0.0
    hold = -float(prices[0])
    for p in prices[1:]:
        cash = max(cash, hold + p - fee)
        hold = max(hold, cash - p)
    return cash


def max_profit_with_k_trades(prices: Sequence[float], k: int) -> float:
    """Maximum profit with at most ``k`` non-overlapping round-trips.

    Standard 2D DP: ``buy[j]`` and ``sell[j]`` for j = 1..k transactions.
    Runs in O(n*k) time and O(k) space.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    n = len(prices)
    if n < 2 or k == 0:
        return 0.0
    if k >= n // 2:
        return _greedy_unlimited(prices)
    buy = [-float(prices[0])] * (k + 1)
    sell = [0.0] * (k + 1)
    for p in prices[1:]:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    return sell[k]


def _greedy_unlimited(prices: Sequence[float]) -> float:
    """When unlimited trades are allowed the optimum is the sum of positive diffs."""
    diffs = np.diff(np.asarray(prices, dtype=float))
    return float(np.maximum(diffs, 0).sum())
