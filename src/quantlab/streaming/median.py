"""Two-heap running median (the HW3 algorithm, packaged as a class).

Maintains a max-heap of the lower half and a min-heap of the upper half.
``add(x)`` runs in O(log n); ``median`` is O(1).
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

from quantlab.exceptions import InsufficientHistoryError


class RunningMedian:
    """Online estimator for the median of a numeric stream.

    Example:
        >>> rm = RunningMedian()
        >>> for x in [3, 1, 4, 1, 5, 9, 2, 6]:
        ...     rm.add(x)
        >>> rm.median
        3.5
    """

    def __init__(self) -> None:
        self._lo: list[float] = []
        self._hi: list[float] = []

    def __len__(self) -> int:
        return len(self._lo) + len(self._hi)

    def add(self, x: float) -> None:
        """Insert a value into the running stream."""
        if not self._lo or x <= -self._lo[0]:
            heapq.heappush(self._lo, -x)
        else:
            heapq.heappush(self._hi, x)
        self._rebalance()

    def add_many(self, xs: Iterable[float]) -> None:
        """Insert each value from ``xs``."""
        for x in xs:
            self.add(x)

    def _rebalance(self) -> None:
        if len(self._lo) > len(self._hi) + 1:
            heapq.heappush(self._hi, -heapq.heappop(self._lo))
        elif len(self._hi) > len(self._lo):
            heapq.heappush(self._lo, -heapq.heappop(self._hi))

    @property
    def median(self) -> float:
        """Return the current running median."""
        if not self:
            raise InsufficientHistoryError("median undefined for empty stream")
        if len(self._lo) > len(self._hi):
            return float(-self._lo[0])
        return (-self._lo[0] + self._hi[0]) / 2.0
