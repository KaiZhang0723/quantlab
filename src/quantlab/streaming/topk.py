"""Heap-based top-K tracker (HW2's K-largest-from-stream, generalised).

Stores the K largest values seen so far in O(K) space; each ``push`` runs in
O(log K). Useful for "biggest movers today" computed in a single pass over a
daily price file.
"""

from __future__ import annotations

import heapq
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class TopK(Generic[T]):
    """Maintain the K largest items observed so far.

    Args:
        k: Number of top items to retain. Must be positive.

    Example:
        >>> tk = TopK(3)
        >>> for x in [4, 1, 7, 3, 9, 2, 6]:
        ...     tk.push(x)
        >>> tk.items()
        [6, 7, 9]
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._k = k
        self._heap: list[T] = []

    def push(self, x: T) -> None:
        """Insert ``x`` into the tracker, evicting the smallest if full."""
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, x)
        elif x > self._heap[0]:
            heapq.heappushpop(self._heap, x)

    def push_many(self, xs: Iterable[T]) -> None:
        for x in xs:
            self.push(x)

    def items(self) -> list[T]:
        """Return the current top-K items in ascending order."""
        return sorted(self._heap)

    def __len__(self) -> int:
        return len(self._heap)
