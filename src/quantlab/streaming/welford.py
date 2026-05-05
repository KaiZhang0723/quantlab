"""Welford's online algorithm for mean and variance.

Numerically stable single-pass estimator. Ideal for streaming volatility
without peek-ahead — production risk systems use exactly this kind of
estimator on tick data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from quantlab.exceptions import InsufficientHistoryError


@dataclass
class OnlineMoments:
    """Online estimator for the count, mean, variance, and std of a stream.

    Implements Welford's algorithm: each ``update`` runs in O(1) and uses
    O(1) memory.

    Example:
        >>> m = OnlineMoments()
        >>> for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     m.update(x)
        >>> round(m.mean, 6)
        3.0
        >>> round(m.variance, 6)
        2.5
    """

    n: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def update(self, x: float) -> None:
        """Incorporate a new sample into the running statistics."""
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def update_many(self, xs: Iterable[float]) -> None:
        """Convenience for feeding an iterable of values."""
        for x in xs:
            self.update(x)

    @property
    def mean(self) -> float:
        if self.n == 0:
            raise InsufficientHistoryError("mean undefined for empty stream")
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance (n-1 denominator). Requires at least 2 observations."""
        if self.n < 2:
            raise InsufficientHistoryError("variance requires n >= 2")
        return self._m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return self.variance**0.5
