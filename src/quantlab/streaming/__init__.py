"""Online statistics over streaming numerical data.

Each estimator updates in O(1) or O(log K) per element and never stores the
full input stream — the same constraints used in HW2 (top-K bid heap) and
HW3 (running median).
"""

from __future__ import annotations

from quantlab.streaming.median import RunningMedian
from quantlab.streaming.topk import TopK
from quantlab.streaming.welford import OnlineMoments

__all__ = ["RunningMedian", "TopK", "OnlineMoments"]
