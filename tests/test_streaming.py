"""Tests for the streaming subpackage.

Streaming algorithms are tested both with deterministic small examples and
with property-based tests against a brute-force numpy / Python reference.
"""

from __future__ import annotations

import statistics
import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.exceptions import InsufficientHistoryError
from quantlab.streaming.median import RunningMedian
from quantlab.streaming.topk import TopK
from quantlab.streaming.welford import OnlineMoments


class WelfordTests(unittest.TestCase):
    def test_matches_numpy_on_small_input(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = OnlineMoments()
        m.update_many(xs)
        self.assertAlmostEqual(m.mean, np.mean(xs))
        self.assertAlmostEqual(m.variance, np.var(xs, ddof=1))
        self.assertAlmostEqual(m.std, np.std(xs, ddof=1))

    def test_empty_raises(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            _ = OnlineMoments().mean

    def test_single_observation_variance_raises(self) -> None:
        m = OnlineMoments()
        m.update(7.0)
        with self.assertRaises(InsufficientHistoryError):
            _ = m.variance

    def test_merge_matches_combined_stream(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=200).tolist()
        b = rng.normal(loc=2.0, size=300).tolist()
        ma, mb = OnlineMoments(), OnlineMoments()
        ma.update_many(a)
        mb.update_many(b)
        merged = OnlineMoments()
        merged.merge(ma).merge(mb)
        ref = a + b
        self.assertAlmostEqual(merged.mean, float(np.mean(ref)), places=10)
        self.assertAlmostEqual(merged.variance, float(np.var(ref, ddof=1)), places=8)

    def test_merge_handles_empty_sides(self) -> None:
        m = OnlineMoments()
        empty = OnlineMoments()
        m.update_many([1.0, 2.0])
        m.merge(empty)
        self.assertEqual(m.n, 2)
        empty.merge(m)
        self.assertEqual(empty.n, 2)

    @settings(max_examples=50, deadline=None)
    @given(st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=2, max_size=200))
    def test_property_matches_numpy(self, xs: list[float]) -> None:
        m = OnlineMoments()
        m.update_many(xs)
        ref_mean = float(np.mean(xs))
        ref_var = float(np.var(xs, ddof=1))
        # Compare with relative tolerance — Welford and numpy each accumulate
        # floating-point error differently, so absolute tolerance is unfair.
        self.assertTrue(np.isclose(m.mean, ref_mean, rtol=1e-9, atol=1e-9))
        self.assertTrue(np.isclose(m.variance, ref_var, rtol=1e-7, atol=1e-9))


class RunningMedianTests(unittest.TestCase):
    def test_small_example(self) -> None:
        rm = RunningMedian()
        rm.add_many([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(rm.median, 3.5)

    def test_odd_length(self) -> None:
        rm = RunningMedian()
        rm.add_many([5, 2, 8])
        self.assertAlmostEqual(rm.median, 5.0)

    def test_empty_raises(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            _ = RunningMedian().median

    @settings(max_examples=50, deadline=None)
    @given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False),
                    min_size=1, max_size=200))
    def test_property_matches_statistics(self, xs: list[float]) -> None:
        rm = RunningMedian()
        for x in xs:
            rm.add(x)
        self.assertAlmostEqual(rm.median, float(statistics.median(xs)), places=6)


class TopKTests(unittest.TestCase):
    def test_keeps_top_k(self) -> None:
        tk = TopK(3)
        tk.push_many([4, 1, 7, 3, 9, 2, 6])
        self.assertEqual(tk.items(), [6, 7, 9])

    def test_handles_fewer_than_k(self) -> None:
        tk = TopK(5)
        tk.push_many([1, 2, 3])
        self.assertEqual(tk.items(), [1, 2, 3])

    def test_rejects_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            TopK(0)

    def test_len(self) -> None:
        tk = TopK(2)
        self.assertEqual(len(tk), 0)
        tk.push(1)
        tk.push(2)
        tk.push(3)
        self.assertEqual(len(tk), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
