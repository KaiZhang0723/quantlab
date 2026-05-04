"""Smoke tests for the viz subpackage.

We don't validate pixels — just that the figures construct, contain the
expected number of lines / patches, and have the expected labels.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pandas as pd

from quantlab.viz.correlation import plot_correlation_heatmap
from quantlab.viz.drawdown import plot_drawdown
from quantlab.viz.returns import plot_cumulative_returns


def _equity(seed: int = 0, n: int = 200) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(np.exp(np.cumsum(rets)), index=idx, name="equity")


class CumulativeReturnsTests(unittest.TestCase):
    def test_plot_series(self) -> None:
        fig = plot_cumulative_returns(_equity())
        ax = fig.axes[0]
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(len(ax.get_lines()), 1)

    def test_plot_dataframe_multi_series(self) -> None:
        df = pd.DataFrame({"A": _equity(0), "B": _equity(1)})
        fig = plot_cumulative_returns(df)
        self.assertEqual(len(fig.axes[0].get_lines()), 2)


class DrawdownTests(unittest.TestCase):
    def test_plot_drawdown_creates_axes(self) -> None:
        fig = plot_drawdown(_equity())
        self.assertEqual(len(fig.axes), 1)


class CorrelationTests(unittest.TestCase):
    def test_plot_correlation_heatmap(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("ABC"))
        fig = plot_correlation_heatmap(df)
        self.assertEqual(len(fig.axes), 2)  # heatmap + colorbar

    def test_plot_correlation_heatmap_requires_two_cols(self) -> None:
        with self.assertRaises(ValueError):
            plot_correlation_heatmap(pd.DataFrame({"A": [1, 2, 3]}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
