"""Drawdown ("underwater") plot."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from quantlab.compute.rolling import drawdown_series


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Shade the drawdown of an equity curve.

    Args:
        equity: Equity / cumulative-return curve, indexed by date.
        title: Chart title.
        ax: Existing matplotlib axes; if ``None`` a new figure is created.
    """
    dd = drawdown_series(equity)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
