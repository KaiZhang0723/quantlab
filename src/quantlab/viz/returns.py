"""Cumulative returns plot."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_cumulative_returns(
    equity: pd.Series | pd.DataFrame,
    title: str = "Cumulative returns",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Plot one or more equity curves on a log-y axis.

    Args:
        equity: Either a single equity-curve Series or a DataFrame whose
            columns are individual equity curves (one per ticker / strategy).
        title: Chart title.
        ax: Existing matplotlib axes; if ``None`` a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    if isinstance(equity, pd.Series):
        ax.plot(equity.index, equity.values, label=equity.name or "equity")
    else:
        for col in equity.columns:
            ax.plot(equity.index, equity[col].values, label=str(col))
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (log scale)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig
