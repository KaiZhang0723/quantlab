"""Correlation-matrix heatmap (seaborn)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Return correlations",
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot the pairwise return correlation matrix as a heatmap.

    Args:
        returns: Wide-format DataFrame with one return column per ticker.
        title: Chart title.
        ax: Existing matplotlib axes; if ``None`` a new figure is created.
    """
    if returns.shape[1] < 2:
        raise ValueError("need at least 2 columns to compute correlations")
    corr = returns.corr()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f",
                cbar=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
