"""Feature engineering for return / direction forecasting.

Each feature is causal — only uses information available at or before time
``t`` — to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.exceptions import InsufficientHistoryError

DEFAULT_LAGS = (1, 2, 3, 5, 10, 20)
DEFAULT_VOL_WINDOWS = (5, 20)


def build_features(
    prices: pd.DataFrame,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    vol_windows: tuple[int, ...] = DEFAULT_VOL_WINDOWS,
) -> pd.DataFrame:
    """Construct a feature matrix from a single ticker's price history.

    Args:
        prices: DataFrame indexed by date with at least a ``close`` column.
        lags: Return lags (in trading days) to include as predictors.
        vol_windows: Trailing windows over which to compute realised volatility.

    Returns:
        DataFrame indexed by date with the engineered features and a target
        column ``y_next_ret`` (next-day log return). Rows with any missing
        feature are dropped.
    """
    if "close" not in prices.columns:
        raise ValueError("prices must contain a 'close' column")
    if len(prices) < max(max(lags), max(vol_windows)) + 2:
        raise InsufficientHistoryError("not enough history for requested lags / windows")

    df = pd.DataFrame(index=prices.index)
    log_close = np.log(prices["close"])
    daily = log_close.diff()

    for lag in lags:
        df[f"ret_lag_{lag}"] = daily.shift(lag)

    for w in vol_windows:
        df[f"vol_{w}"] = daily.rolling(w).std(ddof=1).shift(1)
        df[f"mean_{w}"] = daily.rolling(w).mean().shift(1)

    df["y_next_ret"] = daily.shift(-1)
    df["y_next_dir"] = (df["y_next_ret"] > 0).astype(int)

    return df.dropna()
