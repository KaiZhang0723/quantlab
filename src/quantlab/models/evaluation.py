"""Forecast evaluation metrics.

Includes the standard regression / classification scores plus a walk-forward
helper that mirrors HW5's group-wise evaluation pattern.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """RMSE, MAE, and information coefficient (Spearman corr)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if y_true.nunique() < 2 or y_pred.nunique() < 2:
        ic = float("nan")
    else:
        ic = float(pd.Series(y_true.values).corr(pd.Series(y_pred.values), method="spearman"))
    return {"rmse": rmse, "mae": mae, "ic": ic}


def classification_metrics(y_true: pd.Series, p_pos: pd.Series) -> dict[str, float]:
    """ROC AUC and accuracy at a 0.5 threshold."""
    if len(y_true) != len(p_pos):
        raise ValueError("y_true and p_pos length mismatch")
    auc = float("nan") if y_true.nunique() < 2 else float(roc_auc_score(y_true, p_pos))
    acc = float(accuracy_score(y_true, (p_pos > 0.5).astype(int)))
    return {"auc": auc, "accuracy": acc}


def walk_forward_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    groups: Iterable,
    task: str = "regression",
) -> pd.DataFrame:
    """Compute metrics within each group (e.g. by ticker or year).

    Mirrors HW5's ``evaluate_by_age_group`` pattern.
    """
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values, "g": list(groups)})
    rows = []
    for g, sub in df.groupby("g"):
        if task == "regression":
            rows.append({"group": g, **regression_metrics(sub["y"], sub["p"]), "n": len(sub)})
        else:
            rows.append({"group": g, **classification_metrics(sub["y"], sub["p"]), "n": len(sub)})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
