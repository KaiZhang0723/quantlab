"""scikit-learn pipeline for next-day return / direction forecasting.

Uses :class:`~sklearn.model_selection.TimeSeriesSplit` so the validation
folds never include training data from the future. Mirrors the ``HW5``
pattern (Pipeline + ColumnTransformer + scoring) but adapted to a regression
target (next-day log return) and a classification target (next-day direction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TaskKind = str  # "regression" or "classification"


@dataclass
class ForecastFitResult:
    """Container returned by :meth:`ReturnForecaster.fit`."""

    cv_scores: np.ndarray
    cv_mean: float
    n_features: int
    n_obs: int
    task: TaskKind = "regression"
    params: dict = field(default_factory=dict)


class ReturnForecaster:
    """Train and apply a next-day return / direction forecaster.

    Args:
        task: ``"regression"`` for next-day log return (Ridge), or
            ``"classification"`` for next-day direction (LogisticRegression).
        n_splits: Number of TimeSeriesSplit folds.
        alpha: Regularisation strength for the Ridge regressor.
        C: Inverse regularisation for the logistic classifier.
    """

    def __init__(
        self,
        task: TaskKind = "regression",
        n_splits: int = 5,
        alpha: float = 1.0,
        C: float = 1.0,
    ) -> None:
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.task = task
        self.n_splits = n_splits
        self.alpha = alpha
        self.C = C
        self.pipeline_: Optional[Pipeline] = None
        self.feature_cols_: Optional[list[str]] = None

    def _make_pipeline(self) -> Pipeline:
        if self.task == "regression":
            estimator = Ridge(alpha=self.alpha)
        else:
            estimator = LogisticRegression(C=self.C, max_iter=500, solver="lbfgs")
        return Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])

    @property
    def _target_col(self) -> str:
        return "y_next_ret" if self.task == "regression" else "y_next_dir"

    @property
    def _scoring(self) -> str:
        return "neg_mean_squared_error" if self.task == "regression" else "roc_auc"

    def fit(self, features: pd.DataFrame) -> ForecastFitResult:
        """Cross-validate then fit the final model on the full feature frame."""
        target = self._target_col
        if target not in features.columns:
            raise ValueError(f"features missing target column {target!r}")
        X = features.drop(columns=[c for c in features.columns if c.startswith("y_")])
        y = features[target]
        if len(X) < self.n_splits + 1:
            raise ValueError("not enough rows for the requested n_splits")

        pipe = self._make_pipeline()
        cv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=self._scoring)
        pipe.fit(X, y)

        self.pipeline_ = pipe
        self.feature_cols_ = list(X.columns)
        return ForecastFitResult(
            cv_scores=scores,
            cv_mean=float(scores.mean()),
            n_features=X.shape[1],
            n_obs=len(X),
            task=self.task,
            params={"alpha": self.alpha, "C": self.C, "n_splits": self.n_splits},
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict targets for new feature rows."""
        if self.pipeline_ is None or self.feature_cols_ is None:
            raise RuntimeError("call fit() before predict()")
        X = features[self.feature_cols_]
        return pd.Series(self.pipeline_.predict(X), index=features.index)

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Predict positive-class probability (classification only)."""
        if self.task != "classification":
            raise RuntimeError("predict_proba only available for classification task")
        if self.pipeline_ is None or self.feature_cols_ is None:
            raise RuntimeError("call fit() before predict_proba()")
        X = features[self.feature_cols_]
        proba = self.pipeline_.predict_proba(X)[:, 1]
        return pd.Series(proba, index=features.index)
