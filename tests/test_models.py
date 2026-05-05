"""Tests for the models subpackage."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quantlab.exceptions import InsufficientHistoryError
from quantlab.models.evaluation import (
    classification_metrics,
    per_group_metrics,
    regression_metrics,
)
from quantlab.models.features import build_features
from quantlab.models.forecaster import ReturnForecaster


def _synthetic_close(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"close": close}, index=idx)


class FeaturesTests(unittest.TestCase):
    def test_returns_target_columns(self) -> None:
        feats = build_features(_synthetic_close(300))
        self.assertIn("y_next_ret", feats.columns)
        self.assertIn("y_next_dir", feats.columns)
        for lag in (1, 2, 3, 5, 10, 20):
            self.assertIn(f"ret_lag_{lag}", feats.columns)

    def test_no_lookahead_in_lags(self) -> None:
        df = _synthetic_close(300)
        feats = build_features(df)
        # ret_lag_1 at index t should equal log return between t-2 and t-1.
        expected = np.log(df["close"]).diff().shift(1).reindex(feats.index)
        np.testing.assert_allclose(feats["ret_lag_1"].values, expected.values, rtol=1e-12)

    def test_rejects_short_history(self) -> None:
        with self.assertRaises(InsufficientHistoryError):
            build_features(_synthetic_close(10))

    def test_rejects_missing_close(self) -> None:
        with self.assertRaises(ValueError):
            build_features(pd.DataFrame({"open": [1.0] * 100}))


class ForecasterTests(unittest.TestCase):
    def test_regression_fit_predict(self) -> None:
        feats = build_features(_synthetic_close(400))
        fc = ReturnForecaster(task="regression", n_splits=3)
        res = fc.fit(feats)
        self.assertEqual(res.task, "regression")
        preds = fc.predict(feats)
        self.assertEqual(len(preds), len(feats))

    def test_classification_fit_predict_proba(self) -> None:
        feats = build_features(_synthetic_close(400))
        fc = ReturnForecaster(task="classification", n_splits=3)
        fc.fit(feats)
        proba = fc.predict_proba(feats)
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            ReturnForecaster().predict(pd.DataFrame())

    def test_predict_proba_for_regression_raises(self) -> None:
        feats = build_features(_synthetic_close(200))
        fc = ReturnForecaster(task="regression", n_splits=3)
        fc.fit(feats)
        with self.assertRaises(RuntimeError):
            fc.predict_proba(feats)

    def test_invalid_task_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReturnForecaster(task="ranking")

    def test_invalid_n_splits_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReturnForecaster(n_splits=1)

    def test_missing_target_rejected(self) -> None:
        feats = build_features(_synthetic_close(200)).drop(columns=["y_next_ret", "y_next_dir"])
        with self.assertRaises(ValueError):
            ReturnForecaster(task="regression", n_splits=3).fit(feats)


class EvaluationTests(unittest.TestCase):
    def test_regression_metrics(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        p = pd.Series([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, p)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["mae"], 0.0)

    def test_regression_metrics_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            regression_metrics(pd.Series([1.0, 2.0]), pd.Series([1.0]))

    def test_classification_metrics_perfect_separation(self) -> None:
        y = pd.Series([0, 0, 1, 1])
        p = pd.Series([0.1, 0.2, 0.8, 0.9])
        m = classification_metrics(y, p)
        self.assertAlmostEqual(m["auc"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 1.0)

    def test_classification_metrics_one_class_returns_nan_auc(self) -> None:
        y = pd.Series([1, 1, 1, 1])
        p = pd.Series([0.6, 0.7, 0.8, 0.9])
        m = classification_metrics(y, p)
        self.assertTrue(np.isnan(m["auc"]))

    def test_classification_metrics_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(pd.Series([0]), pd.Series([0.1, 0.2]))

    def test_per_group_metrics(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        p = pd.Series([1.1, 1.9, 3.2, 4.1, 4.8, 6.3])
        groups = ["a", "a", "a", "b", "b", "b"]
        out = per_group_metrics(y, p, groups, task="regression")
        self.assertEqual(set(out["group"]), {"a", "b"})
        self.assertEqual(out.shape[0], 2)

    def test_per_group_metrics_classification(self) -> None:
        y = pd.Series([0, 1, 0, 1, 1, 0])
        p = pd.Series([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        groups = ["a", "a", "a", "b", "b", "b"]
        out = per_group_metrics(y, p, groups, task="classification")
        self.assertEqual(out.shape[0], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
