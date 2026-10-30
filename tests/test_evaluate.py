"""Tests for metrics and the at-risk decision rule."""

import numpy as np
import pandas as pd
import pytest

from student_performance.evaluate import (
    choose_threshold,
    flag_at_risk,
    flag_metrics,
    regression_metrics,
)


def test_regression_metrics_known_values() -> None:
    metrics = regression_metrics(pd.Series([10, 12]), np.array([11, 14]))
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))


def test_flag_is_strictly_below_threshold() -> None:
    flags = flag_at_risk(np.array([9.9, 10.0, 11.2]), threshold=10.0)
    assert flags.tolist() == [True, False, False]


def test_flag_metrics_counts() -> None:
    y_true = pd.Series([8, 9, 12, 15])
    y_pred = np.array([9, 12, 10, 14])
    metrics = flag_metrics(y_true, y_pred, threshold=11)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)


def test_choose_threshold_reaches_target_recall() -> None:
    y_true = pd.Series([8, 9, 12, 15])
    y_pred = np.array([9, 12, 10, 14])
    threshold = choose_threshold(y_true, y_pred, target_recall=1.0)
    assert flag_metrics(y_true, y_pred, threshold)["recall"] == 1.0
    assert threshold == pytest.approx(12.1)
