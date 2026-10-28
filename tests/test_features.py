"""Tests for the preprocessing ColumnTransformer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_performance.data import DatasetLoader
from student_performance.features import build_preprocessor

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "student-por.csv"

Splits = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@pytest.fixture
def splits() -> Splits:
    """Train and test sets with the project settings."""
    loader = DatasetLoader(DATA_PATH, drop_columns=["G1", "G2"])
    return loader.split(loader.clean(loader.load()))


def test_output_is_numeric_with_same_rows(splits: Splits) -> None:
    X_train, X_test, _, _ = splits
    preprocessor = build_preprocessor().fit(X_train)
    out = preprocessor.transform(X_test)
    assert out.shape[0] == len(X_test)
    assert not np.isnan(out).any()


def test_scaler_is_fitted_on_train_only(splits: Splits) -> None:
    X_train, _, _, _ = splits
    preprocessor = build_preprocessor().fit(X_train)
    scaler = preprocessor.named_transformers_["num"]
    numeric_columns = X_train.select_dtypes("number").columns
    assert np.allclose(scaler.mean_, X_train[numeric_columns].mean())


def test_unseen_category_does_not_crash(splits: Splits) -> None:
    X_train, X_test, _, _ = splits
    preprocessor = build_preprocessor().fit(X_train)
    new_case = X_test.iloc[[0]].copy()
    new_case["Mjob"] = "astronaut"
    with pytest.warns(UserWarning, match="unknown categories"):
        out = preprocessor.transform(new_case)
    assert out.shape[0] == 1
