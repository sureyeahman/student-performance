"""Tests for the Trainer: pipeline, cross-validation, predictions, save/load."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from student_performance.data import DatasetLoader
from student_performance.models import Trainer

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "student-por.csv"

Splits = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@pytest.fixture
def splits() -> Splits:
    """Train and test sets with the project settings."""
    loader = DatasetLoader(DATA_PATH, drop_columns=["G1", "G2"])
    return loader.split(loader.clean(loader.load()))


def test_pipeline_starts_with_preprocessor() -> None:
    pipeline = Trainer.build_pipeline(LinearRegression())
    assert list(pipeline.named_steps) == ["prep", "model"]


def test_linear_model_beats_baseline(splits: Splits) -> None:
    X_train, _, y_train, _ = splits
    results = Trainer().compare(
        {"dummy": DummyRegressor(), "linear": LinearRegression()}, X_train, y_train
    )
    assert results.loc["linear", "mae"] < results.loc["dummy", "mae"]


def test_predictions_have_right_shape_and_range(splits: Splits) -> None:
    X_train, X_test, y_train, _ = splits
    predictions = Trainer().fit(LinearRegression(), X_train, y_train).predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
    assert predictions.min() > -5 and predictions.max() < 25


def test_save_and_load_give_same_predictions(splits: Splits, tmp_path: Path) -> None:
    X_train, X_test, y_train, _ = splits
    trainer = Trainer()
    trainer.fit(LinearRegression(), X_train, y_train)
    trainer.save(tmp_path / "model.joblib")
    loaded = Trainer.load(tmp_path / "model.joblib")
    assert np.allclose(loaded.predict(X_test), trainer.pipeline.predict(X_test))
