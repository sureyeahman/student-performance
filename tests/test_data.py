"""Tests for DatasetLoader: cleaning and splitting."""

from pathlib import Path

import pandas as pd
import pytest

from student_performance.data import DatasetLoader

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "student-por.csv"


@pytest.fixture
def loader() -> DatasetLoader:
    """A loader with the project settings."""
    return DatasetLoader(DATA_PATH, drop_columns=["G1", "G2"])


@pytest.fixture
def clean_df(loader: DatasetLoader) -> pd.DataFrame:
    """The cleaned dataset."""
    return loader.clean(loader.load())


def test_clean_drops_leakage_columns(clean_df: pd.DataFrame) -> None:
    assert "G1" not in clean_df.columns
    assert "G2" not in clean_df.columns


def test_clean_removes_students_without_exam(clean_df: pd.DataFrame) -> None:
    assert (clean_df["G3"] != 0).all()


def test_split_has_no_test_row_in_train(
    loader: DatasetLoader, clean_df: pd.DataFrame
) -> None:
    X_train, X_test, _, _ = loader.split(clean_df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_keeps_every_row(
    loader: DatasetLoader, clean_df: pd.DataFrame
) -> None:
    X_train, X_test, y_train, y_test = loader.split(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert len(X_train) == len(y_train)
    assert "G3" not in X_train.columns
