"""Loading, cleaning and splitting the data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Load, clean and split the Student Performance dataset."""

    def __init__(
        self,
        path: str | Path,
        target: str = "G3",
        drop_columns: list[str] | None = None,
        test_size: float = 0.2,
        seed: int = 42,
        pass_mark: int = 10,
        separator: str = ";",
    ) -> None:
        """Store the settings used by the other methods."""
        self.path = Path(path)
        self.target = target
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.test_size = test_size
        self.seed = seed
        self.pass_mark = pass_mark
        self.separator = separator

    def load(self) -> pd.DataFrame:
        """Read the raw CSV file into a DataFrame."""
        return pd.read_csv(self.path, sep=self.separator)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop students who did not sit the exam and the leakage columns."""
        sat_exam = df[self.target] != 0
        cleaned = df[sat_exam].drop(columns=self.drop_columns)
        return cleaned.reset_index(drop=True)

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split once into train and test sets, stratified on pass/fail."""
        y = df[self.target]
        X = df.drop(columns=[self.target])
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=(y >= self.pass_mark),
        )
        return X_train, X_test, y_train, y_test
