"""Model pipelines, cross-validation, saving and loading."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from student_performance.features import build_preprocessor

SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}


class Trainer:
    """Build, cross-validate, fit, save and load model pipelines."""

    def __init__(self, n_splits: int = 5, seed: int = 42) -> None:
        """Set up the cross-validation folds."""
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.pipeline: Pipeline | None = None

    @staticmethod
    def build_pipeline(model: RegressorMixin) -> Pipeline:
        """Chain the preprocessor and a model."""
        return Pipeline([("prep", build_preprocessor()), ("model", model)])

    def cross_validate(
        self, model: RegressorMixin, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        """Mean MAE and RMSE over the CV folds, plus the MAE spread."""
        scores = cross_validate(self.build_pipeline(model), X, y, cv=self.cv,
                                scoring=SCORING)
        mae = -scores["test_mae"]
        return {
            "mae": mae.mean(),
            "mae_std": mae.std(),
            "rmse": -scores["test_rmse"].mean(),
        }

    def compare(
        self, models: dict[str, RegressorMixin], X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Cross-validate several models under the same folds, best MAE first."""
        rows = {name: self.cross_validate(m, X, y) for name, m in models.items()}
        return pd.DataFrame(rows).T.sort_values("mae")

    def fit(self, model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Fit the full pipeline on the given data and keep it."""
        self.pipeline = self.build_pipeline(model).fit(X, y)
        return self.pipeline

    def save(self, path: str | Path) -> None:
        """Save the fitted pipeline with joblib."""
        if self.pipeline is None:
            raise ValueError("Nothing to save: call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @staticmethod
    def load(path: str | Path) -> Pipeline:
        """Load a saved pipeline."""
        return joblib.load(path)
