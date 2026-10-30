"""Model pipelines, cross-validation, tuning, saving and loading."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR

from student_performance.features import build_preprocessor

SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

Candidates = dict[str, tuple[RegressorMixin, dict[str, list[Any]]]]


def candidate_models(
    grids: dict[str, dict[str, list[Any]]], seed: int = 42
) -> Candidates:
    """The models to compare, each with its hyperparameter grid."""
    models = {
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=10_000),
        "knn": KNeighborsRegressor(),
        "linear_svr": LinearSVR(max_iter=20_000, random_state=seed),
        "huber": HuberRegressor(max_iter=2_000),
    }
    return {
        name: (model, {f"model__{key}": values for key, values in grids[name].items()})
        for name, model in models.items()
    }


class Trainer:
    """Build, cross-validate, tune, fit, save and load model pipelines."""

    def __init__(self, n_splits: int = 5, seed: int = 42) -> None:
        """Set up the cross-validation folds."""
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.pipeline: Pipeline | None = None
        self.model_name: str | None = None
        self.threshold: float | None = None

    @staticmethod
    def build_pipeline(model: RegressorMixin) -> Pipeline:
        """Chain the preprocessor and a model."""
        return Pipeline([("prep", build_preprocessor()), ("model", model)])

    def cross_validate(
        self, model: RegressorMixin, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        """Mean MAE and RMSE over the CV folds, plus the MAE spread."""
        scores = cross_validate(
            self.build_pipeline(model), X, y, cv=self.cv, scoring=SCORING
        )
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

    def tune(
        self,
        model: RegressorMixin,
        grid: dict[str, list[Any]],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> GridSearchCV:
        """Grid-search one model's hyperparameters on MAE, then refit the best."""
        search = GridSearchCV(
            self.build_pipeline(model), grid, cv=self.cv, scoring=SCORING, refit="mae"
        )
        return search.fit(X, y)

    def select_best(
        self, candidates: Candidates, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Tune every candidate under the same folds and keep the best one by MAE."""
        rows, searches = {}, {}
        for name, (model, grid) in candidates.items():
            search = self.tune(model, grid, X, y)
            i = search.best_index_
            results = search.cv_results_
            rows[name] = {
                "mae": -results["mean_test_mae"][i],
                "mae_std": results["std_test_mae"][i],
                "rmse": -results["mean_test_rmse"][i],
                "best_params": {
                    k.removeprefix("model__"): v for k, v in search.best_params_.items()
                },
            }
            searches[name] = search
        table = (
            pd.DataFrame(rows)
            .T.astype({"mae": float, "mae_std": float, "rmse": float})
            .sort_values("mae")
        )
        self.model_name = table.index[0]
        self.pipeline = searches[self.model_name].best_estimator_
        return table

    def out_of_fold_predictions(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Predict each training row with a model that never saw it."""
        if self.pipeline is None:
            raise ValueError("No model yet: call fit() or select_best() first.")
        return cross_val_predict(clone(self.pipeline), X, y, cv=self.cv)

    def fit(self, model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Fit the full pipeline on the given data and keep it."""
        self.pipeline = self.build_pipeline(model).fit(X, y)
        return self.pipeline

    def save(self, path: str | Path) -> None:
        """Save the fitted pipeline, its name and its decision threshold."""
        if self.pipeline is None:
            raise ValueError("Nothing to save: call fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "model_name": self.model_name,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Trainer":
        """Load a saved Trainer."""
        saved = joblib.load(path)
        trainer = cls()
        trainer.pipeline = saved["pipeline"]
        trainer.model_name = saved["model_name"]
        trainer.threshold = saved["threshold"]
        return trainer
