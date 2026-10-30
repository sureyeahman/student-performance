"""Metrics, the at-risk decision rule, interpretation and plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAE and RMSE, in grade points."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def flag_at_risk(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    """True for every student whose predicted grade is below the threshold."""
    return np.asarray(y_pred) < threshold


def flag_metrics(
    y_true: pd.Series, y_pred: np.ndarray, threshold: float, pass_mark: float = 10
) -> dict[str, float]:
    """How well the flag finds the students who really fail."""
    fails = np.asarray(y_true) < pass_mark
    flagged = flag_at_risk(y_pred, threshold)
    caught = (fails & flagged).sum()
    return {
        "recall": float(caught / fails.sum()) if fails.sum() else 0.0,
        "precision": float(caught / flagged.sum()) if flagged.sum() else 0.0,
        "flagged_share": float(flagged.mean()),
    }


def choose_threshold(
    y_true: pd.Series, y_pred: np.ndarray, target_recall: float, pass_mark: float = 10
) -> float:
    """Lowest threshold that flags at least target_recall of failing students."""
    for threshold in np.arange(pass_mark, 20.01, 0.1):
        if (
            flag_metrics(y_true, y_pred, threshold, pass_mark)["recall"]
            >= target_recall
        ):
            return round(float(threshold), 1)
    return 20.0


def flag_table(
    y_true: pd.Series, y_pred: np.ndarray, threshold: float, pass_mark: float = 10
) -> pd.DataFrame:
    """Confusion matrix: real outcome against the tool's flag."""
    actual = np.where(np.asarray(y_true) < pass_mark, "fails", "passes")
    flag = np.where(flag_at_risk(y_pred, threshold), "flagged", "not flagged")
    return pd.crosstab(pd.Series(actual, name="actual"), pd.Series(flag, name="tool"))


def top_coefficients(pipeline: Pipeline, n: int = 10) -> pd.DataFrame:
    """The n features with the largest coefficients (linear models only)."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(pipeline.named_steps["model"].coef_, index=names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(n)
    return top.rename("coefficient").to_frame()


def plot_residuals(
    y_true: pd.Series, y_pred: np.ndarray, path: str | Path | None = None
) -> Figure:
    """Residuals (real minus predicted) against the predicted grade."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted final grade (G3)")
    ax.set_ylabel("Real minus predicted")
    ax.set_title("Residuals on the test set")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
