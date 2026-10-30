"""Command-line tool: train, evaluate and predict."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from student_performance.config import load_config
from student_performance.data import DatasetLoader
from student_performance.evaluate import (
    choose_threshold,
    flag_at_risk,
    flag_metrics,
    flag_table,
    plot_residuals,
    regression_metrics,
)
from student_performance.models import Trainer, candidate_models


def load_trainer(config: dict[str, Any]) -> Trainer:
    """Load the saved model, or stop with a clear message."""
    path = config["paths"]["model"]
    if not Path(path).exists():
        raise SystemExit(f"No model at {path}: run `student-performance train` first.")
    return Trainer.load(path)


def train(config: dict[str, Any]) -> None:
    """Tune every candidate on the training set, keep the best and save it."""
    X_train, _, y_train, _ = DatasetLoader.from_config(config).load_and_split()
    trainer = Trainer(n_splits=config["cv"]["n_splits"], seed=config["seed"])
    candidates = candidate_models(config["grids"], seed=config["seed"])
    table = trainer.select_best(candidates, X_train, y_train)

    out_of_fold = trainer.out_of_fold_predictions(X_train, y_train)
    trainer.threshold = choose_threshold(
        y_train,
        out_of_fold,
        config["decision"]["target_recall"],
        config["data"]["pass_mark"],
    )
    trainer.save(config["paths"]["model"])

    print("Cross-validation on the training set (MAE in grade points):")
    print(table.round(3).to_string())
    print(f"\nChosen model: {trainer.model_name}")
    print(f"Flag threshold: predicted G3 < {trainer.threshold}")
    print(f"Saved to {config['paths']['model']}")


def evaluate(config: dict[str, Any]) -> None:
    """Score the saved model once on the test set."""
    _, X_test, _, y_test = DatasetLoader.from_config(config).load_and_split()
    trainer = load_trainer(config)
    y_pred = trainer.pipeline.predict(X_test)
    pass_mark = config["data"]["pass_mark"]

    metrics = regression_metrics(y_test, y_pred)
    flags = flag_metrics(y_test, y_pred, trainer.threshold, pass_mark)
    figure_path = config["paths"]["figures"] / "residuals.png"
    plt.close(plot_residuals(y_test, y_pred, figure_path))

    print(f"Model: {trainer.model_name}  ({len(X_test)} test students)")
    print(f"MAE:  {metrics['mae']:.2f} grade points")
    print(f"RMSE: {metrics['rmse']:.2f} grade points")
    print(f"\nFlag (predicted G3 < {trainer.threshold}):")
    print(f"  catches {flags['recall']:.0%} of failing students")
    print(f"  {flags['precision']:.0%} of flagged students really fail")
    print(f"  flags {flags['flagged_share']:.0%} of all students")
    print(flag_table(y_test, y_pred, trainer.threshold, pass_mark).to_string())
    print(f"\nResidual plot saved to {figure_path}")


def predict(config: dict[str, Any], input_path: str) -> None:
    """Predict the final grade of the student(s) in a JSON file."""
    data = json.loads(Path(input_path).read_text())
    cases = pd.DataFrame(data if isinstance(data, list) else [data])
    trainer = load_trainer(config)
    grades = trainer.pipeline.predict(cases)
    for grade, at_risk in zip(grades, flag_at_risk(grades, trainer.threshold)):
        status = "AT RISK" if at_risk else "not at risk"
        print(
            f"Predicted final grade: {grade:.1f} / 20 -> {status} "
            f"(threshold {trainer.threshold})"
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `student-performance` command."""
    parser = argparse.ArgumentParser(
        prog="student-performance",
        description="Predict a student's final grade and flag students at risk.",
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="tune the models and save the best one")
    commands.add_parser("evaluate", help="print the test-set metrics")
    predict_parser = commands.add_parser("predict", help="predict one new student")
    predict_parser.add_argument("--input", required=True, help="JSON file of a student")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.command == "train":
        train(config)
    elif args.command == "evaluate":
        evaluate(config)
    else:
        predict(config, args.input)


if __name__ == "__main__":
    main()
