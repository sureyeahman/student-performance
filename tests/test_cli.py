"""End-to-end tests of the command-line tool and the config."""

from pathlib import Path

import pytest
import yaml

from student_performance.cli import main
from student_performance.config import load_config

ROOT = Path(__file__).parent.parent


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A copy of config.yaml that saves into a temporary folder, with small grids."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["paths"] = {
        "raw_data": str(ROOT / config["paths"]["raw_data"]),
        "model": str(tmp_path / "model.joblib"),
        "figures": str(tmp_path / "figures"),
    }
    config["grids"] = {
        name: {key: values[:1] for key, values in grid.items()}
        for name, grid in config["grids"].items()
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_config_paths_are_absolute_and_exist() -> None:
    config = load_config(ROOT / "config.yaml")
    assert config["paths"]["raw_data"].is_absolute()
    assert config["paths"]["raw_data"].exists()


def test_train_evaluate_predict(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--config", str(config_path), "train"])
    assert (tmp_path / "model.joblib").exists()

    main(["--config", str(config_path), "evaluate"])
    assert "MAE" in capsys.readouterr().out
    assert (tmp_path / "figures" / "residuals.png").exists()

    main(
        [
            "--config",
            str(config_path),
            "predict",
            "--input",
            str(ROOT / "new_case.json"),
        ]
    )
    assert "Predicted final grade" in capsys.readouterr().out


def test_predict_without_model_gives_clear_error(config_path: Path) -> None:
    with pytest.raises(SystemExit, match="run `student-performance train` first"):
        main(
            [
                "--config",
                str(config_path),
                "predict",
                "--input",
                str(ROOT / "new_case.json"),
            ]
        )
