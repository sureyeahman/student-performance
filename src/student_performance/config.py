"""Read the project settings from config.yaml."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML config; its paths are made absolute from the file's folder."""
    path = Path(path).resolve()
    with path.open() as file:
        config = yaml.safe_load(file)
    config["paths"] = {
        key: path.parent / value for key, value in config["paths"].items()
    }
    return config
