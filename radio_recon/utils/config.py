"""YAML configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config and apply simple public-release overrides."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if config is None:
        raise ValueError(f"Empty config file: {path}")

    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        config.setdefault("data", {})["data_dir"] = data_dir

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        config.setdefault("training", {})["output_dir"] = output_dir

    return config
