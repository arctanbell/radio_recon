"""Input-channel selection helpers."""

from __future__ import annotations

import torch


def select_model_input(condition: torch.Tensor, input_mode: str = "all") -> torch.Tensor:
    """Select condition channels according to a training/evaluation input mode."""
    if input_mode == "all":
        return condition
    if input_mode == "psf_dirty":
        return condition[:, 0:2]
    if input_mode == "dirty_only":
        return condition[:, 1:2]
    if input_mode == "dirty_sd":
        return condition[:, 1:3]
    raise ValueError(f"Unknown input_mode: {input_mode}")


def input_mode_from_config(config: dict) -> str:
    """Read the model input mode from a config dict."""
    return config.get("training", {}).get("input_mode", "all")
