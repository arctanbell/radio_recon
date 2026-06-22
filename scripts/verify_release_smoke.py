#!/usr/bin/env python
"""Smoke-check the paper-release repository on a full dependency environment."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from radio_recon.data.dataset import RadioReconstructionDataset
from radio_recon.losses.combined import create_loss_from_config
from radio_recon.models.model_factory import create_model_from_config
from radio_recon.utils.config import load_config
from radio_recon.utils.input_mode import input_mode_from_config, select_model_input


CONDITIONAL_FULL_INPUT_MODELS = {"conditional_unet_film", "conditional_unet_attention"}


def synthetic_input_shape(config: dict) -> tuple[int, int]:
    """Return channel count and image size for direct model-forward smoke tests."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "simple_unet")

    channels = int(model_cfg.get("in_channels", 3))
    if model_type in CONDITIONAL_FULL_INPUT_MODELS:
        channels = 3

    if model_type == "dit":
        size = int(model_cfg.get("image_size", 192))
    else:
        size = int(config.get("smoke", {}).get("image_size", 32))
    return channels, size


def check_models(config_paths: list[str], device: torch.device) -> None:
    print(f"device: {device}")
    for config_path in config_paths:
        cfg = load_config(config_path)
        cfg.setdefault("loss", {"type": "mse"})
        model = create_model_from_config(cfg).to(device)
        model.eval()
        in_channels, image_size = synthetic_input_shape(cfg)
        x = torch.rand(1, in_channels, image_size, image_size, device=device)
        y = torch.rand(1, 1, image_size, image_size, device=device)
        with torch.no_grad():
            out = model(x)
            loss_result = create_loss_from_config(cfg)(out, y)
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
        print(
            f"model ok: {config_path} -> "
            f"input={tuple(x.shape)} output={tuple(out.shape)} loss={float(loss.detach().cpu()):.6f}"
        )
        del model, x, y, out, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()


def check_dataset(config_path: str, data_dir: str) -> None:
    cfg = load_config(config_path)
    cfg["data"]["data_dir"] = data_dir
    dataset = RadioReconstructionDataset(data_dir=data_dir, config=cfg, split="test", augment=False)
    condition, target, name = dataset[0]
    print(
        "dataset ok: "
        f"samples={len(dataset)} first={name} condition={tuple(condition.shape)} target={tuple(target.shape)}"
    )


def check_real_forward(config_path: str, data_dir: str, device: torch.device) -> None:
    cfg = load_config(config_path)
    cfg["data"]["data_dir"] = data_dir
    dataset = RadioReconstructionDataset(data_dir=data_dir, config=cfg, split="test", augment=False)
    condition, target, name = dataset[0]
    model = create_model_from_config(cfg).to(device)
    model.eval()
    x = condition.unsqueeze(0).to(device)
    y = target.unsqueeze(0).to(device)
    input_mode = input_mode_from_config(cfg)
    with torch.no_grad():
        out = model(select_model_input(x, input_mode))
        loss_result = create_loss_from_config(cfg)(out, y)
        loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
    print(
        "real forward ok: "
        f"config={config_path} sample={name} input_mode={input_mode} "
        f"condition={tuple(x.shape)} output={tuple(out.shape)} loss={float(loss.detach().cpu()):.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_glob", default="configs/paper/*.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--dataset_config", default="configs/paper/main_unet.yaml")
    parser.add_argument("--real_forward_config", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = sorted(glob.glob(args.config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched: {args.config_glob}")

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda devices: {torch.cuda.device_count()}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    check_models(config_paths, device)
    if args.data_dir:
        check_dataset(args.dataset_config, args.data_dir)
    if args.data_dir and args.real_forward_config:
        check_real_forward(args.real_forward_config, args.data_dir, device)


if __name__ == "__main__":
    main()
