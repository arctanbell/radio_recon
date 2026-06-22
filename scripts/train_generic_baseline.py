#!/usr/bin/env python
"""Train generic restoration baselines on shared radio dataset."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from radio_recon.data.dataset import create_dataloaders
from radio_recon.evaluation.metrics import compute_all_metrics
from radio_recon.models.conditional_unet_attention import create_conditional_unet_attention
from radio_recon.models.conditional_unet_film import create_conditional_unet_film
from radio_recon.models.dncnn import create_dncnn
from radio_recon.models.simple_unet import create_simple_unet
from radio_recon.models.swinir_radio import create_swinir_radio
from radio_recon.utils.config import load_config


def pick_input(condition: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "all":
        return condition
    if input_mode == "psf_dirty":
        return condition[:, 0:2]
    if input_mode == "dirty_only":
        return condition[:, 1:2]
    if input_mode == "dirty_sd":
        return condition[:, 1:3]
    raise ValueError(f"Unknown input_mode: {input_mode}")


def build_model(config: dict) -> nn.Module:
    model_type = config["model"]["type"]
    if model_type == "dncnn":
        return create_dncnn(config)
    if model_type in {"simple_unet", "unet"}:
        return create_simple_unet(config)
    if model_type == "swinir":
        return create_swinir_radio(config)
    if model_type == "conditional_unet_film":
        return create_conditional_unet_film(config)
    if model_type == "conditional_unet_attention":
        return create_conditional_unet_attention(config)
    raise ValueError(f"Unsupported model type: {model_type}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, input_mode: str) -> tuple:
    model.eval()
    losses = []
    all_metrics = []
    criterion = nn.MSELoss()

    for condition, target, _ in loader:
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        x = pick_input(condition, input_mode)
        pred = model(x)
        loss = criterion(pred, target)
        losses.append(loss.item())

        pred_np = pred[:, 0].detach().cpu().numpy()
        target_np = target[:, 0].detach().cpu().numpy()
        for i in range(pred_np.shape[0]):
            all_metrics.append(compute_all_metrics(pred_np[i], target_np[i]))

    mean_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return float(np.mean(losses)), mean_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config.get("experiment", {}).get("seed", 42))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mode = config.get("training", {}).get("input_mode", "dirty_only")
    epochs = int(config["training"].get("num_epochs", 100))
    val_interval = int(config["training"].get("val_interval", 1))

    cfg = copy.deepcopy(config)
    train_loader, val_loader, _ = create_dataloaders(cfg, num_workers=config["data"].get("num_workers", 4))

    model = build_model(config).to(device)
    optimizer = Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-4)))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config["training"].get("min_lr", 1e-6)))
    scaler = GradScaler(enabled=torch.cuda.is_available())
    criterion = nn.MSELoss()

    best_psnr = -1e9
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        train_losses = []

        for condition, target, _ in pbar:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            x = pick_input(condition, input_mode)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                pred = model(x)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        scheduler.step()
        train_loss = float(np.mean(train_losses))

        record = {"epoch": epoch, "train_loss": train_loss}
        if epoch % val_interval == 0 or epoch == epochs:
            val_loss, val_metrics = evaluate(model, val_loader, device, input_mode)
            record.update({"val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"psnr={val_metrics['psnr']:.2f} ssim={val_metrics['ssim']:.4f}"
            )

            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": config,
                    },
                    output_dir / "best.pt",
                )

        history.append(record)

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "config": config,
                },
                output_dir / "latest.pt",
            )

    with (output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"Training done. best_psnr={best_psnr:.2f} output={output_dir}")


if __name__ == "__main__":
    main()
