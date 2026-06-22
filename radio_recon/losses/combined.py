"""Configurable image reconstruction losses."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def _laplacian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
    pred_lap = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    target_lap = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return F.l1_loss(pred_lap, target_lap)


def _ssim_surrogate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=(-2, -1), keepdim=True)
    target_mean = target.mean(dim=(-2, -1), keepdim=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    covariance = (pred_centered * target_centered).mean(dim=(-2, -1), keepdim=True)
    pred_var = pred_centered.square().mean(dim=(-2, -1), keepdim=True)
    target_var = target_centered.square().mean(dim=(-2, -1), keepdim=True)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * pred_mean * target_mean + c1) * (2 * covariance + c2))
    ssim = ssim / ((pred_mean.square() + target_mean.square() + c1) * (pred_var + target_var + c2))
    return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of reconstruction losses configured from YAML."""

    def __init__(self, components: List[dict]):
        super().__init__()
        self.components = components or [{"name": "mse", "weight": 1.0}]

    def _component_loss(self, name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if name in {"mse", "l2"}:
            return F.mse_loss(pred, target)
        if name in {"l1", "mae"}:
            return F.l1_loss(pred, target)
        if name in {"fft", "freq_physics"}:
            return _fft_loss(pred, target)
        if name in {"gradient", "grad", "conv_consistency"}:
            return _gradient_loss(pred, target)
        if name in {"laplacian", "laplace"}:
            return _laplacian_loss(pred, target)
        if name == "ssim":
            return _ssim_surrogate_loss(pred, target)
        raise ValueError(f"Unknown loss component: {name}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        total = pred.new_tensor(0.0)
        details: Dict[str, float] = {}
        for component in self.components:
            name = component.get("name", "mse")
            weight = float(component.get("weight", 1.0))
            value = self._component_loss(name, pred, target)
            total = total + weight * value
            details[f"{name}_loss"] = float(value.detach().cpu())
        details["total_loss"] = float(total.detach().cpu())
        return total, details


def create_loss_from_config(config: dict) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "combined")
    if loss_type in {"mse", "l2"}:
        return nn.MSELoss()
    if loss_type in {"l1", "mae"}:
        return nn.L1Loss()
    if loss_type != "combined":
        raise ValueError(f"Unknown loss type: {loss_type}")
    return CombinedLoss(loss_cfg.get("components", [{"name": "mse", "weight": 1.0}]))
