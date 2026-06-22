"""Image reconstruction metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse(pred, target)
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(target, pred, data_range=data_range))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> Dict[str, float]:
    return {
        "psnr": compute_psnr(pred, target, data_range=data_range),
        "ssim": compute_ssim(pred, target, data_range=data_range),
        "mse": compute_mse(pred, target),
        "mae": compute_mae(pred, target),
    }
