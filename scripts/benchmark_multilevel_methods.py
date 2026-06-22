#!/usr/bin/env python
"""Run multi-level baseline comparison on a shared test split.

This script evaluates three baseline groups under a unified protocol:
1) Classical radio reconstruction: CLEAN and Multi-scale CLEAN.
2) Generic image restoration: DnCNN / U-Net / SwinIR standard configs.
3) Radio deep learning methods: in-repo models or external prediction folders.
"""

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from radio_recon.data.dataset import create_dataloaders
from radio_recon.evaluation.metrics import compute_all_metrics
from radio_recon.models.conditional_unet_attention import create_conditional_unet_attention
from radio_recon.models.conditional_unet_film import create_conditional_unet_film
from radio_recon.models.dncnn import create_dncnn
from radio_recon.models.dit_radio import create_dit_radio
from radio_recon.models.simple_unet import create_simple_unet
from radio_recon.models.swinir_radio import create_swinir_radio
from radio_recon.utils.config import load_config


def shifted_stamp(stamp: np.ndarray, center_y: int, center_x: int, out_shape: Tuple[int, int]) -> np.ndarray:
    """Stamp a kernel centered at (center_y, center_x) into an output array."""
    h, w = out_shape
    kh, kw = stamp.shape
    cy = kh // 2
    cx = kw // 2
    y0 = center_y - cy
    x0 = center_x - cx
    y1 = y0 + kh
    x1 = x0 + kw

    oy0 = max(0, y0)
    ox0 = max(0, x0)
    oy1 = min(h, y1)
    ox1 = min(w, x1)

    sy0 = oy0 - y0
    sx0 = ox0 - x0
    sy1 = sy0 + (oy1 - oy0)
    sx1 = sx0 + (ox1 - ox0)

    out = np.zeros(out_shape, dtype=np.float32)
    out[oy0:oy1, ox0:ox1] = stamp[sy0:sy1, sx0:sx1]
    return out


def make_clean_beam(psf: np.ndarray, sigma_px: float) -> np.ndarray:
    """Create a normalized Gaussian clean beam."""
    h, w = psf.shape
    y = np.arange(h, dtype=np.float32) - h // 2
    x = np.arange(w, dtype=np.float32) - w // 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    beam = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_px * sigma_px + 1e-8)).astype(np.float32)
    beam /= max(np.sum(beam), 1e-8)
    return beam


def clean_hogbom(dirty: np.ndarray, psf: np.ndarray, niter: int, gain: float, threshold: float, clean_beam_sigma: float) -> np.ndarray:
    """Basic Hogbom CLEAN implementation in image domain."""
    psf = psf.astype(np.float32)
    dirty = dirty.astype(np.float32)
    peak = np.max(np.abs(psf)) + 1e-8
    psf = psf / peak

    model = np.zeros_like(dirty, dtype=np.float32)
    residual = dirty.copy()

    for _ in range(niter):
        idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        amp = gain * residual[idx]
        if abs(amp) < threshold:
            break
        model[idx] += amp
        shifted_psf = shifted_stamp(psf, idx[0], idx[1], residual.shape)
        residual -= amp * shifted_psf

    clean_beam = make_clean_beam(psf, clean_beam_sigma)
    restored_model = fftconvolve(model, clean_beam, mode="same").astype(np.float32)
    return restored_model + residual


def clean_multiscale(
    dirty: np.ndarray,
    psf: np.ndarray,
    niter: int,
    gain: float,
    threshold: float,
    scales: List[float],
    clean_beam_sigma: float,
) -> np.ndarray:
    """Simple multi-scale CLEAN with scale-selective components."""
    psf = psf.astype(np.float32)
    dirty = dirty.astype(np.float32)
    psf /= np.max(np.abs(psf)) + 1e-8

    h, w = dirty.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    kernels = []
    for sigma in scales:
        if sigma <= 0:
            k = np.zeros((h, w), dtype=np.float32)
            k[h // 2, w // 2] = 1.0
        else:
            k = np.exp(-(((yy - h // 2) ** 2 + (xx - w // 2) ** 2) / (2.0 * sigma * sigma + 1e-8))).astype(np.float32)
            k /= max(np.sum(k), 1e-8)
        kernels.append(k)

    model = np.zeros_like(dirty, dtype=np.float32)
    residual = dirty.copy()

    for _ in range(niter):
        best_amp = 0.0
        best_idx = None
        best_kernel = None

        for k in kernels:
            score = fftconvolve(residual, k[::-1, ::-1], mode="same")
            idx = np.unravel_index(np.argmax(np.abs(score)), score.shape)
            amp = gain * score[idx]
            if abs(amp) > abs(best_amp):
                best_amp = amp
                best_idx = idx
                best_kernel = k

        if best_idx is None or abs(best_amp) < threshold:
            break

        component = shifted_stamp(best_kernel, best_idx[0], best_idx[1], residual.shape)
        model += best_amp * component
        response = fftconvolve(component, psf, mode="same").astype(np.float32)
        residual -= best_amp * response

    clean_beam = make_clean_beam(psf, clean_beam_sigma)
    restored_model = fftconvolve(model, clean_beam, mode="same").astype(np.float32)
    return restored_model + residual


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


def build_model(method_cfg: dict, base_config: dict, device: torch.device) -> nn.Module:
    arch = method_cfg["arch"]
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("model", {})
    cfg["model"].update(method_cfg.get("model_overrides", {}))

    if arch == "dncnn":
        model = create_dncnn(cfg)
    elif arch in {"simple_unet", "unet"}:
        model = create_simple_unet(cfg)
    elif arch == "swinir":
        model = create_swinir_radio(cfg)
    elif arch == "conditional_unet_film":
        model = create_conditional_unet_film(cfg)
    elif arch == "conditional_unet_attention":
        model = create_conditional_unet_attention(cfg)
    elif arch == "dit":
        model = create_dit_radio(cfg)
    else:
        raise ValueError(f"Unknown model arch: {arch}")

    checkpoint = method_cfg.get("checkpoint")
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=method_cfg.get("strict_load", True))
    elif method_cfg.get("warn_random_init", True):
        print(f"[WARN] {method_cfg.get('name', arch)} has no checkpoint; evaluating random initialized weights.")

    model = model.to(device)
    model.eval()
    return model


def postprocess(arr: np.ndarray, post_cfg: dict) -> np.ndarray:
    arr = arr.astype(np.float32)
    if post_cfg.get("clip", True):
        arr = np.clip(arr, post_cfg.get("clip_min", 0.0), post_cfg.get("clip_max", 1.0))
    return arr


def evaluate_method(
    method_cfg: dict,
    category: str,
    test_dataset,
    batch_size: int,
    num_workers: int,
    base_config: dict,
    post_cfg: dict,
    device: torch.device,
    max_samples: int,
) -> dict:
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    method_type = method_cfg["type"]
    name = method_cfg["name"]
    metrics_all = []

    model = None
    input_mode = method_cfg.get("input_mode", "all")
    if method_type == "model":
        model = build_model(method_cfg, base_config, device)

    external_dir = method_cfg.get("prediction_dir")
    seen = 0

    for condition, target, names in tqdm(loader, desc=f"[{category}] {name}"):
        if max_samples > 0 and seen >= max_samples:
            break

        cond_eval = condition
        if method_cfg.get("zero_sd", False) and condition.shape[1] >= 3:
            cond_eval = condition.clone()
            cond_eval[:, 2:3] = 0.0

        if method_type == "model":
            assert model is not None
            x = pick_input(cond_eval.to(device), input_mode)
            with torch.no_grad():
                pred = model(x).cpu().numpy()
        else:
            pred = None

        for i in range(condition.shape[0]):
            if max_samples > 0 and seen >= max_samples:
                break

            target_np = postprocess(target[i, 0].numpy(), post_cfg)

            if method_type == "classical_clean":
                pred_np = clean_hogbom(
                    dirty=cond_eval[i, 1].numpy(),
                    psf=cond_eval[i, 0].numpy(),
                    niter=method_cfg.get("niter", 200),
                    gain=method_cfg.get("gain", 0.1),
                    threshold=method_cfg.get("threshold", 1e-3),
                    clean_beam_sigma=method_cfg.get("clean_beam_sigma", 1.5),
                )
            elif method_type == "classical_multiscale_clean":
                pred_np = clean_multiscale(
                    dirty=cond_eval[i, 1].numpy(),
                    psf=cond_eval[i, 0].numpy(),
                    niter=method_cfg.get("niter", 200),
                    gain=method_cfg.get("gain", 0.1),
                    threshold=method_cfg.get("threshold", 1e-3),
                    scales=method_cfg.get("scales", [0.0, 2.0, 5.0]),
                    clean_beam_sigma=method_cfg.get("clean_beam_sigma", 1.5),
                )
            elif method_type == "external_predictions":
                sample_name = names[i]
                if external_dir is None:
                    raise ValueError(f"prediction_dir is required for method: {name}")
                file_npy = os.path.join(external_dir, f"{sample_name}.npy")
                if not os.path.exists(file_npy):
                    raise FileNotFoundError(f"Missing external prediction: {file_npy}")
                pred_np = np.load(file_npy)
            elif method_type == "model":
                assert pred is not None
                pred_np = pred[i, 0]
            else:
                raise ValueError(f"Unknown method type: {method_type}")

            pred_np = postprocess(pred_np, post_cfg)
            metrics_all.append(compute_all_metrics(pred_np, target_np))
            seen += 1

    if not metrics_all:
        raise RuntimeError(f"No samples evaluated for method: {name}")

    mean_metrics = {k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0].keys()}
    std_metrics = {k: float(np.std([m[k] for m in metrics_all])) for k in metrics_all[0].keys()}

    return {
        "category": category,
        "name": name,
        "type": method_type,
        "num_samples": len(metrics_all),
        "metrics": {"mean": mean_metrics, "std": std_metrics},
    }


def format_table(results: List[dict]) -> str:
    lines = []
    lines.append(f"{'Category':<22} {'Method':<28} {'PSNR':>8} {'SSIM':>8} {'MSE':>12} {'MAE':>12}")
    lines.append("-" * 96)
    for r in sorted(results, key=lambda x: x["metrics"]["mean"]["psnr"], reverse=True):
        m = r["metrics"]["mean"]
        lines.append(
            f"{r['category']:<22} {r['name']:<28} {m['psnr']:>8.2f} {m['ssim']:>8.4f} {m['mse']:>12.6f} {m['mae']:>12.6f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-level baseline benchmark")
    parser.add_argument("--benchmark_config", required=True, help="Path to benchmark YAML")
    args = parser.parse_args()

    bench_cfg = load_config(args.benchmark_config)
    base_cfg = load_config(bench_cfg["base_config"])

    batch_size = bench_cfg.get("batch_size", base_cfg["training"].get("batch_size", 4))
    num_workers = bench_cfg.get("num_workers", base_cfg["data"].get("num_workers", 4))
    max_samples = bench_cfg.get("max_samples", 0)
    post_cfg = bench_cfg.get("postprocess", {"clip": True, "clip_min": 0.0, "clip_max": 1.0})

    out_dir = Path(bench_cfg.get("output_dir", "evaluation_multilevel_benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(bench_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Base config: {bench_cfg['base_config']}")

    data_cfg = copy.deepcopy(base_cfg)
    data_cfg["training"]["batch_size"] = batch_size
    _, _, test_loader = create_dataloaders(data_cfg, num_workers=num_workers)
    test_dataset = test_loader.dataset
    print(f"Shared test set size: {len(test_dataset)}")

    results = []
    groups = bench_cfg.get("methods", {})
    for category, methods in groups.items():
        for method_cfg in methods:
            result = evaluate_method(
                method_cfg=method_cfg,
                category=category,
                test_dataset=test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                base_config=base_cfg,
                post_cfg=post_cfg,
                device=device,
                max_samples=max_samples,
            )
            results.append(result)
            mm = result["metrics"]["mean"]
            print(f"{category}/{result['name']}: PSNR={mm['psnr']:.2f} SSIM={mm['ssim']:.4f}")

    table = format_table(results)
    print("\n" + table)

    with (out_dir / "benchmark_results.json").open("w") as f:
        json.dump({"results": results, "config": bench_cfg}, f, indent=2)

    with (out_dir / "benchmark_table.txt").open("w") as f:
        f.write(table + "\n")

    with (out_dir / "benchmark_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "method", "num_samples", "psnr", "ssim", "mse", "mae"])
        for r in results:
            mm = r["metrics"]["mean"]
            writer.writerow([r["category"], r["name"], r["num_samples"], mm["psnr"], mm["ssim"], mm["mse"], mm["mae"]])

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
