#!/usr/bin/env python

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from radio_recon.data.dataset import RadioReconstructionDataset
from radio_recon.data.normalization import NormalizationFactory
from radio_recon.data.utils import load_fits
from radio_recon.utils.config import load_config


def build_test_sample_paths(config_path: str):
    cfg = load_config(config_path)
    cfg_nofilter = deepcopy(cfg)
    cfg_nofilter.setdefault("data", {})
    cfg_nofilter["data"].setdefault("filter", {})
    cfg_nofilter["data"]["filter"] = dict(cfg_nofilter["data"]["filter"])
    cfg_nofilter["data"]["filter"]["enabled"] = False

    ds = RadioReconstructionDataset(
        data_dir=cfg_nofilter["data"]["data_dir"],
        config=cfg_nofilter,
        split="train",
        augment=False,
    )

    total_size = len(ds)
    test_ratio = cfg_nofilter["data"].get("test_ratio", 0.1)
    val_ratio = cfg_nofilter["data"].get("val_ratio", 0.1)
    train_ratio = cfg_nofilter["data"].get("train_ratio", 0.8)
    _ = train_ratio

    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size

    train_subset, val_subset, test_subset = random_split(
        ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg_nofilter.get("experiment", {}).get("seed", 42)),
    )
    _ = train_subset
    _ = val_subset
    return [ds.samples[i] for i in test_subset.indices]


def export_sd_benchmark(benchmark_csv: Path, out_csv: Path):
    rows = []
    with benchmark_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    order = {
        "classical_radio": 0,
        "generic_restoration": 1,
        "radio_deep_learning": 2,
    }
    rows.sort(key=lambda r: (order.get(r["category"], 99), -float(r["psnr"])))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "psnr", "ssim", "mse", "mae", "num_samples"])
        for r in rows:
            w.writerow([
                r["category"],
                r["method"],
                f"{float(r['psnr']):.6f}",
                f"{float(r['ssim']):.6f}",
                f"{float(r['mse']):.9f}",
                f"{float(r['mae']):.9f}",
                int(r["num_samples"]),
            ])


def export_normalization_hist(test_samples, out_hist_csv: Path, out_summary_csv: Path, bins: int):
    strategies = ["independent", "ratio_preserving", "arcsinh", "adaptive"]
    edges = np.linspace(0.0, 1.0, bins + 1)
    hist = {s: np.zeros(bins, dtype=np.float64) for s in strategies}
    weak_stats = {s: [] for s in strategies}
    gt_range_stats = {s: [] for s in strategies}

    for sample in test_samples:
        data = {
            "psf": load_fits(sample["psf"]),
            "dirty": load_fits(sample["dirty"]),
            "sd": load_fits(sample["sd"]),
            "gt": load_fits(sample["gt"]),
        }
        for s in strategies:
            cfg = {"data": {"normalization": {"strategy": s}}}
            norm = NormalizationFactory.create(cfg)
            n = norm.normalize(data)
            gt = n["gt"].astype(np.float32)
            h, _ = np.histogram(gt, bins=edges)
            hist[s] += h
            weak_stats[s].append(float((gt < 0.05).mean()))
            gt_range_stats[s].append(float(gt.max() - gt.min()))

    out_hist_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_hist_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "bin_left", "bin_right", "count", "density"])
        for s in strategies:
            total = hist[s].sum()
            for i in range(bins):
                cnt = float(hist[s][i])
                den = cnt / total if total > 0 else 0.0
                w.writerow([s, f"{edges[i]:.6f}", f"{edges[i+1]:.6f}", int(cnt), f"{den:.12f}"])

    with out_summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy",
            "weak_tail_ratio_lt_0p05_mean",
            "weak_tail_ratio_lt_0p05_std",
            "normalized_gt_range_mean",
            "normalized_gt_range_std",
        ])
        for s in strategies:
            weak = np.array(weak_stats[s], dtype=np.float64)
            rng = np.array(gt_range_stats[s], dtype=np.float64)
            w.writerow([
                s,
                f"{weak.mean():.8f}",
                f"{weak.std():.8f}",
                f"{rng.mean():.8f}",
                f"{rng.std():.8f}",
            ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(project_root / "configs/paper/main_unet.yaml"),
    )
    p.add_argument(
        "--benchmark_csv",
        default=str(project_root / "paper/figure_data/sd_multilevel_metrics.csv"),
    )
    p.add_argument(
        "--output_dir",
        default=str(project_root / "paper/figure_data"),
    )
    p.add_argument("--bins", type=int, default=50)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_sd_benchmark(Path(args.benchmark_csv), output_dir / "sd_multilevel_metrics.csv")

    test_samples = build_test_sample_paths(args.config)
    export_normalization_hist(
        test_samples,
        output_dir / "normalization_histograms.csv",
        output_dir / "normalization_tail_summary.csv",
        bins=args.bins,
    )

    manifest = {
        "config": args.config,
        "benchmark_csv": args.benchmark_csv,
        "num_test_samples": len(test_samples),
        "outputs": [
            "sd_multilevel_metrics.csv",
            "normalization_histograms.csv",
            "normalization_tail_summary.csv",
        ],
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved figure data to: {output_dir}")


if __name__ == "__main__":
    main()
