#!/usr/bin/env python

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent


def load_norm_hist(csv_path: Path):
    data = defaultdict(lambda: {"x": [], "y": []})
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = row["strategy"]
            left = float(row["bin_left"])
            right = float(row["bin_right"])
            center = 0.5 * (left + right)
            density = float(row["density"])
            data[strategy]["x"].append(center)
            data[strategy]["y"].append(density)
    return data


def load_norm_summary(csv_path: Path):
    out = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["strategy"]] = row
    return out


def plot_normalization_hist(hist_csv: Path, summary_csv: Path, out_pdf: Path, out_png: Path):
    hist = load_norm_hist(hist_csv)
    summary = load_norm_summary(summary_csv)
    colors = {
        "independent": "#1f77b4",
        "ratio_preserving": "#d62728",
        "arcsinh": "#2ca02c",
        "adaptive": "#9467bd",
    }
    labels = {
        "independent": "Independent",
        "ratio_preserving": "Ratio-preserving",
        "arcsinh": "Arcsinh",
        "adaptive": "Adaptive",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    order = ["independent", "adaptive", "arcsinh", "ratio_preserving"]
    for k in order:
        ax.plot(hist[k]["x"], hist[k]["y"], lw=2.0, color=colors[k], label=labels[k])

    ax.set_xlabel("Normalized GT intensity")
    ax.set_ylabel("Density")
    ax.set_title("Normalization Strategy Comparison (GT Histogram)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    tail_text = []
    for k in order:
        weak = float(summary[k]["weak_tail_ratio_lt_0p05_mean"])
        tail_text.append(f"{labels[k]}: low-intensity occupancy={weak:.3f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(tail_text),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_sd_metrics(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["psnr"] = float(row["psnr"])
            row["ssim"] = float(row["ssim"])
            row["mse"] = float(row["mse"])
            row["mae"] = float(row["mae"])
            rows.append(row)
    return rows


def plot_sd_comparison(sd_csv: Path, out_pdf: Path, out_png: Path):
    rows = load_sd_metrics(sd_csv)
    order = {
        "classical_radio": 0,
        "input_only": 1,
        "generic_restoration": 2,
        "radio_deep_learning": 3,
    }
    rows.sort(key=lambda r: (order[r["group"]], r["psnr"]))

    labels = []
    psnr = []
    colors = []
    c_map = {
        "classical_radio": "#8c564b",
        "input_only": "#2ca02c",
        "generic_restoration": "#ff7f0e",
        "radio_deep_learning": "#1f77b4",
    }
    for r in rows:
        method = r["method"].replace("-standard", "")
        labels.append(method)
        psnr.append(r["psnr"])
        colors.append(c_map[r["group"]])

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(x, psnr, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Shared 1,272-Sample Benchmark")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, psnr):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#8c564b", label="Classical"),
        Patch(color="#2ca02c", label="Input-only"),
        Patch(color="#ff7f0e", label="Generic (dirty-only)"),
        Patch(color="#1f77b4", label="Radio DL (PSF+dirty+SD)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sd_training_delta(sd_train_csv: Path, out_pdf: Path, out_png: Path):
    rows = load_sd_metrics(sd_train_csv)
    with_sd = {}
    no_sd = {}
    for r in rows:
        method = r["method"]
        if "(with SD)" in method:
            key = method.replace(" (with SD)", "")
            with_sd[key] = r
        if "(trained no SD)" in method:
            key = method.replace(" (trained no SD)", "")
            no_sd[key] = r

    backbones = [k for k in ["Radio-FiLM-UNet", "Radio-UNet-Structure", "Radio-SwinIR"] if k in with_sd and k in no_sd]
    delta = [with_sd[k]["psnr"] - no_sd[k]["psnr"] for k in backbones]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(backbones))
    bars = ax.bar(x, delta, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("Radio-", "") for b in backbones], rotation=15, ha="right")
    ax.set_ylabel("PSNR Gain from SD (dB)")
    ax.set_title("Same-Backbone SD Ablation (N=1272)")
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, delta):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.12, f"+{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        default=str(project_root / "paper/figure_data"),
    )
    p.add_argument(
        "--output_dir",
        default=str(project_root / "paper/figures"),
    )
    p.add_argument(
        "--sd_train_csv",
        default=str(project_root / "paper/figure_data/sd_training_ablation_metrics.csv"),
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    plot_normalization_hist(
        input_dir / "normalization_histograms.csv",
        input_dir / "normalization_tail_summary.csv",
        output_dir / "normalization_histogram_comparison.pdf",
        output_dir / "normalization_histogram_comparison.png",
    )

    plot_sd_comparison(
        input_dir / "sd_multilevel_metrics.csv",
        output_dir / "sd_ablation_comparison.pdf",
        output_dir / "sd_ablation_comparison.png",
    )

    plot_sd_training_delta(
        Path(args.sd_train_csv),
        output_dir / "sd_training_delta_comparison.pdf",
        output_dir / "sd_training_delta_comparison.png",
    )

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
