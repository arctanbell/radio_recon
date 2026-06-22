#!/usr/bin/env python
"""Run CASA tclean baselines on the shared simulated test split.

This script is intended for paper baselines, not model training. It evaluates
CASA-level Hogbom and multi-scale tclean outputs on the same split used by the
paper tables.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from astropy.io import fits
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


METHODS = {
    "hogbom": {
        "deconvolver": "hogbom",
        "niter": 1000,
        "threshold": "1mJy",
        "extra": "",
    },
    "multiscale": {
        "deconvolver": "multiscale",
        "niter": 1000,
        "threshold": "1mJy",
        "extra": "scales=[0, 2, 5],",
    },
}


def build_test_samples(config_path: Path) -> list[dict]:
    cfg = load_config(config_path)
    data_dir = Path(cfg["data"]["data_dir"])
    samples = []
    for sample_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        psf = sorted(sample_dir.glob("*_dirty.psf.fits"))
        dirty = sorted(sample_dir.glob("*_dirty.image.pbcor.fits")) or sorted(sample_dir.glob("*_dirty.image.fits"))
        sd = sorted(sample_dir.glob("*_rg_fast.fits"))
        gt = sorted(sample_dir.glob("*_rg_dirty.fits"))
        if psf and dirty and sd and gt:
            samples.append(
                {
                    "psf": str(psf[0]),
                    "dirty": str(dirty[0]),
                    "sd": str(sd[0]),
                    "gt": str(gt[0]),
                    "name": sample_dir.name,
                }
            )

    total_size = len(samples)
    test_size = int(total_size * cfg["data"].get("test_ratio", 0.1))
    val_size = int(total_size * cfg["data"].get("val_ratio", 0.1))
    train_size = total_size - test_size - val_size

    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(cfg.get("experiment", {}).get("seed", 42))).tolist()
    test_indices = indices[train_size + val_size :]
    return [samples[i] for i in test_indices]


def load_config(path: Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_fits(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        data = np.squeeze(hdul[0].data).astype(np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D FITS data, got {data.shape}: {path}")
    return data


def norm_linear(array: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    if vmin is None:
        vmin = float(np.percentile(array, 1.0))
    if vmax is None:
        vmax = float(np.percentile(array, 99.0))
    if vmax - vmin <= 1e-10:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def sample_ms_path(sample: dict) -> Path:
    sample_dir = Path(sample["gt"]).parent
    name = sample["name"]
    concat_ms = sample_dir / f"{name}_concat.ms"
    if concat_ms.exists():
        return concat_ms
    day_ms = sorted(sample_dir.glob(f"{name}_d*/*noisy.ms"))
    if day_ms:
        return day_ms[0]
    day_ms = sorted(sample_dir.glob(f"{name}_d*/*.ms"))
    if day_ms:
        return day_ms[0]
    raise FileNotFoundError(f"No MS found for sample {name} in {sample_dir}")


def write_casa_script(script_path: Path, sample: dict, method_name: str, product_dir: Path) -> tuple[Path, Path]:
    method = METHODS[method_name]
    sample_dir = Path(sample["gt"]).parent
    name = sample["name"]
    vis = sample_ms_path(sample)
    imagename = product_dir / method_name / name / f"{name}_{method_name}_tclean"
    fits_path = Path(f"{imagename}.image.pbcor.fits")
    imagename.parent.mkdir(parents=True, exist_ok=True)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    script = f"""
import glob
import os

vis = r"{vis}"
imagename = r"{imagename}"

def clear_products(base):
    for path in glob.glob(base + ".*"):
        if os.path.isdir(path):
            os.system('rm -rf "%s"' % path)
        else:
            os.remove(path)

clear_products(imagename)
tclean(
    vis=[vis],
    imagename=imagename,
    field="",
    spw="",
    specmode="mfs",
    imsize=[192, 192],
    cell="5arcsec",
    weighting="briggs",
    robust=0.5,
    niter={method["niter"]},
    gain=0.1,
    threshold="{method["threshold"]}",
    nterms=1,
    gridder="standard",
    deconvolver="{method["deconvolver"]}",
    {method["extra"]}
    savemodel="none",
    pbcor=True,
)
exportfits(imagename=imagename + ".image", fitsimage=imagename + ".image.fits", overwrite=True, dropstokes=True)
if os.path.exists(imagename + ".image.pbcor"):
    exportfits(imagename=imagename + ".image.pbcor", fitsimage=imagename + ".image.pbcor.fits", overwrite=True, dropstokes=True)
if os.path.exists(imagename + ".psf"):
    exportfits(imagename=imagename + ".psf", fitsimage=imagename + ".psf.fits", overwrite=True, dropstokes=True)
if os.path.exists(imagename + ".pb"):
    exportfits(imagename=imagename + ".pb", fitsimage=imagename + ".pb.fits", overwrite=True, dropstokes=True)
"""
    script_path.write_text(script, encoding="utf-8")
    return script_path, fits_path


def normalized_metrics(pred_raw: np.ndarray, sample: dict, config: dict) -> dict:
    strategy = config.get("data", {}).get("normalization", {}).get("strategy", "independent")
    target_raw = load_fits(sample["gt"])
    if strategy != "independent":
        raise ValueError(f"Only independent normalization is supported for CASA tclean baseline, got {strategy}")
    pred = norm_linear(pred_raw)
    target = norm_linear(target_raw)
    return {
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, data_range=1.0)),
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
    }


def raw_metrics(pred_raw: np.ndarray, sample: dict) -> dict:
    target = load_fits(sample["gt"]).astype(np.float32)
    pred = pred_raw.astype(np.float32)
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return {"raw_psnr": np.nan, "raw_ssim": np.nan, "raw_mse": np.nan, "raw_mae": np.nan}
    pred = pred[mask]
    target_flat = target[mask]
    data_range = float(np.max(target_flat) - np.min(target_flat))
    if data_range <= 1e-12:
        data_range = 1.0
    target_img = target.copy()
    pred_img = pred_raw.copy()
    pred_img[~np.isfinite(pred_img)] = 0.0
    target_img[~np.isfinite(target_img)] = 0.0
    return {
        "raw_psnr": float(peak_signal_noise_ratio(target_flat, pred, data_range=data_range)),
        "raw_ssim": float(structural_similarity(target_img, pred_img, data_range=data_range)),
        "raw_mse": float(np.mean((pred - target_flat) ** 2)),
        "raw_mae": float(np.mean(np.abs(pred - target_flat))),
    }


def summarize(rows: list[dict], method_name: str) -> dict:
    method_rows = [r for r in rows if r["method"] == method_name and r["status"] == "ok"]
    metric_names = ["psnr", "ssim", "mse", "mae", "raw_psnr", "raw_ssim", "raw_mse", "raw_mae"]
    return {
        "method": method_name,
        "num_samples": len(method_rows),
        "metrics": {
            name: {
                "mean": float(np.nanmean([r[name] for r in method_rows])) if method_rows else np.nan,
                "std": float(np.nanstd([r[name] for r in method_rows])) if method_rows else np.nan,
            }
            for name in metric_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/main_unet.yaml")
    parser.add_argument("--output-dir", default="outputs/casa_tclean_baseline")
    parser.add_argument("--casa-wrapper", default=os.environ.get("CASA_WRAPPER", "casa"))
    parser.add_argument(
        "--casa-path-prefix",
        default=os.environ.get("CASA_PATH_PREFIX", ""),
        help="Optional PATH prefix for CASA dependencies, separated with os.pathsep.",
    )
    parser.add_argument("--methods", nargs="+", default=["hogbom", "multiscale"], choices=sorted(METHODS))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    out_dir = Path(args.output_dir)
    scripts_dir = out_dir / "casa_scripts"
    products_dir = out_dir / "products"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_test_samples(cfg_path)
    if args.start_index:
        samples = samples[args.start_index :]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    manifest_rows = [{"index": i + args.start_index, "name": s["name"], **s} for i, s in enumerate(samples)]
    with (out_dir / "test_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "name", "psf", "dirty", "sd", "gt"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    rows = []
    for i, sample in enumerate(samples, start=args.start_index):
        for method in args.methods:
            casa_script, fits_path = write_casa_script(
                scripts_dir / f"{i:05d}_{sample['name']}_{method}.py",
                sample,
                method,
                products_dir,
            )
            status = "ok"
            error = ""
            if not (args.skip_existing and fits_path.exists()):
                cmd = [args.casa_wrapper, "-c", str(casa_script)]
                env = os.environ.copy()
                if args.casa_path_prefix:
                    env["PATH"] = args.casa_path_prefix + os.pathsep + env.get("PATH", "")
                proc = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True, env=env)
                (out_dir / "logs").mkdir(exist_ok=True)
                (out_dir / "logs" / f"{i:05d}_{sample['name']}_{method}.stdout").write_text(proc.stdout, encoding="utf-8")
                (out_dir / "logs" / f"{i:05d}_{sample['name']}_{method}.stderr").write_text(proc.stderr, encoding="utf-8")
                if proc.returncode != 0:
                    status = "failed"
                    error = f"casa_returncode={proc.returncode}"
            if status == "ok" and not fits_path.exists():
                status = "failed"
                error = f"missing_fits={fits_path}"

            result = {
                "index": i,
                "sample": sample["name"],
                "method": method,
                "status": status,
                "fits": str(fits_path),
                "error": error,
                "psnr": np.nan,
                "ssim": np.nan,
                "mse": np.nan,
                "mae": np.nan,
                "raw_psnr": np.nan,
                "raw_ssim": np.nan,
                "raw_mse": np.nan,
                "raw_mae": np.nan,
            }
            if status == "ok":
                pred_raw = load_fits(str(fits_path))
                result.update(normalized_metrics(pred_raw, sample, cfg))
                result.update(raw_metrics(pred_raw, sample))
            rows.append(result)

            with (out_dir / "per_sample_metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = {
                "config": str(cfg_path),
                "output_dir": str(out_dir),
                "methods": args.methods,
                "num_requested_samples": len(samples),
                "summaries": [summarize(rows, m) for m in args.methods],
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[{i}] {sample['name']} / {method}: {status}")

    print(f"Saved CASA tclean baseline to {out_dir}")


if __name__ == "__main__":
    main()
