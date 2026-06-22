"""FITS loading and image normalization utilities."""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import numpy as np


def load_fits(filepath: str) -> np.ndarray:
    """Load a FITS image and return a 2D float32 array."""
    from astropy.io import fits

    with fits.open(filepath, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D FITS data after squeeze, got shape {data.shape}: {filepath}")
    return data


def find_dirty_fits(sample_dir: str) -> Optional[str]:
    """Return the preferred dirty-image FITS path for a sample directory."""
    for pattern in ("*_dirty.image.pbcor.fits", "*_dirty.image.fits"):
        matches = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        if matches:
            return matches[0]
    return None


def robust_minmax(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Percentile clip and scale an image to [0, 1]."""
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32), (0.0, 0.0)

    values = image[finite]
    vmin = float(np.percentile(values, percentile_low))
    vmax = float(np.percentile(values, percentile_high))
    if vmax - vmin <= 1e-10:
        return np.zeros_like(image, dtype=np.float32), (vmin, vmax)

    normalized = (image - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return normalized.astype(np.float32, copy=False), (vmin, vmax)


def normalize_image(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    return_stats: bool = False,
):
    """Normalize an image with percentile clipping."""
    normalized, stats = robust_minmax(image, percentile_low, percentile_high)
    if return_stats:
        return normalized, stats
    return normalized


def normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Normalize a PSF by its sum before display/range normalization."""
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf_sum = float(np.sum(psf))
    if abs(psf_sum) <= 1e-10:
        return psf
    return psf / psf_sum


def denormalize_image(normalized: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Invert robust min-max scaling."""
    return normalized * (vmax - vmin) + vmin


def center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    """Center crop a square 2D image."""
    h, w = image.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} exceeds image shape {image.shape}")
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return image[start_h:start_h + crop_size, start_w:start_w + crop_size]


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to a normalized image."""
    noise = np.random.normal(0.0, sigma, image.shape).astype(image.dtype)
    return image + noise
