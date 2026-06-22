"""Normalization strategies used by the manuscript experiments."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from radio_recon.data.utils import denormalize_image, normalize_psf, robust_minmax


ArrayDict = Dict[str, np.ndarray]
StatsDict = Dict[str, Dict[str, float]]


class BaseNormalizer:
    """Base class storing per-sample normalization statistics."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.stats: StatsDict = {}

    def normalize(self, data: ArrayDict) -> ArrayDict:
        raise NotImplementedError

    def denormalize(self, data: ArrayDict) -> ArrayDict:
        restored: ArrayDict = {}
        for key, array in data.items():
            stats = self.stats.get(key)
            if not stats:
                restored[key] = array
                continue
            if stats.get("mode") == "arcsinh":
                scale = max(stats.get("scale", 3.0), 1e-8)
                linear = np.sinh(array * np.arcsinh(scale)) / scale
                restored[key] = denormalize_image(linear, stats["vmin"], stats["vmax"])
            else:
                restored[key] = denormalize_image(array, stats["vmin"], stats["vmax"])
        return restored

    def _normalize_key(self, key: str, array: np.ndarray) -> np.ndarray:
        normalized, (vmin, vmax) = robust_minmax(array.astype(np.float32, copy=False))
        self.stats[key] = {"vmin": vmin, "vmax": vmax, "mode": "linear"}
        return normalized

    def _normalize_psf(self, psf: np.ndarray) -> np.ndarray:
        psf = normalize_psf(psf)
        return self._normalize_key("psf", psf)

    def _normalize_arcsinh(self, key: str, array: np.ndarray, scale: float = 3.0) -> np.ndarray:
        linear, (vmin, vmax) = robust_minmax(array.astype(np.float32, copy=False))
        stretched = np.arcsinh(scale * linear) / np.arcsinh(scale)
        self.stats[key] = {"vmin": vmin, "vmax": vmax, "mode": "arcsinh", "scale": scale}
        return stretched.astype(np.float32, copy=False)


class IndependentNormalizer(BaseNormalizer):
    """Per-sample, per-modality percentile normalization."""

    def normalize(self, data: ArrayDict) -> ArrayDict:
        self.stats = {}
        output: ArrayDict = {}
        if "psf" in data:
            output["psf"] = self._normalize_psf(data["psf"])
        for key in ("dirty", "sd", "gt"):
            if key in data:
                output[key] = self._normalize_key(key, data[key])
        return output


class RatioPreservingNormalizer(BaseNormalizer):
    """Normalize dirty and target with shared robust bounds."""

    def normalize(self, data: ArrayDict) -> ArrayDict:
        self.stats = {}
        output: ArrayDict = {}
        if "psf" in data:
            output["psf"] = self._normalize_psf(data["psf"])

        if "dirty" in data and "gt" in data:
            stacked = np.stack([data["dirty"], data["gt"]], axis=0).astype(np.float32, copy=False)
            normalized, (vmin, vmax) = robust_minmax(stacked)
            output["dirty"] = normalized[0]
            output["gt"] = normalized[1]
            shared = {"vmin": vmin, "vmax": vmax, "mode": "linear"}
            self.stats["dirty"] = dict(shared)
            self.stats["gt"] = dict(shared)
        else:
            for key in ("dirty", "gt"):
                if key in data:
                    output[key] = self._normalize_key(key, data[key])

        if "sd" in data:
            output["sd"] = self._normalize_key("sd", data["sd"])
        return output


class ArcsinhNormalizer(BaseNormalizer):
    """Robust min-max normalization followed by arcsinh stretching."""

    def normalize(self, data: ArrayDict) -> ArrayDict:
        self.stats = {}
        output: ArrayDict = {}
        if "psf" in data:
            output["psf"] = self._normalize_psf(data["psf"])
        for key in ("dirty", "sd", "gt"):
            if key in data:
                output[key] = self._normalize_arcsinh(key, data[key])
        return output


class AdaptiveNormalizer(BaseNormalizer):
    """Historical mixed strategy used for normalization ablations."""

    def normalize(self, data: ArrayDict) -> ArrayDict:
        self.stats = {}
        output: ArrayDict = {}
        if "psf" in data:
            output["psf"] = self._normalize_psf(data["psf"])
        if "dirty" in data:
            output["dirty"] = self._normalize_arcsinh("dirty", data["dirty"])
        for key in ("sd", "gt"):
            if key in data:
                output[key] = self._normalize_key(key, data[key])
        return output


class NormalizationFactory:
    """Create normalizers from experiment configs."""

    @staticmethod
    def create(config: dict | None = None) -> BaseNormalizer:
        cfg = config or {}
        strategy = (
            cfg.get("data", {})
            .get("normalization", {})
            .get("strategy", "independent")
        )
        if strategy == "independent":
            return IndependentNormalizer(cfg)
        if strategy in {"ratio", "ratio_preserving", "paired_dirty_gt"}:
            return RatioPreservingNormalizer(cfg)
        if strategy == "arcsinh":
            return ArcsinhNormalizer(cfg)
        if strategy == "adaptive":
            return AdaptiveNormalizer(cfg)
        raise ValueError(f"Unknown normalization strategy: {strategy}")
