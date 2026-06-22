"""Dataset and dataloader helpers for PC-RIIR experiments."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from radio_recon.data.normalization import NormalizationFactory
from radio_recon.data.utils import add_gaussian_noise, center_crop, find_dirty_fits, load_fits

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RadioReconstructionDataset(Dataset):
    """Load PSF, dirty image, single-dish map, and reference target FITS files."""

    def __init__(
        self,
        data_dir: str,
        config: Optional[dict] = None,
        split: str = "train",
        augment: bool = False,
        image_size: Optional[int] = None,
        gt_suffix: str = "_rg_dirty.fits",
    ):
        self.data_dir = str(data_dir)
        self.config = config or {}
        self.split = split
        self.augment = augment
        self.gt_suffix = gt_suffix
        self.image_size = image_size or self.config.get("data", {}).get("image_size")
        self.normalizer = NormalizationFactory.create(self.config)
        self.samples = self._find_samples()
        self.samples = self._apply_filter(self.samples)
        if not self.samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")

    def _find_samples(self) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        for subdir in sorted(os.listdir(self.data_dir)):
            subdir_path = os.path.join(self.data_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            psf_files = sorted(glob.glob(os.path.join(subdir_path, "*_dirty.psf.fits")))
            dirty_file = find_dirty_fits(subdir_path)
            sd_files = sorted(glob.glob(os.path.join(subdir_path, "*_rg_fast.fits")))
            gt_files = sorted(glob.glob(os.path.join(subdir_path, f"*{self.gt_suffix}")))

            if psf_files and dirty_file and sd_files and gt_files:
                samples.append(
                    {
                        "psf": psf_files[0],
                        "dirty": dirty_file,
                        "sd": sd_files[0],
                        "gt": gt_files[0],
                        "name": subdir,
                    }
                )
        return samples

    def _resolve_filter_path(self, filter_path: str) -> Path:
        path = Path(filter_path)
        if path.is_absolute():
            return path
        for base in (PROJECT_ROOT, Path(self.data_dir)):
            candidate = base / path
            if candidate.exists():
                return candidate
        return PROJECT_ROOT / path

    def _apply_filter(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        filter_cfg = self.config.get("data", {}).get("filter", {})
        if not filter_cfg.get("enabled", False):
            return samples
        if self.split not in filter_cfg.get("splits", ["train", "val", "test"]):
            return samples

        exclude_list = filter_cfg.get("exclude_list")
        if not exclude_list:
            return samples
        path = self._resolve_filter_path(exclude_list)
        if not path.exists():
            raise FileNotFoundError(f"Filter exclude_list does not exist: {path}")

        excluded = {
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if not excluded:
            return samples

        def keep(sample: Dict[str, str]) -> bool:
            name = sample["name"]
            path_names = {Path(value).name for key, value in sample.items() if key != "name"}
            return name not in excluded and path_names.isdisjoint(excluded)

        return [sample for sample in samples if keep(sample)]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample_arrays(self, sample: Dict[str, str]) -> Dict[str, np.ndarray]:
        arrays = {
            "psf": load_fits(sample["psf"]),
            "dirty": load_fits(sample["dirty"]),
            "sd": load_fits(sample["sd"]),
            "gt": load_fits(sample["gt"]),
        }
        shapes = {key: value.shape for key, value in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Shape mismatch for {sample['name']}: {shapes}")
        if self.image_size:
            arrays = {key: center_crop(value, int(self.image_size)) for key, value in arrays.items()}
        return arrays

    def _augment(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        aug_cfg = self.config.get("data", {}).get("augmentation", {})
        if not self.augment or not aug_cfg.get("enabled", self.augment):
            return arrays

        keys = ["psf", "dirty", "sd", "gt"]
        stacked = np.stack([arrays[key] for key in keys], axis=0)
        flip_prob = float(aug_cfg.get("flip_prob", 0.5))
        if np.random.random() < flip_prob:
            stacked = np.flip(stacked, axis=1)
        if np.random.random() < flip_prob:
            stacked = np.flip(stacked, axis=2)
        if np.random.random() < float(aug_cfg.get("rotate_prob", 0.5)):
            stacked = np.rot90(stacked, int(np.random.randint(0, 4)), axes=(1, 2))
        stacked = np.ascontiguousarray(stacked)

        out = {key: stacked[i] for i, key in enumerate(keys)}
        noise_sigma = float(aug_cfg.get("noise_sigma", 0.0))
        if noise_sigma > 0:
            out["dirty"] = np.clip(add_gaussian_noise(out["dirty"], noise_sigma), 0.0, 1.0)
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        raw = self._load_sample_arrays(sample)
        normalized = self.normalizer.normalize(raw)
        normalized = self._augment(normalized)

        psf = torch.from_numpy(normalized["psf"].astype(np.float32, copy=False)).unsqueeze(0)
        dirty = torch.from_numpy(normalized["dirty"].astype(np.float32, copy=False)).unsqueeze(0)
        sd = torch.from_numpy(normalized["sd"].astype(np.float32, copy=False)).unsqueeze(0)
        target = torch.from_numpy(normalized["gt"].astype(np.float32, copy=False)).unsqueeze(0)

        dropout_prob = float(self.config.get("data", {}).get("sd_dropout_prob", 0.0))
        if self.augment and dropout_prob > 0 and np.random.random() < dropout_prob:
            sd = torch.zeros_like(sd)

        condition = torch.cat([psf, dirty, sd], dim=0)
        return condition.float(), target.float(), sample["name"]


RadioDiffusionDataset = RadioReconstructionDataset


def _split_indices(total_size: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    test_size = int(total_size * (1.0 - train_ratio - val_ratio))
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices


def create_dataloaders(config: dict, num_workers: Optional[int] = None):
    """Create train, validation, and test dataloaders from a config dict."""
    data_cfg = config.get("data", {})
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    seed = int(config.get("experiment", {}).get("seed", 42))
    workers = int(num_workers if num_workers is not None else data_cfg.get("num_workers", 0))
    batch_size = int(config.get("training", {}).get("batch_size", 4))
    data_dir = data_cfg["data_dir"]

    split_config = dict(config)
    split_config["data"] = dict(data_cfg)
    split_config["data"]["filter"] = dict(data_cfg.get("filter", {}))
    split_config["data"]["filter"]["enabled"] = False

    full_dataset = RadioReconstructionDataset(data_dir=data_dir, config=split_config, split="all", augment=False)
    train_idx, val_idx, test_idx = _split_indices(len(full_dataset), train_ratio, val_ratio, seed)

    train_dataset = RadioReconstructionDataset(data_dir=data_dir, config=split_config, split="train", augment=True)
    val_dataset = RadioReconstructionDataset(data_dir=data_dir, config=split_config, split="val", augment=False)
    test_dataset = RadioReconstructionDataset(data_dir=data_dir, config=split_config, split="test", augment=False)

    filter_cfg = data_cfg.get("filter", {})
    if filter_cfg.get("enabled", False) and "train" in filter_cfg.get("splits", ["train"]):
        filtered_train = RadioReconstructionDataset(data_dir=data_dir, config=config, split="train", augment=False)
        allowed_names = {sample["name"] for sample in filtered_train.samples}
        train_idx = [idx for idx in train_idx if full_dataset.samples[idx]["name"] in allowed_names]

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
