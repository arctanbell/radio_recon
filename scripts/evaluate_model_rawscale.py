#!/usr/bin/env python
"""Evaluate models on the true test split with normalized and raw-scale metrics."""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from radio_recon.data.utils import load_fits
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import random_split
from tqdm import tqdm

from radio_recon.data.dataset import RadioReconstructionDataset
from radio_recon.models.model_factory import create_model_from_config
from radio_recon.utils.config import load_config
from radio_recon.evaluation.metrics import compute_all_metrics
from radio_recon.utils.input_mode import input_mode_from_config, select_model_input


def create_test_dataset(config: dict):
    cfg_nofilter = deepcopy(config)
    cfg_nofilter.setdefault('data', {})
    cfg_nofilter['data'].setdefault('filter', {})
    cfg_nofilter['data']['filter'] = dict(cfg_nofilter['data']['filter'])
    cfg_nofilter['data']['filter']['enabled'] = False

    data_dir = cfg_nofilter['data']['data_dir']
    seed = cfg_nofilter.get('experiment', {}).get('seed', 42)

    base_dataset = RadioReconstructionDataset(
        data_dir=data_dir,
        config=cfg_nofilter,
        split='train',
        augment=False,
    )

    total_size = len(base_dataset)
    test_ratio = cfg_nofilter['data'].get('test_ratio', 0.1)
    val_ratio = cfg_nofilter['data'].get('val_ratio', 0.1)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - test_size - val_size

    _, _, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    eval_dataset = RadioReconstructionDataset(
        data_dir=data_dir,
        config=cfg_nofilter,
        split='val',
        augment=False,
    )
    return eval_dataset, list(test_subset.indices)


def compute_raw_metrics(pred_raw: np.ndarray, target_raw: np.ndarray) -> dict:
    mse = float(np.mean((pred_raw - target_raw) ** 2))
    mae = float(np.mean(np.abs(pred_raw - target_raw)))
    data_max = float(max(pred_raw.max(), target_raw.max()))
    data_min = float(min(pred_raw.min(), target_raw.min()))
    data_range = max(data_max - data_min, 1e-8)
    psnr = float(peak_signal_noise_ratio(target_raw, pred_raw, data_range=data_range))
    ssim = float(structural_similarity(target_raw, pred_raw, data_range=data_range))
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'mae': mae,
        'data_range': data_range,
    }


def summarize(metric_rows):
    keys = metric_rows[0].keys()
    summary = {
        'mean': {k: float(np.mean([row[k] for row in metric_rows])) for k in keys},
        'std': {k: float(np.std([row[k] for row in metric_rows])) for k in keys},
    }
    return summary


def visualize_sample(raw_inputs, pred_raw, target_raw, name, save_path, metrics_raw):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    pred_gt_stack = np.concatenate([pred_raw.ravel(), target_raw.ravel()])
    pred_gt_vmin, pred_gt_vmax = np.percentile(pred_gt_stack, [1, 99])

    axes[0].imshow(raw_inputs['psf'], cmap='RdYlBu_r', origin='lower')
    axes[0].set_title('PSF raw')
    axes[0].axis('off')

    axes[1].imshow(raw_inputs['dirty'], cmap='RdYlBu_r', origin='lower')
    axes[1].set_title('Dirty raw')
    axes[1].axis('off')

    axes[2].imshow(raw_inputs['sd'], cmap='RdYlBu_r', origin='lower')
    axes[2].set_title('SD raw')
    axes[2].axis('off')

    axes[3].imshow(pred_raw, cmap='RdYlBu_r', origin='lower', vmin=pred_gt_vmin, vmax=pred_gt_vmax)
    axes[3].set_title(f'Pred raw\nPSNR={metrics_raw["psnr"]:.2f}')
    axes[3].axis('off')

    axes[4].imshow(target_raw, cmap='RdYlBu_r', origin='lower', vmin=pred_gt_vmin, vmax=pred_gt_vmax)
    axes[4].set_title(f'GT raw\nSSIM={metrics_raw["ssim"]:.4f}')
    axes[4].axis('off')

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def evaluate_model_rawscale(checkpoint_path, config_path, output_dir, num_samples=20, gpu=0):
    config = load_config(config_path)
    exp_name = config['experiment']['name']
    print(f'评估实验: {exp_name}')
    print(f'Checkpoint: {checkpoint_path}')

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = create_model_from_config(config).to(device)
    input_mode = input_mode_from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    dataset, test_indices = create_test_dataset(config)
    print(f'Test split size: {len(test_indices)}')

    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    norm_rows = []
    raw_rows = []

    with torch.no_grad():
        for vis_idx, dataset_idx in enumerate(tqdm(test_indices, desc='Testing raw-scale')):
            condition, target, name = dataset[dataset_idx]
            condition_batch = condition.unsqueeze(0).to(device)
            pred = model(select_model_input(condition_batch, input_mode))[0, 0].cpu().numpy()
            target_norm = target[0].numpy()

            metrics_norm = compute_all_metrics(pred, target_norm)
            norm_rows.append({k: float(v) for k, v in metrics_norm.items()})

            sample = dataset.samples[dataset_idx]
            raw_inputs = {
                'psf': load_fits(sample['psf']),
                'dirty': load_fits(sample['dirty']),
                'sd': load_fits(sample['sd']),
            }
            target_raw = dataset.normalizer.denormalize({'gt': target_norm.copy()})['gt']
            pred_raw = dataset.normalizer.denormalize({'gt': pred.copy()})['gt']
            metrics_raw = compute_raw_metrics(pred_raw, target_raw)
            raw_rows.append(metrics_raw)

            if vis_idx < num_samples:
                visualize_sample(
                    raw_inputs,
                    pred_raw,
                    target_raw,
                    name,
                    os.path.join(viz_dir, f'sample_{vis_idx:03d}_{name}.png'),
                    metrics_raw,
                )

    results = {
        'experiment': exp_name,
        'checkpoint': str(checkpoint_path),
        'num_samples': len(test_indices),
        'normalized_metrics': summarize(norm_rows),
        'raw_metrics': summarize(raw_rows),
    }

    with open(os.path.join(output_dir, 'rawscale_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 60)
    print(f'Normalized PSNR: {results["normalized_metrics"]["mean"]["psnr"]:.2f}')
    print(f'Normalized SSIM: {results["normalized_metrics"]["mean"]["ssim"]:.4f}')
    print(f'Raw PSNR:        {results["raw_metrics"]["mean"]["psnr"]:.2f}')
    print(f'Raw SSIM:        {results["raw_metrics"]["mean"]["ssim"]:.4f}')
    print(f'Raw MSE:         {results["raw_metrics"]["mean"]["mse"]:.6e}')
    print(f'Raw MAE:         {results["raw_metrics"]["mean"]["mae"]:.6e}')
    print('=' * 60)
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    evaluate_model_rawscale(
        args.checkpoint,
        args.config,
        args.output_dir,
        args.num_samples,
        args.gpu,
    )
