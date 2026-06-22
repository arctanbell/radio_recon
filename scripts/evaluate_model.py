#!/usr/bin/env python
"""
评估训练好的模型

详细评估最佳模型在测试集上的表现
"""

import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from radio_recon.data.dataset import RadioReconstructionDataset, create_dataloaders
from radio_recon.models.model_factory import create_model_from_config
from radio_recon.utils.config import load_config
from radio_recon.evaluation.metrics import compute_all_metrics
from radio_recon.utils.input_mode import input_mode_from_config, select_model_input


def evaluate_model(checkpoint_path, config_path, output_dir, num_samples=10):
    """评估模型"""
    
    # Load config
    config = load_config(config_path)
    exp_name = config['experiment']['name']
    
    print(f"评估实验: {exp_name}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model_from_config(config).to(device)
    input_mode = input_mode_from_config(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, num_workers=4)
    
    # Evaluate on test set
    print("\n评估测试集...")
    all_metrics = []
    sample_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    with torch.no_grad():
        for condition, target, names in tqdm(test_loader, desc="Testing"):
            condition = condition.to(device)
            target = target.to(device)
            
            # Generate predictions
            pred = model(select_model_input(condition, input_mode))
            
            # Compute metrics for each sample in batch
            for i in range(pred.shape[0]):
                pred_np = pred[i, 0].cpu().numpy()
                target_np = target[i, 0].cpu().numpy()
                
                metrics = compute_all_metrics(pred_np, target_np)
                all_metrics.append(metrics)
                
                # Visualize first N samples
                if sample_count < num_samples:
                    visualize_sample(
                        condition[i].cpu().numpy(),
                        pred_np,
                        target_np,
                        metrics,
                        names[i],
                        os.path.join(viz_dir, f'sample_{sample_count:03d}_{names[i]}.png')
                    )
                    sample_count += 1
    
    # Compute statistics
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}
    
    # Print results
    print("\n" + "="*60)
    print(f"测试集结果 ({len(all_metrics)} 个样本)")
    print("="*60)
    print(f"PSNR: {avg_metrics['psnr']:.2f} ± {std_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim']:.4f}")
    print(f"MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse']:.6f}")
    print(f"MAE:  {avg_metrics['mae']:.6f} ± {std_metrics['mae']:.6f}")
    print("="*60)
    
    # Save results
    import json
    results = {
        'experiment': exp_name,
        'checkpoint': str(checkpoint_path),
        'num_samples': len(all_metrics),
        'metrics': {
            'mean': {k: float(v) for k, v in avg_metrics.items()},
            'std': {k: float(v) for k, v in std_metrics.items()},
        }
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"可视化已保存到: {viz_dir}")
    
    return avg_metrics, std_metrics


def visualize_sample(condition, pred, target, metrics, name, save_path):
    """可视化单个样本"""
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Use a shared display range for prediction and GT so the visual
    # comparison is not distorted by per-panel autoscaling.
    stacked = np.concatenate([pred.ravel(), target.ravel()])
    vmin = np.percentile(stacked, 1)
    vmax = np.percentile(stacked, 99)
    if vmax - vmin < 1e-10:
        vmin = float(stacked.min())
        vmax = float(stacked.max())
    
    # PSF
    axes[0].imshow(condition[0], cmap='RdYlBu_r', origin='lower')
    axes[0].set_title('PSF')
    axes[0].axis('off')
    
    # Dirty
    axes[1].imshow(condition[1], cmap='RdYlBu_r', origin='lower')
    axes[1].set_title('Dirty')
    axes[1].axis('off')
    
    # SD
    axes[2].imshow(condition[2], cmap='RdYlBu_r', origin='lower')
    axes[2].set_title('SD')
    axes[2].axis('off')
    
    # Prediction
    axes[3].imshow(pred, cmap='RdYlBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[3].set_title(f'Predicted\nPSNR={metrics["psnr"]:.2f}dB')
    axes[3].axis('off')
    
    # Ground Truth
    axes[4].imshow(target, cmap='RdYlBu_r', origin='lower', vmin=vmin, vmax=vmax)
    axes[4].set_title(f'Ground Truth\nSSIM={metrics["ssim"]:.4f}')
    axes[4].axis('off')
    
    plt.suptitle(name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--config', required=True, help='Path to config')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.output_dir, args.num_samples)
