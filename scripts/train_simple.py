#!/usr/bin/env python
"""
简化的训练脚本 - MVP版本

快速启动训练，专注于验证归一化修复效果
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        class SummaryWriter:  # type: ignore[override]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

from radio_recon.data.dataset import create_dataloaders
from radio_recon.losses.combined import create_loss_from_config
from radio_recon.utils.config import load_config
from radio_recon.evaluation.metrics import compute_all_metrics
from radio_recon.models.model_factory import create_model_from_config
from radio_recon.utils.input_mode import input_mode_from_config, select_model_input


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, global_step, input_mode='all'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (condition, target, names) in enumerate(pbar):
        condition = condition.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward with mixed precision
        with autocast():
            pred = model(select_model_input(condition, input_mode))
            result = criterion(pred, target)
            if isinstance(result, tuple):  # CombinedLoss returns (loss, dict)
                loss, loss_dict = result
            else:
                loss = result
                loss_dict = {}
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Log to TensorBoard
        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step[0])
            for name, value in loss_dict.items():
                writer.add_scalar(f'train/{name}', value, global_step[0])
            global_step[0] += 1
    
    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, writer, input_mode='all'):
    """Validate."""
    model.eval()
    total_loss = 0.0
    all_metrics = []
    
    for condition, target, names in tqdm(val_loader, desc="Validating"):
        condition = condition.to(device)
        target = target.to(device)
        
        pred = model(select_model_input(condition, input_mode))
        
        result = criterion(pred, target)
        if isinstance(result, tuple):
            loss, _ = result
        else:
            loss = result
        
        total_loss += loss.item()
        
        # Compute metrics on the full validation set so checkpoint
        # selection tracks the same objectives we care about later.
        for i in range(pred.shape[0]):
            pred_np = pred[i, 0].cpu().numpy()
            target_np = target[i, 0].cpu().numpy()
            metrics = compute_all_metrics(pred_np, target_np)
            all_metrics.append(metrics)
    
    avg_loss = total_loss / max(1, len(val_loader))
    
    # Average metrics
    avg_metrics = None
    if all_metrics:
        avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        writer.add_scalar('val/psnr', avg_metrics['psnr'], epoch)
        writer.add_scalar('val/ssim', avg_metrics['ssim'], epoch)
        writer.add_scalar('val/mse', avg_metrics['mse'], epoch)
        writer.add_scalar('val/mae', avg_metrics['mae'], epoch)
        print(f"  Val: Loss={avg_loss:.6f}, PSNR={avg_metrics['psnr']:.2f}, SSIM={avg_metrics['ssim']:.4f}")
    else:
        print(f"  Val: Loss={avg_loss:.6f}")

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    exp_name = config['experiment']['name']
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Experiment: {exp_name}")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config, num_workers=4)
    
    # Create model
    print("Creating model...")
    model = create_model_from_config(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    input_mode = input_mode_from_config(config)
    print(f"Input mode: {input_mode}")
    
    # Create loss
    criterion = create_loss_from_config(config)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    best_val_psnr = float('-inf')
    best_val_mse = float('inf')
    global_step = [0]  # Mutable for inner function
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, writer, global_step, input_mode
        )
        
        # Validate
        if (epoch + 1) % config['training'].get('val_interval', 5) == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch, writer, input_mode)
        else:
            val_loss = float('inf')
            val_metrics = None
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, output_dir / 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
            }, output_dir / 'best_loss.pt')
            print(f"  ✓ Saved best model!")

        if val_metrics is not None and val_metrics['psnr'] > best_val_psnr:
            best_val_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
            }, output_dir / 'best_psnr.pt')
            print(f"  ✓ Saved best PSNR model! ({best_val_psnr:.2f} dB)")

        if val_metrics is not None and val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
            }, output_dir / 'best_mse.pt')
            print(f"  ✓ Saved best MSE model! ({best_val_mse:.6f})")
        
        # Save latest
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'metrics': val_metrics,
            }, output_dir / 'latest.pt')

    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    if best_val_psnr > float('-inf'):
        print(f"Best val PSNR: {best_val_psnr:.2f} dB")
    if best_val_mse < float('inf'):
        print(f"Best val MSE: {best_val_mse:.6f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
