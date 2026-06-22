# Paper Results Reproduction Map

This file maps manuscript results to public configs, checkpoints, and commands.

## Main Model Evaluation

| Paper Item | Config | Checkpoint | Command | Expected Output |
| --- | --- | --- | --- | --- |
| Main structure-aware U-Net row | `configs/paper/main_unet.yaml` | `checkpoints/main_unet.pt` | `DATA_DIR=/path/to/simobs python scripts/evaluate_model.py --config configs/paper/main_unet.yaml --checkpoint checkpoints/main_unet.pt --output_dir outputs/eval_main_unet` | TBD |
| Main SwinIR row | `configs/paper/main_swinir.yaml` | `checkpoints/main_swinir.pt` | `DATA_DIR=/path/to/simobs python scripts/evaluate_model.py --config configs/paper/main_swinir.yaml --checkpoint checkpoints/main_swinir.pt --output_dir outputs/eval_main_swinir` | TBD |

## Ablations

| Paper Item | Config | Checkpoint | Command | Expected Output |
| --- | --- | --- | --- | --- |
| No-SD U-Net | `configs/paper/ablation_no_sd_unet.yaml` | `checkpoints/ablation_no_sd_unet.pt` | `DATA_DIR=/path/to/simobs python scripts/evaluate_model.py --config configs/paper/ablation_no_sd_unet.yaml --checkpoint checkpoints/ablation_no_sd_unet.pt --output_dir outputs/eval_no_sd_unet` | TBD |

## Architecture Comparison

| Paper Item | Config | Checkpoint | Command | Expected Output |
| --- | --- | --- | --- | --- |
| FiLM U-Net | `configs/paper/architecture_film_unet.yaml` | `checkpoints/architecture_film_unet.pt` | `DATA_DIR=/path/to/simobs python scripts/evaluate_model.py --config configs/paper/architecture_film_unet.yaml --checkpoint checkpoints/architecture_film_unet.pt --output_dir outputs/eval_architecture_film_unet` | TBD |
| DnCNN Dirty Only | `configs/paper/architecture_dncnn_dirty_only.yaml` | `checkpoints/architecture_dncnn_dirty_only.pt` | `DATA_DIR=/path/to/simobs python scripts/evaluate_model.py --config configs/paper/architecture_dncnn_dirty_only.yaml --checkpoint checkpoints/architecture_dncnn_dirty_only.pt --output_dir outputs/eval_architecture_dncnn_dirty_only` | TBD |
| DiT Patch Transformer | `configs/paper/architecture_dit_patch8.yaml` | `checkpoints/architecture_dit_patch8.pt` | `DATA_DIR=/path/to/simobs python scripts/evaluate_model.py --config configs/paper/architecture_dit_patch8.yaml --checkpoint checkpoints/architecture_dit_patch8.pt --output_dir outputs/eval_architecture_dit_patch8` | TBD |

## Traditional Reconstruction Baseline

| Paper Item | Config | Command | Expected Output |
| --- | --- | --- | --- |
| CASA CLEAN / Hogbom | `configs/paper/main_unet.yaml` | `CASA_WRAPPER=/path/to/casa python scripts/run_casa_tclean_baseline.py --config configs/paper/main_unet.yaml --methods hogbom --output-dir outputs/casa_clean_hogbom` | `outputs/casa_clean_hogbom/metrics_summary.json` |
| CASA T-CLEAN / Multi-scale | `configs/paper/main_unet.yaml` | `CASA_WRAPPER=/path/to/casa python scripts/run_casa_tclean_baseline.py --config configs/paper/main_unet.yaml --methods multiscale --output-dir outputs/casa_tclean_multiscale` | `outputs/casa_tclean_multiscale/metrics_summary.json` |

The CASA version, imaging cell size, image size, deconvolver, iteration limit, threshold, and exact sample list should be recorded with the submitted metric table.

## Release Notes

- Replace `TBD` entries after final checkpoint upload and checksum verification.
- Keep this file synchronized with the manuscript tables before submission.
