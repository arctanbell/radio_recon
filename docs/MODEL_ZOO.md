# Model Zoo

Large checkpoints should be hosted outside Git. Fill the URL and checksum fields once the final weights are uploaded.

| Name | Paper Role | Config | Weight URL | SHA256 | Expected PSNR | Expected SSIM | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main structure-aware U-Net | Main supervised U-Net run with structure-aware loss | `configs/paper/main_unet.yaml` | TBD | TBD | TBD | TBD | pending |
| Main SwinIR | Transformer comparison | `configs/paper/main_swinir.yaml` | TBD | TBD | TBD | TBD | pending |
| No-SD U-Net | Single-dish input ablation | `configs/paper/ablation_no_sd_unet.yaml` | TBD | TBD | TBD | TBD | pending |
| FiLM U-Net | Architecture comparison | `configs/paper/architecture_film_unet.yaml` | TBD | TBD | TBD | TBD | pending |
| DnCNN Dirty Only | Generic baseline | `configs/paper/architecture_dncnn_dirty_only.yaml` | TBD | TBD | TBD | TBD | pending |
| DiT Patch Transformer | Architecture comparison | `configs/paper/architecture_dit_patch8.yaml` | TBD | TBD | TBD | TBD | pending |

## Checksum Command

```bash
shasum -a 256 checkpoints/main_unet.pt
```

## Evaluation Template

```bash
DATA_DIR=/path/to/simobs python scripts/evaluate_model.py \
  --config configs/paper/main_unet.yaml \
  --checkpoint checkpoints/main_unet.pt \
  --output_dir outputs/eval_main_unet
```
