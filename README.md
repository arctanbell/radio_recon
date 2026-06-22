# radio_recon

Official code and reproducibility materials for the paper **Structure-Aware Multimodal Learning for Radio Interferometric Image Reconstruction**.

This repository provides the model implementations, configuration files, evaluation scripts, figure data, and manuscript assets needed to reproduce the main experiments.

## Repository Layout

```text
radio_recon/
├── configs/paper/          # Stable configs for paper-facing runs
├── docs/                   # Data, checkpoint, and reproducibility notes
├── paper/                  # Manuscript source, figures, and figure data
├── radio_recon/            # Python package for datasets, models, losses, metrics
├── scripts/                # Training, evaluation, and benchmark helpers
└── tests/                  # Lightweight sanity checks
```

## Installation

```bash
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches the target server first, then install the remaining requirements.

For editable package development:

```bash
python -m pip install -e .[torch,dev]
```

## Data

The main experiments use simulated FITS samples with one directory per sample. Each sample directory is expected to contain:

- `*_dirty.psf.fits`
- `*_dirty.image.pbcor.fits` or `*_dirty.image.fits`
- `*_rg_fast.fits`
- `*_rg_dirty.fits`

Public commands use `data/simobs` as a placeholder. Override it with `DATA_DIR` or edit `data.data_dir` in the config:

```bash
DATA_DIR=/path/to/simobs python scripts/evaluate_model.py \
  --config configs/paper/main_unet.yaml \
  --checkpoint checkpoints/main_unet.pt \
  --output_dir outputs/eval_main_unet
```

See [docs/DATA.md](docs/DATA.md) for the dataset contract and release checklist.

## Checkpoints

Large model weights are not stored in Git. Download links, checksums, configs, and expected metrics are tracked in [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

## Minimal Checks

```bash
python tests/test_imports.py
./test_imports.sh
python -m compileall radio_recon scripts tests/test_imports.py
```

`tests/test_imports.py` skips torch-dependent modules when PyTorch is not installed. In the final release environment, install `requirements.txt` and rerun the check to exercise the model, dataset, and loss imports.

## Paper Reproduction

The paper-facing configs are:

- `configs/paper/main_unet.yaml`
- `configs/paper/main_swinir.yaml`
- `configs/paper/ablation_no_sd_unet.yaml`
- `configs/paper/eval_swinir_fulltest.yaml`
- `configs/paper/architecture_film_unet.yaml`
- `configs/paper/architecture_dncnn_dirty_only.yaml`
- `configs/paper/architecture_dit_patch8.yaml`

Result-to-command mapping is tracked in [docs/PAPER_RESULTS.md](docs/PAPER_RESULTS.md). Entries marked `TBD` must be filled after the final checkpoint URLs and checksums are frozen.

## Release Status

The paper-facing source package, configs, model factory, smoke checks, and release manifest are in place. Final release still needs public checkpoint URLs, checksums, and metric reproduction rows filled in after the weights are uploaded.

Local and remote smoke evidence is tracked in [docs/RELEASE_VERIFICATION.md](docs/RELEASE_VERIFICATION.md).
