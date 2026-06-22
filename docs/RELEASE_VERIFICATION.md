# Release Verification

This file records smoke-level evidence for the public release file set listed in `docs/PUBLIC_RELEASE_FILES.txt`.

## Local Release-Tree Checks

The release tarball was built from the 71 paths in `docs/PUBLIC_RELEASE_FILES.txt`.

Observed status:

- Every manifest path exists in the working tree.
- The tarball file list contains no Python caches, AppleDouble `._*` files, internal sync folders, internal workflow folders, or document scratch folders.
- Static compilation passed for `radio_recon`, `scripts`, and `tests/test_imports.py`.
- All 7 `configs/paper/*.yaml` files loaded successfully in a local Python environment with `PyYAML` available.
- Full dependency import checks were deferred to the remote validation environment because local Python installations were incomplete or architecture-mismatched for this project.

## Remote Checks

Environment:

- Python: `3.13.11`
- PyTorch: `2.9.1+cu128`
- CUDA available: `True`
- CUDA device count: `8`

Observed status:

- Public release tarball extracted cleanly without AppleDouble files.
- Internal absolute-path scan returned no matches.
- `tests/test_imports.py` passed, including torch-dependent modules and all model implementations.
- `python -m compileall radio_recon scripts tests/test_imports.py` passed.
- CLI help passed for `scripts/train_simple.py`, `scripts/evaluate_model.py`, `scripts/run_casa_tclean_baseline.py`, `scripts/plot_paper_figures.py`, `scripts/prepare_paper_figure_data.py`, and `scripts/verify_release_smoke.py`.
- Synthetic forward/loss smoke passed for all 7 paper configs:
  - `configs/paper/ablation_no_sd_unet.yaml`
  - `configs/paper/architecture_dit_patch8.yaml`
  - `configs/paper/architecture_dncnn_dirty_only.yaml`
  - `configs/paper/architecture_film_unet.yaml`
  - `configs/paper/eval_swinir_fulltest.yaml`
  - `configs/paper/main_swinir.yaml`
  - `configs/paper/main_unet.yaml`
- FITS dataset smoke passed with 12,726 detected samples and first sample `CAR_B01_MP_C0400_run260105`.
- Real FITS forward smoke passed for:
  - `main_unet.yaml`, `input_mode=all`, output shape `(1, 1, 192, 192)`
  - `ablation_no_sd_unet.yaml`, `input_mode=psf_dirty`, output shape `(1, 1, 192, 192)`
  - `architecture_dncnn_dirty_only.yaml`, `input_mode=dirty_only`, output shape `(1, 1, 192, 192)`
  - `architecture_film_unet.yaml`, `input_mode=all`, output shape `(1, 1, 192, 192)`
  - `main_swinir.yaml`, `input_mode=all`, output shape `(1, 1, 192, 192)`
  - `architecture_dit_patch8.yaml`, `input_mode=all`, output shape `(1, 1, 192, 192)`

## Known Non-Code Release Gates

- Public checkpoint URLs are still `TBD`.
- SHA256 checksums are still `TBD`.
- Exact metric reproduction rows in `docs/PAPER_RESULTS.md` remain `TBD` until the released checkpoints are frozen.
