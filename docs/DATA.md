# Data

The `radio_recon` release repository expects a FITS sample directory layout compatible with the manuscript experiments.

## Sample Layout

Each sample is stored in its own subdirectory:

```text
simobs/
└── sample_name/
    ├── sample_name_dirty.psf.fits
    ├── sample_name_dirty.image.pbcor.fits
    ├── sample_name_rg_fast.fits
    └── sample_name_rg_dirty.fits
```

If `*_dirty.image.pbcor.fits` is unavailable, the loader falls back to `*_dirty.image.fits`.

## Channel Order

The model condition tensor uses this channel order:

1. PSF
2. Dirty image
3. Single-dish / fast map

The target is `*_rg_dirty.fits`.

## Normalization

The main paper configs use per-sample, per-modality robust percentile normalization:

- PSF: sum normalization followed by percentile min-max scaling.
- Dirty image: independent 1st/99th percentile scaling.
- Single-dish map: independent 1st/99th percentile scaling.
- Target: independent 1st/99th percentile scaling.

Historical ablation normalizers are implemented for figure reproduction: `ratio_preserving`, `arcsinh`, and `adaptive`.

## Public Release Checklist

- Document whether the full simulated dataset can be publicly redistributed.
- If the full dataset is too large, publish a manifest and a small smoke-test subset.
- Provide the exact train/validation/test split seed and ratios.
- Confirm that all paper-facing configs avoid local absolute paths.
