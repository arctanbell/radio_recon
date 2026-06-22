"""Minimal import checks for the paper-release package."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


BASE_MODULES = [
    "radio_recon",
    "radio_recon.data.utils",
    "radio_recon.data.normalization",
    "radio_recon.evaluation.metrics",
    "radio_recon.utils.config",
]

TORCH_MODULES = [
    "radio_recon.data.dataset",
    "radio_recon.losses.combined",
    "radio_recon.models.conditional_unet_attention",
    "radio_recon.models.conditional_unet_film",
    "radio_recon.models.dit_radio",
    "radio_recon.models.dncnn",
    "radio_recon.models.simple_unet",
    "radio_recon.models.swinir_radio",
    "radio_recon.models.model_factory",
    "radio_recon.utils.input_mode",
]


def import_modules(module_names):
    for module_name in module_names:
        importlib.import_module(module_name)
        print(f"ok: {module_name}")


def main() -> None:
    import_modules(BASE_MODULES)
    if importlib.util.find_spec("torch") is None:
        print("skip: torch-dependent modules because torch is not installed")
        return
    import_modules(TORCH_MODULES)


if __name__ == "__main__":
    main()
