from __future__ import annotations

import torch.nn as nn

from radio_recon.models.conditional_unet_attention import create_conditional_unet_attention
from radio_recon.models.conditional_unet_film import create_conditional_unet_film
from radio_recon.models.dit_radio import create_dit_radio
from radio_recon.models.dncnn import create_dncnn
from radio_recon.models.simple_unet import create_simple_unet
from radio_recon.models.swinir_radio import create_swinir_radio


def create_model_from_config(config: dict) -> nn.Module:
    model_type = config.get('model', {}).get('type', 'simple_unet')

    if model_type in {'simple_unet', 'unet'}:
        return create_simple_unet(config)
    if model_type == 'swinir':
        return create_swinir_radio(config)
    if model_type == 'conditional_unet_film':
        return create_conditional_unet_film(config)
    if model_type == 'conditional_unet_attention':
        return create_conditional_unet_attention(config)
    if model_type == 'dncnn':
        return create_dncnn(config)
    if model_type == 'dit':
        return create_dit_radio(config)

    raise ValueError(f"Unknown model type: {model_type}")
