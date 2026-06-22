"""DnCNN model for generic image restoration baseline."""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """Standard DnCNN denoiser with optional residual learning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 17,
        features: int = 64,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual

        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            return x - out
        return out


def create_dncnn(config: dict) -> DnCNN:
    model_config = config.get("model", {})
    return DnCNN(
        in_channels=model_config.get("in_channels", 1),
        out_channels=model_config.get("out_channels", 1),
        depth=model_config.get("depth", 17),
        features=model_config.get("features", 64),
        residual=model_config.get("residual", True),
    )
