from __future__ import annotations

import warnings
from typing import Sequence

import torch
import torch.nn as nn


class ConvNature(nn.Module):
    """Deeper conv net mirroring the Nature DQN backbone."""

    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        input_height: int = 84,
        input_width: int = 84,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out = self._conv_output_dim(in_channels, input_height, input_width)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def _conv_output_dim(self, in_channels: int, h: int, w: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            out = self.conv(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.head(self.conv(x))
