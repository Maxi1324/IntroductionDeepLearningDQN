from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class FeedForwardLarge(nn.Module):
    """Wider/deeper MLP for low-dimensional state vectors."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (512, 256, 128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        return self.net(x)
