from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from torch import nn

# Export individual models.
from .feedforward_small import FeedForwardSmall
from .feedforward_medium import FeedForwardMedium
from .conv_small import ConvSmall
from .conv_nature import ConvNature

MODEL_REGISTRY: Dict[str, Tuple[Type[nn.Module], Dict[str, Any]]] = {
    "mlp_small": (FeedForwardSmall, {"state_dim": 4, "action_dim": 2}),
    "mlp_medium": (FeedForwardMedium, {"state_dim": 4, "action_dim": 2}),
    "conv_small": (ConvSmall, {"in_channels": 4, "action_dim": 2, "input_height": 84, "input_width": 84}),
    "conv_nature": (ConvNature, {"in_channels": 4, "action_dim": 2, "input_height": 84, "input_width": 84}),
}


def get_model(name: str, **kwargs: Any) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_cls, defaults = MODEL_REGISTRY[key]
    cfg = {**defaults, **kwargs}
    return model_cls(**cfg)

