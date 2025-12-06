from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Optional

from torch import nn

# Export individual models.
from .feedforward_small import FeedForwardSmall
from .feedforward_medium import FeedForwardMedium
from .feedforward_large import FeedForwardLarge

MODEL_REGISTRY: Dict[str, Tuple[Type[nn.Module], Dict[str, Any]]] = {
    "mlp_small": (FeedForwardSmall, {"state_dim": 4, "action_dim": 2}),
    "mlp_medium": (FeedForwardMedium, {"state_dim": 4, "action_dim": 2}),
    "mlp_large": (FeedForwardLarge, {"state_dim": 4, "action_dim": 2}),
}


def get_model(
    name: str,
    *,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    **kwargs: Any,
) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_cls, defaults = MODEL_REGISTRY[key]
    cfg: Dict[str, Any] = {**defaults}
    if state_dim is not None and "state_dim" in defaults:
        cfg["state_dim"] = state_dim
    if action_dim is not None and "action_dim" in defaults:
        cfg["action_dim"] = action_dim
    cfg.update(kwargs)
    return model_cls(**cfg)
