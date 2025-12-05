from __future__ import annotations

from typing import Optional

import numpy as np


class BoltzmannPolicy:
    """
    Samples actions according to a Boltzmann/softmax distribution over Q-values.
    Expects Q-values already computed (e.g., by OnlinePredictor) and returns int32 actions.
    """

    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

    def policy(self, q_values: np.ndarray) -> np.ndarray:
        q = np.asarray(q_values, dtype=np.float32, order="C")
        if q.ndim != 2:
            raise ValueError(f"Expected q_values shape (batch, actions), got {q.shape}")

        # numerically stable softmax
        logits = q / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float64)
        probs = exp / exp.sum(axis=1, keepdims=True)

        actions = np.empty(q.shape[0], dtype=np.int32)
        for i, p in enumerate(probs):
            actions[i] = self.rng.choice(p.size, p=p)
        return actions
