from __future__ import annotations

from typing import Optional

import numpy as np

from dqn.policies.policy_base import Policy


class BoltzmannPolicy(Policy):
    def __init__(self, temperature: float = 1.0, temperature_end: float = 0.1, seed: Optional[int] = None) -> None:
        if temperature <= 0 or temperature_end <= 0:
            raise ValueError("temperature values must be > 0")
        super().__init__(seed=seed)
        self.temperature_start = float(temperature)
        self.temperature_end = float(temperature_end)
        self.rng = np.random.default_rng(seed)

    def _current_temp(
        self,
        current_step: Optional[int],
        max_steps: Optional[int],
        current_epoch: Optional[int],
        max_epochs: Optional[int],
    ) -> float:
        if current_step is not None and max_steps is not None and max_steps > 0:
            frac = min(max(current_step, 0), max_steps) / float(max_steps)
            return self.temperature_start + frac * (self.temperature_end - self.temperature_start)
        if current_epoch is None or max_epochs is None or max_epochs <= 1:
            return self.temperature_start
        frac = min(max(current_epoch, 0), max_epochs - 1) / float(max_epochs - 1)
        return self.temperature_start + frac * (self.temperature_end - self.temperature_start)

    def policy(
        self,
        q_values: np.ndarray,
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
        current_epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> np.ndarray:
        q = np.asarray(q_values, dtype=np.float32, order="C")
        if q.ndim != 2:
            raise ValueError(f"Expected q_values shape (batch, actions), got {q.shape}")

        temperature = self._current_temp(current_step, max_steps, current_epoch, max_epochs)
        logits = q / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float64)
        probs = exp / exp.sum(axis=1, keepdims=True)

        actions = np.empty(q.shape[0], dtype=np.int32)
        for i, p in enumerate(probs):
            actions[i] = self.rng.choice(p.size, p=p)
        return actions
