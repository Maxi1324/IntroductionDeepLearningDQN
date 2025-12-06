from __future__ import annotations

from typing import Optional

import numpy as np

from dqn.policies.policy_base import Policy


class GreedyPolicy(Policy):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)

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
        actions = np.argmax(q, axis=1).astype(np.int32, copy=False)
        return actions
