from __future__ import annotations

from typing import Optional

import numpy as np

from dqn.policies.policy_base import Policy


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy policy over provided Q-values. Uses greedy argmax with
    probability (1 - epsilon) and a random action with probability epsilon.
    Expects Q-values already computed and returns int32 actions.
    """

    def __init__(self, epsilon: float = 0.1, epsilon_end: float = 0.05, seed: Optional[int] = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 <= epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be in [0, 1]")
        super().__init__(seed=seed)
        self.epsilon_start = float(epsilon)
        self.epsilon_end = float(epsilon_end)
        self.rng = np.random.default_rng(seed)

    def _current_epsilon(self, current_epoch: Optional[int], max_epochs: Optional[int]) -> float:
        if current_epoch is None or max_epochs is None or max_epochs <= 1:
            return self.epsilon_start
        frac = min(max(current_epoch, 0), max_epochs - 1) / float(max_epochs - 1)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def policy(self, q_values: np.ndarray, current_epoch: Optional[int] = None, max_epochs: Optional[int] = None) -> np.ndarray:
        q = np.asarray(q_values, dtype=np.float32, order="C")
        if q.ndim != 2:
            raise ValueError(f"Expected q_values shape (batch, actions), got {q.shape}")

        batch, num_actions = q.shape
        actions = np.argmax(q, axis=1).astype(np.int32, copy=False)

        eps = self._current_epsilon(current_epoch, max_epochs)

        if eps <= 0.0 or num_actions == 1:
            return actions

        explore_mask = self.rng.random(batch) < eps
        explore_count = int(explore_mask.sum())
        if explore_count:
            actions[explore_mask] = self.rng.integers(
                low=0, high=num_actions, size=explore_count, dtype=np.int32
            )

        return actions
