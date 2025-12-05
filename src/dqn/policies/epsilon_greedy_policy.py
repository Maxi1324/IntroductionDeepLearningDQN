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

    def __init__(self, epsilon: float = 0.1, seed: Optional[int] = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        super().__init__(seed=seed)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def policy(self, q_values: np.ndarray) -> np.ndarray:
        q = np.asarray(q_values, dtype=np.float32, order="C")
        if q.ndim != 2:
            raise ValueError(f"Expected q_values shape (batch, actions), got {q.shape}")

        batch, num_actions = q.shape
        actions = np.argmax(q, axis=1).astype(np.int32, copy=False)

        if self.epsilon == 0.0 or num_actions == 1:
            return actions

        explore_mask = self.rng.random(batch) < self.epsilon
        explore_count = int(explore_mask.sum())
        if explore_count:
            actions[explore_mask] = self.rng.integers(
                low=0, high=num_actions, size=explore_count, dtype=np.int32
            )

        return actions
