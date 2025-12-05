from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray

from infrastructure.replay_buffer import ReplayBuffer


class DataCollect:
    """
    Data collector that runs a Gymnasium vector environment with a provided policy
    and writes transitions into a replay buffer.
    """

    def __init__(
        self,
        env_id: str,
        env_count: int,
        replay_buffer: ReplayBuffer,
        policy_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.env: VectorEnv = gym.make_vec(
            env_id,
            num_envs=env_count,
            vectorization_mode="sync"
        )
        self.replay_buffer = replay_buffer
        self.policy_fn = policy_fn
        self.env_count = env_count

        states, _ = self.env.reset()
        self._states: NDArray[np.float32] = np.asarray(states, dtype=np.float32)

    def collect(self, num_steps: int) -> None:
        for _ in range(num_steps):
            states = self._states

            actions = self.policy_fn(states)
            next_states, rewards, terminated, truncated, _ = self.env.step(actions)

            next_states = next_states.astype(np.float32)
            rewards = rewards.astype(np.float32)
            dones = np.logical_or(terminated, truncated)

            assert isinstance(states, np.ndarray) and states.dtype == np.float32
            assert isinstance(next_states, np.ndarray) and next_states.dtype == np.float32
            assert isinstance(actions, np.ndarray) and actions.dtype == np.int64
            assert isinstance(rewards, np.ndarray) and rewards.dtype == np.float32
            assert isinstance(dones, np.ndarray) and dones.dtype == np.bool_

            self.replay_buffer.push(
                states=states,
                next_states=next_states,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )

            self._states = next_states
