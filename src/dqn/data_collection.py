from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray

from dqn.replay_buffer import ReplayBuffer
from dqn.policies import Policy

class DataCollect:
    def __init__(
        self,
        env_id: str,
        env_count: int,
        replay_buffer: ReplayBuffer,
        policy: Policy
    ) -> None:
        self.env: VectorEnv = gym.make_vec(
            env_id,
            num_envs=env_count,
            vectorization_mode="sync",
          #  max_episode_steps=10_000,
        )
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.env_count = env_count

        states, _ = self.env.reset()
        self._states: NDArray[np.float32] = np.asarray(states, dtype=np.float32)
        self._ep_len: NDArray[np.int32] = np.zeros(env_count, dtype=np.int32)
        self._ep_return: NDArray[np.float32] = np.zeros(env_count, dtype=np.float32)

    def collect(
        self,
        num_steps: int,
        current_step: int | None = None,
        max_steps: int | None = None,
        current_epoch: int | None = None,
        max_epochs: int | None = None,
    ) -> tuple[float | None, float | None]:
        completed_lengths: list[int] = []
        completed_returns: list[float] = []

        for t in range(num_steps):
            states = self._states
            step_val: Optional[int] = None
            if current_step is not None:
                step_val = current_step + t * self.env_count
                if max_steps is not None:
                    step_val = min(step_val, max_steps)

            actions = self.policy.getAction(
                states,
                current_step=step_val,
                max_steps=max_steps,
                current_epoch=current_epoch,
                max_epochs=max_epochs,
            )
            next_states, rewards, terminated, truncated, _ = self.env.step(actions)

            next_states = next_states.astype(np.float32)
            rewards = rewards.astype(np.float32)
            dones = np.logical_or(terminated, truncated)

            # Track episode stats
            self._ep_len += 1
            self._ep_return += rewards
            if dones.any():
                done_mask = dones
                completed_lengths.extend(self._ep_len[done_mask].tolist())
                completed_returns.extend(self._ep_return[done_mask].tolist())
                self._ep_len[done_mask] = 0
                self._ep_return[done_mask] = 0.0

            assert isinstance(states, np.ndarray) and states.dtype == np.float32
            assert isinstance(next_states, np.ndarray) and next_states.dtype == np.float32
            assert isinstance(actions, np.ndarray) and actions.dtype == np.int32
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

        avg_len = float(np.mean(completed_lengths)) if completed_lengths else None
        avg_ret = float(np.mean(completed_returns)) if completed_returns else None
        return avg_len, avg_ret
