from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


class ReplayBuffer:
    """Cyclic replay buffer that stores transitions in numpy arrays and returns Torch tensors on demand."""
    def __init__(self, capacity: int, device: Optional[torch.device] = None) -> None:
        self.capacity = capacity
        self.device = device
        self._size = 0
        self._idx = 0

        # Main storage
        self.states: Optional[NDArray[np.float32]] = None
        self.next_states: Optional[NDArray[np.float32]] = None
        self.actions: Optional[NDArray[np.int64]] = None
        self.rewards: Optional[NDArray[np.float32]] = None
        self.dones: Optional[NDArray[np.bool_]] = None

        # Reusable CPU tensors (views on scratch arrays)
        self._cpu_states: Optional[Tensor] = None
        self._cpu_next_states: Optional[Tensor] = None
        self._cpu_actions: Optional[Tensor] = None
        self._cpu_rewards: Optional[Tensor] = None
        self._cpu_dones: Optional[Tensor] = None

        # Reusable device tensors (if device != cpu)
        self._dev_states: Optional[Tensor] = None
        self._dev_next_states: Optional[Tensor] = None
        self._dev_actions: Optional[Tensor] = None
        self._dev_rewards: Optional[Tensor] = None
        self._dev_dones: Optional[Tensor] = None
        self._dev: Optional[torch.device] = None

        # Scratch numpy arrays for samples
        self._tmp_states: Optional[np.ndarray] = None
        self._tmp_next_states: Optional[np.ndarray] = None
        self._tmp_actions: Optional[np.ndarray] = None
        self._tmp_rewards: Optional[np.ndarray] = None
        self._tmp_dones: Optional[np.ndarray] = None

    def _init_buffer(self, state_shape: Tuple[int, ...]) -> None:
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros_like(self.states, dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)

    def push(
        self,
        states: NDArray[np.float32],
        next_states: NDArray[np.float32],
        actions: NDArray[np.int64],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
    ) -> None:
        """
        Append a batch of transitions to the buffer (overwrites old data when full).

        Parameters
        ----------
        states: float32 array shaped [batch, *state_shape]
            Current observations.
        next_states: float32 array shaped [batch, *state_shape]
            Observations after executing actions.
        actions: int64 array shaped [batch] or [batch, ...]
            Actions taken.
        rewards: float32 array shaped [batch]
            Scalar rewards.
        dones: bool array shaped [batch]
            Episode done flags (True if next state is terminal).
        """
        added = states.shape[0]
        if added > self.capacity:
            raise ValueError(f"Batch size {added} exceeds capacity {self.capacity}")
        if self.states is None:
            self._init_buffer(states.shape[1:])

        assert self.states is not None and self.next_states is not None
        assert self.actions is not None and self.rewards is not None and self.dones is not None

        end = self._idx + added
        if end <= self.capacity:
            self.states[self._idx:end] = states
            self.next_states[self._idx:end] = next_states
            self.actions[self._idx:end] = actions
            self.rewards[self._idx:end] = rewards
            self.dones[self._idx:end] = dones
        else:
            first = self.capacity - self._idx
            second = added - first
            self.states[self._idx:] = states[:first]
            self.states[:second] = states[first:]
            self.next_states[self._idx:] = next_states[:first]
            self.next_states[:second] = next_states[first:]
            self.actions[self._idx:] = actions[:first]
            self.actions[:second] = actions[first:]
            self.rewards[self._idx:] = rewards[:first]
            self.rewards[:second] = rewards[first:]
            self.dones[self._idx:] = dones[:first]
            self.dones[:second] = dones[first:]

        self._idx = (self._idx + added) % self.capacity
        self._size = min(self._size + added, self.capacity)

    def _ensure_tmp(self, batch_size: int, device: Optional[torch.device]) -> None:
        assert self.states is not None and self.next_states is not None
        assert self.actions is not None and self.rewards is not None and self.dones is not None

        state_shape = (batch_size, *self.states.shape[1:])
        action_shape = (batch_size,) if self.actions.ndim == 1 else (batch_size, *self.actions.shape[1:])
        reward_shape = (batch_size,)
        done_shape = (batch_size,)

        if self._tmp_states is None or self._tmp_states.shape != state_shape:
            self._tmp_states = np.empty(state_shape, dtype=self.states.dtype)
            self._tmp_next_states = np.empty(state_shape, dtype=self.next_states.dtype)
            self._tmp_actions = np.empty(action_shape, dtype=self.actions.dtype)
            self._tmp_rewards = np.empty(reward_shape, dtype=self.rewards.dtype)
            self._tmp_dones = np.empty(done_shape, dtype=self.dones.dtype)

            self._cpu_states = torch.from_numpy(self._tmp_states)
            self._cpu_next_states = torch.from_numpy(self._tmp_next_states)
            self._cpu_actions = torch.from_numpy(self._tmp_actions)
            self._cpu_rewards = torch.from_numpy(self._tmp_rewards)
            self._cpu_dones = torch.from_numpy(self._tmp_dones)

        target = device or self.device
        if target is not None and target.type != "cpu":
            needs_alloc = (
                self._dev is None
                or self._dev != target
                or self._dev_states is None
                or self._dev_states.shape != state_shape
            )
            if needs_alloc:
                self._dev = target
                self._dev_states = torch.empty(state_shape, dtype=torch.float32, device=target)
                self._dev_next_states = torch.empty(state_shape, dtype=torch.float32, device=target)
                self._dev_actions = torch.empty(action_shape, dtype=self._cpu_actions.dtype, device=target)  # type: ignore[arg-type]
                self._dev_rewards = torch.empty(reward_shape, dtype=torch.float32, device=target)
                self._dev_dones = torch.empty(done_shape, dtype=torch.bool, device=target)

    def sample(
        self, batch_size: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Uniformly sample a batch of transitions.

        Returns
        -------
        states: Tensor
        actions: Tensor
        rewards: Tensor
        next_states: Tensor
        dones: Tensor (bool)

        Tensors are placed on `self.device` (if set) or stay on CPU. Raises ValueError
        if fewer than `batch_size` samples are stored.
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: have {self._size}, need {batch_size}")

        device = self.device
        idxs = np.random.randint(0, self._size, size=batch_size)

        self._ensure_tmp(batch_size, device)

        assert (
            self._tmp_states is not None
            and self._tmp_next_states is not None
            and self._tmp_actions is not None
            and self._tmp_rewards is not None
            and self._tmp_dones is not None
            and self._cpu_states is not None
            and self._cpu_next_states is not None
            and self._cpu_actions is not None
            and self._cpu_rewards is not None
            and self._cpu_dones is not None
        )

        # In-place gather into scratch arrays
        np.take(self.states, idxs, axis=0, out=self._tmp_states)           # type: ignore[arg-type]
        np.take(self.next_states, idxs, axis=0, out=self._tmp_next_states) # type: ignore[arg-type]
        np.take(self.actions, idxs, axis=0, out=self._tmp_actions)         # type: ignore[arg-type]
        np.take(self.rewards, idxs, axis=0, out=self._tmp_rewards)         # type: ignore[arg-type]
        np.take(self.dones, idxs, axis=0, out=self._tmp_dones)             # type: ignore[arg-type]

        if device is None or device.type == "cpu":
            states, next_states = self._cpu_states, self._cpu_next_states
            actions, rewards = self._cpu_actions, self._cpu_rewards
            dones = self._cpu_dones.bool()
        else:
            assert (
                self._dev_states is not None
                and self._dev_next_states is not None
                and self._dev_actions is not None
                and self._dev_rewards is not None
                and self._dev_dones is not None
            )
            self._dev_states.copy_(self._cpu_states, non_blocking=True)
            self._dev_next_states.copy_(self._cpu_next_states, non_blocking=True)
            self._dev_actions.copy_(self._cpu_actions, non_blocking=True)
            self._dev_rewards.copy_(self._cpu_rewards, non_blocking=True)
            self._dev_dones.copy_(self._cpu_dones.bool(), non_blocking=True)

            states, next_states = self._dev_states, self._dev_next_states
            actions, rewards, dones = self._dev_actions, self._dev_rewards, self._dev_dones

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self._size
