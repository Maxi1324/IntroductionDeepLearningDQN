from __future__ import annotations
from typing import Callable

import numpy as np
import torch
from torch import Tensor


class OnlinePredictor:
    """
    Runs the current online network on a batch of states and returns raw Q-values
    (numpy float32) so downstream policies like greedy or Boltzmann can decide.
    """

    def __init__(self, get_online_network: Callable[[], torch.nn.Module], device: torch.device) -> None:
        self.get_online_network = get_online_network
        self.device = device
        self._states_t: Tensor | None = None
        self._q_t: Tensor | None = None
        self._batch_shape: tuple[int, ...] | None = None

    def getLogits(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32, order="C")

        if self._batch_shape != states.shape:
            self._batch_shape = states.shape
            self._states_t = torch.empty(states.shape, dtype=torch.float32, device=self.device)

        assert self._states_t is not None

        self._states_t.copy_(torch.from_numpy(states), non_blocking=True)

        net = self.get_online_network()

        with torch.no_grad():
            q: Tensor = net(self._states_t)
            if self._q_t is None or self._q_t.shape != q.shape:
                self._q_t = torch.empty_like(q)
            self._q_t.copy_(q)

        out = self._q_t.cpu().numpy().astype(np.float32, copy=False)
        return out
