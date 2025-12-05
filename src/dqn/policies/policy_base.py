from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dqn.online_predictor import OnlinePredictor


class Policy(ABC):
    """
    Base class for policies that consume Q-values and return int32 actions.
    Subclasses should implement `policy`.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.online_predictor: Optional[OnlinePredictor] = None

    @abstractmethod
    def policy(self, q_values: np.ndarray) -> NDArray[np.int32]:
        ...

    def __call__(self, q_values: np.ndarray) -> NDArray[np.int32]:
        return self.policy(q_values)

    def setOnlinePredictor(self, online_predictor: OnlinePredictor) -> None:
        self.online_predictor = online_predictor

    def getAction(self, states: np.ndarray) -> NDArray[np.int32]:
        """
        Convenience: compute Q-values via the attached OnlinePredictor and return actions.
        """
        assert self.online_predictor is not None, "online_predictor is not set on this Policy instance."
        q_values = self.online_predictor.getLogits(states)
        return self.policy(q_values)
