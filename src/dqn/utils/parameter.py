from dataclasses import dataclass
from typing import Type, Union, Callable

import torch
from torch import nn

from dqn.policies.policy_base import Policy


@dataclass
class Parameter:
    Device: torch.device
    Network: nn.Module
    Policy: Policy
    env: str
    envsCount: int
    learningRate: float
    discount: float
    collectSteps: int
    replayBufferSize: int
    epochs: int
    optimizationPerEpoch: int
    batchSize: int
    prewarm:int
