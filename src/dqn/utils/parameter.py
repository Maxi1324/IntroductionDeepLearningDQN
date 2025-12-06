from dataclasses import dataclass
from typing import Any, Optional

from dataclasses import dataclass


@dataclass
class Parameter:
    Device: str
    Network: str
    Policy: str
    env: str
    envsCount: int
    learningRate: float
    discount: float
    collectSteps: int
    replayBufferSize: int
    epochs: int
    optimizationPerEpoch: int
    batchSize: int
    prewarm: int
    name: str
    seed: Optional[int] = None
    epsilon: Optional[float] = None
    epsilonEnd: Optional[float] = None
    targetUpdateEvery: int = 1
    experiment: Optional[str] = None
