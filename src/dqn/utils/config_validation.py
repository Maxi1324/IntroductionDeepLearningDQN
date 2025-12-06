from __future__ import annotations

from typing import Any, Optional, Tuple, cast

import gymnasium as gym
from gymnasium import spaces
import torch

from dqn.utils.parameter import Parameter
from dqn.data_collection import DataCollect
from dqn.optimizer import Optimizer
from dqn.policies.policy_base import Policy


def validate_configuration(
    param: Parameter,
    data_collection: DataCollect,
    optimizer: Optimizer,
    policy: Policy,
    device: torch.device,
) -> None:
    """
    Validate that parameters, environment, and network dimensions are consistent.
    Exits the program with a short English message on failure.
    """
    p = param
    states_np = data_collection._states

    def fail(msg: str) -> None:
        print(f"Configuration error: {msg}")
        raise SystemExit(1)

    if not isinstance(p.Network, str) or not p.Network:
        fail("Network must be a non-empty string key.")
    if p.learningRate <= 0:
        fail("learningRate must be > 0.")
    if not (0.0 <= p.discount <= 1.0):
        fail("discount must be in [0, 1].")
    if p.batchSize <= 0:
        fail("batchSize must be > 0.")
    if p.replayBufferSize < p.batchSize:
        fail("replayBufferSize must be >= batchSize.")
    if p.collectSteps <= 0:
        fail("collectSteps must be > 0.")
    if p.prewarm < p.batchSize:
        fail("prewarm must be >= batchSize for sampling.")
    if not isinstance(p.Device, str) or not p.Device:
        fail("Device must be a non-empty string.")
    if p.envsCount <= 0:
        fail("envsCount must be > 0.")
    if not p.env:
        fail("env id must be set.")
    if not isinstance(p.Policy, str) or not p.Policy:
        fail("Policy must be a non-empty string key.")
    if p.Policy.lower() == "epsilon_greedy":
        if p.epsilon is None or p.epsilonEnd is None:
            fail("epsilon and epsilonEnd must be set for epsilon_greedy policy.")
    if p.Policy.lower() == "boltzmann":
        if p.epsilon is None or p.epsilonEnd is None:
            fail("temperature start/end (epsilon/epsilonEnd) must be set for boltzmann policy.")
    if policy.online_predictor is None:
        fail("Policy has no OnlinePredictor attached.")

    env_single: Optional[gym.Env[Any, Any]] = None
    try:
        env_single = gym.make(p.env)
        obs_reset = env_single.reset()
        if isinstance(obs_reset, tuple):
            _obs, _info = obs_reset 
        else:
            _obs = obs_reset

        action_space = env_single.action_space
        obs_space = env_single.observation_space
        if not isinstance(action_space, spaces.Discrete):
            fail("Action space must be Discrete.")
        action_space = cast(spaces.Discrete, action_space)
        if not isinstance(obs_space, spaces.Box):
            fail("Observation space must be a Box.")
        obs_space = cast(spaces.Box, obs_space)
        if obs_space.shape is None:
            fail("Observation space must define a shape.")

        shape = obs_space.shape
        expected_shape = (p.envsCount, *shape)
        if states_np.shape != expected_shape:
            fail(f"Env reset states shape {states_np.shape} does not match expected {expected_shape}.")

        states_t = torch.as_tensor(states_np, device=device, dtype=torch.float32)
        with torch.no_grad():
            logits = optimizer.online_network(states_t)
        if logits.ndim < 2:
            fail("Network output must be at least 2D [batch, actions].")
        if logits.shape[0] != p.envsCount:
            fail(f"Network batch dimension {logits.shape[0]} != envsCount {p.envsCount}.")
        if logits.shape[-1] != action_space.n:
            fail(f"Network action dimension {logits.shape[-1]} != env action space {action_space.n}.")
    except Exception as e:
        fail(f"Environment/network validation failed: {e}")
    finally:
        if env_single is not None:
            env_single.close()
