from __future__ import annotations

import torch
from dqn.parameter import Parameter
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from dqn.data_collection import DataCollect
from dqn.replay_buffer import ReplayBuffer
from dqn.optimizer import Optimizer
from dqn.online_predictor import OnlinePredictor
from dqn.utils.config_validation import validate_configuration
from dqn.utils.training_logger import TrainingLogger
from dqn.policies import get_policy
from dqn.training_finalizer import TrainingFinalizer
import mlflow
import tempfile
from pathlib import Path
import uuid


class Agent:
    def __init__(self, param: Parameter, consoleLogging=True) -> None:
        self.param = param
        self.device = torch.device(param.Device)
        self._epoch_avg_lengths: list[float] = []

        policy_kwargs = {}
        if param.Policy.lower() == "epsilon_greedy" and param.epsilon is not None:
            policy_kwargs["epsilon"] = param.epsilon
            if param.epsilonEnd is not None:
                policy_kwargs["epsilon_end"] = param.epsilonEnd
        if param.seed is not None:
            policy_kwargs["seed"] = param.seed
        if param.Policy.lower() == "boltzmann" and param.epsilon is not None:
            policy_kwargs["temperature"] = param.epsilon
            if param.epsilonEnd is not None:
                policy_kwargs["temperature_end"] = param.epsilonEnd
        policy_instance = get_policy(param.Policy, **policy_kwargs)
        self.replayBuffer: ReplayBuffer = ReplayBuffer(param.replayBufferSize, self.device)
        self.dataCollection: DataCollect = DataCollect(param.env, param.envsCount, self.replayBuffer, policy_instance)
        obs_shape = self.dataCollection._states.shape[1:]
        state_dim = int(np.prod(obs_shape))
        action_space = None
        if hasattr(self.dataCollection.env, "single_action_space"):
            action_space = getattr(self.dataCollection.env, "single_action_space")
        elif hasattr(self.dataCollection.env, "action_space"):
            action_space = getattr(self.dataCollection.env, "action_space")
        action_dim = None
        if isinstance(action_space, spaces.Discrete):
            action_dim = int(action_space.n)
        if action_dim is None or action_dim <= 0:
            raise ValueError("Could not infer action_dim from environment action_space.")
        self.optimizer = Optimizer(param, state_dim=state_dim, action_dim=action_dim)
        onlinePredictor = OnlinePredictor(self.optimizer.getOnlineNetwork, self.device)
        policy_instance.setOnlinePredictor(onlinePredictor)
        self.policy = policy_instance
        validate_configuration(self.param, self.dataCollection, self.optimizer, self.policy, self.device)
        self.logger = TrainingLogger(
            run_name=getattr(param, "name", "dqn-run"),
            experiment_name=getattr(param, "experiment", None),
            consoleLogging=consoleLogging,
        )

    def evaluate(self, episodes: int = 20, max_episode_steps: int = 10_000, env_steps_total: int | None = None) -> float:
        eval_env = gym.make(self.param.env, max_episode_steps=max_episode_steps)
        lengths: list[int] = []
        current_step = env_steps_total if env_steps_total is not None else None

        for _ in range(episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_len = 0
            while not done:
                obs_arr = np.asarray(obs, dtype=np.float32, order="C")
                if obs_arr.ndim == 1:
                    obs_arr = obs_arr[None, ...]
                actions = self.policy.getAction(
                    obs_arr,
                    current_step=current_step,
                    max_steps=env_steps_total,
                    current_epoch=self.param.epochs,
                    max_epochs=self.param.epochs,
                )
                action = int(actions[0])
                obs, _reward, terminated, truncated, _ = eval_env.step(action)
                ep_len += 1
                if terminated or truncated:
                    done = True
                    lengths.append(ep_len)
        eval_env.close()
        return float(np.mean(lengths)) if lengths else 0.0

    def train(self) -> None:
        with self.logger.start_run():
            self.logger.log_all_params(self.param)
            env_steps_total = (self.param.prewarm + self.param.collectSteps * self.param.epochs) * self.param.envsCount
            env_step = 0
            self.dataCollection.collect(
                self.param.prewarm,
                current_step=env_step,
                max_steps=env_steps_total,
            )
            env_step += self.param.prewarm * self.param.envsCount
            global_step = 0
            for epoch in range(0, self.param.epochs):
                self.logger.start_epoch(epoch, self.param.epochs)
                avg_len, avg_ret = self.dataCollection.collect(
                    self.param.collectSteps,
                    current_step=env_step,
                    max_steps=env_steps_total,
                    current_epoch=epoch,
                    max_epochs=self.param.epochs,
                )
                env_step += self.param.collectSteps * self.param.envsCount

                if avg_len is not None:
                    self._epoch_avg_lengths.append(float(avg_len))
                    if len(self._epoch_avg_lengths) > 100:
                        self._epoch_avg_lengths = self._epoch_avg_lengths[-100:]

                for opt_idx in range(0, self.param.optimizationPerEpoch):
                    states, actions, rewards, next_states, dones = self.replayBuffer.sample(self.param.batchSize)
                    loss = self.optimizer.optimize(states, actions, rewards, next_states, dones)
                    self.logger.log_loss(loss, step=global_step)
                    global_step += 1
                    if (opt_idx + 1) % getattr(self.param, "targetUpdateEvery", 1) == 0:
                        self.optimizer.updateTargetNetwork()

                self.logger.log_episode_stats(avg_len, avg_ret)
                self.logger.end_epoch(global_step)

            finalizer = TrainingFinalizer(
                param=self.param,
                logger=self.logger,
                optimizer=self.optimizer,
                evaluate_fn=self.evaluate,
                epoch_avg_lengths=self._epoch_avg_lengths,
            )
            finalizer.finalize_training(env_steps_total=env_steps_total)