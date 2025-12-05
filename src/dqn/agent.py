from __future__ import annotations

import torch
from dqn.parameter import Parameter
from dqn.data_collection import DataCollect
from dqn.replay_buffer import ReplayBuffer
from dqn.optimizer import Optimizer
from dqn.online_predictor import OnlinePredictor
from dqn.utils.config_validation import validate_configuration
from dqn.utils.training_logger import TrainingLogger
from dqn.policies import get_policy


class Agent:
    def __init__(self, param: Parameter, consoleLogging=True) -> None:
        self.param = param
        self.device = torch.device(param.Device)

        policy_kwargs = {}
        if param.Policy.lower() == "epsilon_greedy" and param.epsilon is not None:
            policy_kwargs["epsilon"] = param.epsilon
        if param.seed is not None:
            policy_kwargs["seed"] = param.seed
        policy_instance = get_policy(param.Policy, **policy_kwargs)
        self.replayBuffer: ReplayBuffer = ReplayBuffer(param.replayBufferSize, self.device)
        self.dataCollection: DataCollect = DataCollect(param.env, param.envsCount, self.replayBuffer, policy_instance)
        self.optimizer = Optimizer(param)
        onlinePredictor = OnlinePredictor(self.optimizer.getOnlineNetwork, self.device)
        policy_instance.setOnlinePredictor(onlinePredictor)
        self.policy = policy_instance
        validate_configuration(self.param, self.dataCollection, self.optimizer, self.policy, self.device)
        self.logger = TrainingLogger(run_name=getattr(param, "name", "dqn-run"), consoleLogging=consoleLogging)

    def train(self) -> None:
        with self.logger.start_run():
            self.logger.log_all_params(self.param)
            self.dataCollection.collect(self.param.prewarm)
            global_step = 0
            for epoch in range(0, self.param.epochs):
                self.logger.start_epoch(epoch, self.param.epochs)
                avg_len, avg_ret = self.dataCollection.collect(self.param.collectSteps)

                for _ in range(0, self.param.optimizationPerEpoch):
                    states, actions, rewards, next_states, dones = self.replayBuffer.sample(self.param.batchSize)
                    loss = self.optimizer.optimize(states, actions, rewards, next_states, dones)
                    self.logger.log_loss(loss, step=global_step)
                    global_step += 1

                self.logger.log_episode_stats(avg_len, avg_ret)
                self.logger.end_epoch(global_step)

                self.optimizer.updateTargetNetwork()
