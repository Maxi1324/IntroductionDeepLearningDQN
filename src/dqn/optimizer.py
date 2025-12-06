from __future__ import annotations

import copy

import torch
from torch import nn

from dqn.parameter import Parameter
from dqn.networks import get_model


class Optimizer:
    def __init__(self, param: Parameter, state_dim: int, action_dim: int) -> None:
        self.param: Parameter = param

        self.online_network: nn.Module = get_model(
            param.Network,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.target_network: nn.Module = copy.deepcopy(self.online_network)

        device = torch.device(param.Device)
        self.online_network = self.online_network.to(device)
        self.target_network = self.target_network.to(device)

        self._optimize: torch.optim.Optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=param.learningRate
        )
        self.discount: float = float(param.discount)
        self.loss_fn: nn.Module = nn.MSELoss()

    def optimize(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        q_values = self.online_network(states)
        action_q = q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_online_q = self.online_network(next_states)
            next_actions = torch.argmax(next_online_q, dim=1)
            next_target_q = self.target_network(next_states)
            next_q_selected = next_target_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards + self.discount * (1.0 - dones.float()) * next_q_selected

        loss = self.loss_fn(action_q, target)

        self._optimize.zero_grad(set_to_none=True)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self._optimize.step()

        return float(loss.item())


    def updateTargetNetwork(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def getOnlineNetwork(self):
        return self.online_network
