from dqn.policies.greedy_policy import GreedyPolicy
from dqn.policies.boltzmann_policy import BoltzmannPolicy
from dqn.policies.epsilon_greedy_policy import EpsilonGreedyPolicy


POLICY_REGISTRY = {
    "greedy": GreedyPolicy,
    "boltzmann": BoltzmannPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
}


def get_policy(name: str, **kwargs):
    """
    Factory for policies that operate on already-computed Q-values.
    """
    key = name.lower()
    if key not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY.keys())}")
    return POLICY_REGISTRY[key](**kwargs)


class Policy:
    def __init__(self, dataCollection, policyO) -> None:
        self.dataCollection = dataCollection
        self.policyO = policyO
    
    def policy(self, states):
        logits = self.dataCollection.onlinePredictor.getLogits(states)
        actions = self.policyO.policy(logits)
        return actions