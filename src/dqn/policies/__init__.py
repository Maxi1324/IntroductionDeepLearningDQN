from dqn.policies.policy_base import Policy
from dqn.policies.greedy_policy import GreedyPolicy
from dqn.policies.boltzmann_policy import BoltzmannPolicy
from dqn.policies.epsilon_greedy_policy import EpsilonGreedyPolicy


POLICY_REGISTRY = {
    "greedy": GreedyPolicy,
    "boltzmann": BoltzmannPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
}


def get_policy(name: str, **kwargs):
    key = name.lower()
    if key not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY.keys())}")
    return POLICY_REGISTRY[key](**kwargs)



