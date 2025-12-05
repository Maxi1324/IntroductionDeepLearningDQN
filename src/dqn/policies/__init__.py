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

def getPolicy(seed=None, name: str = "greedy", online_predictor=None, **kwargs):
    """CamelCase helper to fetch a policy by name, primarily for legacy code."""
    return get_policy(name, seed=seed, online_predictor=online_predictor, **kwargs)


__all__ = ["Policy", "GreedyPolicy", "BoltzmannPolicy", "EpsilonGreedyPolicy", "get_policy", "getPolicy"]

