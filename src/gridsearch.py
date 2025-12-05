from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Iterable, Optional

from dqn.agent import Agent
from dqn.parameter import Parameter


# Define your search space here
NETWORKS: Iterable[str] = ["mlp_small", "mlp_medium"]
POLICIES: Iterable[str] = ["epsilon_greedy", "boltzmann"]
EPSILONS: Iterable[Optional[float]] = [0.2, 0.1, 0.3]  # only used for epsilon_greedy
LEARNING_RATES: Iterable[float] = [0.001, 0.0001, 0.0005]

# Common defaults
BASE_CFG = {
    "Device": "cpu",
    "env": "CartPole-v1",
    "envsCount": 4,
    "discount": 0.99,
    "collectSteps": 100,
    "replayBufferSize": 10_000,
    "epochs": 30,
    "optimizationPerEpoch": 200,
    "batchSize": 64,
    "prewarm": 1000,
}


def build_param(idx: int, network: str, policy: str, epsilon: Optional[float], lr: float) -> Parameter:
    eps_val = epsilon if policy.lower() == "epsilon_greedy" else None
    name = f"grid-{idx}-{network}-{policy}-lr{lr}-eps{eps_val}"
    return Parameter(
        Device=BASE_CFG["Device"],
        Network=network,
        Policy=policy,
        env=BASE_CFG["env"],
        envsCount=BASE_CFG["envsCount"],
        learningRate=lr,
        discount=BASE_CFG["discount"],
        collectSteps=BASE_CFG["collectSteps"],
        replayBufferSize=BASE_CFG["replayBufferSize"],
        epochs=BASE_CFG["epochs"],
        optimizationPerEpoch=BASE_CFG["optimizationPerEpoch"],
        batchSize=BASE_CFG["batchSize"],
        prewarm=BASE_CFG["prewarm"],
        name=name,
        epsilon=eps_val,
    )


def run_one(param: Parameter) -> None:

    print(f"Starting run {param.name}")
    agent = Agent(param, False)
    agent.train()


def main() -> None:
    combos = list(itertools.product(NETWORKS, POLICIES, EPSILONS, LEARNING_RATES))
    params = [build_param(idx, net, pol, eps, lr) for idx, (net, pol, eps, lr) in enumerate(combos)]
    with ProcessPoolExecutor(max_workers=min(len(params), 50)) as ex:
        futures = [ex.submit(run_one, p) for p in params]
        wait(futures)
        # Surface any worker exceptions
        for fut in futures:
            exc = fut.exception()
            if exc:
                print(f"Run failed: {exc}")


if __name__ == "__main__":
    main()
