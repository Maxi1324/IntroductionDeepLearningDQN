from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Iterable, Optional

from dqn.agent import Agent
from dqn.parameter import Parameter


# Define your search space here
NETWORKS: Iterable[str] = ["mlp_small"]
POLICIES: Iterable[str] = ["epsilon_greedy", "boltzmann"]
EPSILONS: Iterable[Optional[float]] = [0.2]  # only used for epsilon_greedy
LEARNING_RATES: Iterable[float] = [0.001, 0.0001, 0.0005]
DEVICES: Iterable[str] = ["cpu"]
ENVS: Iterable[str] = ["CartPole-v1"]
ENVS_COUNT: Iterable[int] = [4]
DISCOUNTS: Iterable[float] = [0.99]
COLLECT_STEPS: Iterable[int] = [100]
REPLAY_SIZES: Iterable[int] = [10_000]
EPOCHS: Iterable[int] = [30]
OPTIM_STEPS: Iterable[int] = [200]
BATCH_SIZES: Iterable[int] = [64]
PREWARMS: Iterable[int] = [1000]
SEEDS: Iterable[Optional[int]] = [1]


def build_param(
    idx: int,
    network: str,
    policy: str,
    epsilon: Optional[float],
    lr: float,
    device: str,
    env: str,
    envs_count: int,
    discount: float,
    collect_steps: int,
    replay_size: int,
    epochs: int,
    optim_steps: int,
    batch_size: int,
    prewarm: int,
    seed: Optional[int],
) -> Parameter:
    eps_val = epsilon if policy.lower() == "epsilon_greedy" else None
    name = f"grid-{idx}-{network}-{policy}-lr{lr}-eps{eps_val}-seed{seed}"
    return Parameter(
        Device=device,
        Network=network,
        Policy=policy,
        env=env,
        envsCount=envs_count,
        learningRate=lr,
        discount=discount,
        collectSteps=collect_steps,
        replayBufferSize=replay_size,
        epochs=epochs,
        optimizationPerEpoch=optim_steps,
        batchSize=batch_size,
        prewarm=prewarm,
        name=name,
        seed=seed,
        epsilon=eps_val,
    )


def run_one(param: Parameter) -> None:

    print(f"Starting run {param.name}")
    agent = Agent(param, False)
    agent.train()


def main() -> None:
    combos = list(
        itertools.product(
            NETWORKS,
            POLICIES,
            EPSILONS,
            LEARNING_RATES,
            DEVICES,
            ENVS,
            ENVS_COUNT,
            DISCOUNTS,
            COLLECT_STEPS,
            REPLAY_SIZES,
            EPOCHS,
            OPTIM_STEPS,
            BATCH_SIZES,
            PREWARMS,
            SEEDS,
        )
    )
    params = [
        build_param(idx, *combo)
        for idx, combo in enumerate(combos)
    ]
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
