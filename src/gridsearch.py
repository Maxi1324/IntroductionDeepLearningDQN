from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Iterable, Optional

from dqn.agent import Agent
from dqn.parameter import Parameter

gl = "asd"


NETWORKS: Iterable[str] = ["mlp_large"]
POLICIES: Iterable[str] = ["epsilon_greedy"]

EPSILONS: Iterable[Optional[float]] = [1.0]
EPSILONS_END: Iterable[Optional[float]] = [0.01]
LEARNING_RATES: Iterable[float] = [0.0001]
DEVICES: Iterable[str] = ["cuda"]
ENVS: Iterable[str] = ["CartPole-v1"]
ENVS_COUNT: Iterable[int] = [5]
DISCOUNTS: Iterable[float] = [0.99]
COLLECT_STEPS: Iterable[int] = [200]
REPLAY_SIZES: Iterable[int] = [50_000]
EPOCHS: Iterable[int] = [50]
OPTIM_STEPS: Iterable[int] = [50]
BATCH_SIZES: Iterable[int] = [256,512]
PREWARMS: Iterable[int] = [5_000]
SEEDS: Iterable[Optional[int]] = [1,2,3,4,5,6]
TARGET_UPDATE_EVERY: Iterable[int] = [20]


def build_param(
    idx: int,
    network: str,
    policy: str,
    epsilon: Optional[float],
    epsilon_end: Optional[float],
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
    target_update_every: int,
) -> Parameter:
    eps_val = epsilon if policy.lower() == "epsilon_greedy" else None
    eps_end_val = epsilon_end if policy.lower() == "epsilon_greedy" else None
    if policy.lower() == "boltzmann":
        eps_val = epsilon
        eps_end_val = epsilon_end
    name = f"{gl}"
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
        epsilonEnd=eps_end_val,
        targetUpdateEvery=target_update_every,
    )


i = 0
def run_one(param: Parameter) -> None:
    global i
    i += 1
    print(f"Starting run {param.name}")
    agent = Agent(param, True)
    agent.train()
    print("done ", i)


def main() -> None:
    combos = list(
        itertools.product(
            NETWORKS,
            POLICIES,
            EPSILONS,
            EPSILONS_END,
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
            TARGET_UPDATE_EVERY,
        )
    )
    params = [
        build_param(idx, *combo)
        for idx, combo in enumerate(combos)
    ]
    with ProcessPoolExecutor(max_workers=min(len(params), 20)) as ex:
        futures = [ex.submit(run_one, p) for p in params]
        wait(futures)
        # Surface any worker exceptions
        for fut in futures:
            exc = fut.exception()
            if exc:
                print(f"Run failed: {exc}")


if __name__ == "__main__":
    main()
