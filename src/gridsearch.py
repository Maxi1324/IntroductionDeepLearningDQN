from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Iterable, Optional

from dqn.agent import Agent
from dqn.parameter import Parameter

gl="bigtime3"
NETWORKS: Iterable[str] = ["mlp_small"]
POLICIES: Iterable[str] = ["epsilon_greedy"]
EPSILONS: Iterable[Optional[float]] = [0.2]  # start epsilon / temperature
EPSILONS_END: Iterable[Optional[float]] = [0.01]  # end epsilon / temperature
LEARNING_RATES: Iterable[float] = [ 0.0007]
DEVICES: Iterable[str] = ["cpu"]
ENVS: Iterable[str] = ["CartPole-v1"]
ENVS_COUNT: Iterable[int] = [4,10,30,50]
DISCOUNTS: Iterable[float] = [0.99,0.9,0.999]
COLLECT_STEPS: Iterable[int] = [10,50,100]
REPLAY_SIZES: Iterable[int] = [10_000,20_000,50_000]
EPOCHS: Iterable[int] = [10,20,50,100]
OPTIM_STEPS: Iterable[int] = [10,30,50]
BATCH_SIZES: Iterable[int] = [32,64,128,256]
PREWARMS: Iterable[int] = [2000,10000]
SEEDS: Iterable[Optional[int]] = [1]


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
    )

i = 0
def run_one(param: Parameter) -> None:
    global i
    i +=1
    print(f"Starting run {param.name}")
    agent = Agent(param, False)
    agent.train()
    print("done ",i)

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
        )
    )
    params = [
        build_param(idx, *combo)
        for idx, combo in enumerate(combos)
    ]
    with ProcessPoolExecutor(max_workers=min(len(params), 15)) as ex:
        futures = [ex.submit(run_one, p) for p in params]
        wait(futures)
        # Surface any worker exceptions
        for fut in futures:
            exc = fut.exception()
            if exc:
                print(f"Run failed: {exc}")


if __name__ == "__main__":
    main()
