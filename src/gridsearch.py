from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Iterable, Optional

from dqn.agent import Agent
from dqn.parameter import Parameter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        type=str,
        default="LunarLander-v2",
        help="the id of the gym environment",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=4,
        help="the number of runs to execute, corresponds to the number of seeds",
    )
    return parser.parse_args()


ARGS = parse_args()
ENV_NAME = ARGS.env_name
NUMBER_OF_RUNS = ARGS.num_runs

EXPERIMENT_NAME = ENV_NAME
ENVS: Iterable[str] = [ENV_NAME]
NETWORKS: Iterable[str] = ["mlp_large"]
POLICIES: Iterable[str] = ["epsilon_greedy"]
EPSILONS: Iterable[Optional[float]] = [1.0]
EPSILONS_END: Iterable[Optional[float]] = [0.01]
LEARNING_RATES: Iterable[float] = [0.0001]
DEVICES: Iterable[str] = ["cuda"]
ENVS_COUNT: Iterable[int] = [5]
DISCOUNTS: Iterable[float] = [0.99]
COLLECT_STEPS: Iterable[int] = [200]
REPLAY_SIZES: Iterable[int] = [50_000]
EPOCHS: Iterable[int] = [300]
OPTIM_STEPS: Iterable[int] = [250]
BATCH_SIZES: Iterable[int] = [256]
PREWARMS: Iterable[int] = [5_000]
SEEDS: Iterable[Optional[int]] = [i for i in range(1, NUMBER_OF_RUNS + 1)]
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
    name = f"{env}_{idx}"
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
        experiment=EXPERIMENT_NAME,
    )


def run_one(param: Parameter) -> None:
    print(f"Starting run {param.name}")
    agent = Agent(param, True)
    agent.train()
    print(f"Finished run {param.name}")


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
