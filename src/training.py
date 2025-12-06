from __future__ import annotations

import argparse
from typing import Optional

from dqn.agent import Agent
from dqn.parameter import Parameter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent (CartPole defaults).")
    parser.add_argument("--device", default="cpu", dest="Device", help="Torch device string, e.g., cpu or cuda:0")
    parser.add_argument("--network", default="mlp_small", dest="Network", help="Network key from dqn.networks.get_model")
    parser.add_argument("--policy", default="epsilon_greedy", dest="Policy", help="Policy key from dqn.policies.get_policy")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon_greedy (ignored otherwise)")
    parser.add_argument("--epsilon-end", type=float, default=0.05, dest="epsilonEnd", help="End epsilon/temperature for scheduling")
    parser.add_argument("--env", default="CartPole-v1", dest="env", help="Gymnasium env id")
    parser.add_argument("--envs-count", type=int, default=4, dest="envsCount", help="Number of vectorized envs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, dest="learningRate", help="Adam learning rate")
    parser.add_argument("--discount", type=float, default=0.99, dest="discount", help="Discount factor gamma")
    parser.add_argument("--collect-steps", type=int, default=100, dest="collectSteps", help="Env steps collected per epoch")
    parser.add_argument("--replay-buffer-size", type=int, default=10_000, dest="replayBufferSize", help="Replay buffer capacity")
    parser.add_argument("--epochs", type=int, default=50, dest="epochs", help="Training epochs")
    parser.add_argument("--optim-per-epoch", type=int, default=200, dest="optimizationPerEpoch", help="Optimization steps per epoch")
    parser.add_argument("--batch-size", type=int, default=64, dest="batchSize", help="Mini-batch size")
    parser.add_argument("--prewarm", type=int, default=1000, dest="prewarm", help="Initial random transitions before training")
    parser.add_argument("--run-name", default="cartpole-run", dest="name", help="MLflow run name")
    parser.add_argument("--seed", type=int, default=None, dest="seed", help="Optional random seed for policy/envs")
    return parser.parse_args()


def make_params(args: argparse.Namespace) -> Parameter:
    eps: Optional[float] = args.epsilon
    if args.Policy.lower() != "epsilon_greedy":
        eps = None
    eps_end: Optional[float] = args.epsilonEnd if args.Policy.lower() == "epsilon_greedy" else None
    return Parameter(
        Device=args.Device,
        Network=args.Network,
        Policy=args.Policy,
        env=args.env,
        envsCount=args.envsCount,
        learningRate=args.learningRate,
        discount=args.discount,
        collectSteps=args.collectSteps,
        replayBufferSize=args.replayBufferSize,
        epochs=args.epochs,
        optimizationPerEpoch=args.optimizationPerEpoch,
        batchSize=args.batchSize,
        prewarm=args.prewarm,
        name=args.name,
        seed=args.seed,
        epsilon=eps,
        epsilonEnd=eps_end,
    )


def main() -> None:
    args = parse_args()
    param = make_params(args)
    agent = Agent(param)
    agent.train()


if __name__ == "__main__":
    main()
