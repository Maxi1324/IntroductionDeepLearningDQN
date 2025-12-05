from __future__ import annotations

from dqn.agent import Agent
from dqn.parameter import Parameter


def main() -> None:
    param = Parameter(
        Device="cpu",
        Network="mlp_small",  # uses FeedForwardSmall with CartPole defaults
        Policy="epsilon_greedy",
        env="CartPole-v1",
        envsCount=4,
        learningRate=1e-3,
        discount=0.99,
        collectSteps=100,
        replayBufferSize=10000,
        epochs=50,
        optimizationPerEpoch=200,
        batchSize=64,
        prewarm=1000,
        name="cartpole-run",
        epsilon=0.1,
    )

    agent = Agent(param)
    agent.train()


if __name__ == "__main__":
    main()
