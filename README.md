# Deep Q-Network (DQN) Implementation

## 1. Overview

This project provides a comprehensive implementation of a Deep Q-Network (DQN) agent designed to solve reinforcement learning environments from the [Gymnasium](https://gymnasium.farama.org/) library. The agent is built using PyTorch and includes integrated experiment tracking with [MLflow](https://mlflow.org/).

## 2. Install

Git, Python13, docker
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
docker compose pull
```

## 3 Start

Start postgres for MLflow
```bash
docker compose up
```

Start Mlflow
```bash
mlflow ui --backend-store-uri postgresql://postgres@localhost:5432/mlflow --port 5000
```

Start individual training run
```bash
python src/training.py --device <DEVICE> --network <NETWORK> ...
```
| Argument | Description | Default |
|---|---|---|
| `--device` | Torch device string, e.g., cpu or cuda:0 | `cpu` |
| `--network` | Network key from dqn.networks.get_model | `mlp_small` |
| `--policy` | Policy key from dqn.policies.get_policy | `epsilon_greedy` |
| `--epsilon` | Epsilon for epsilon_greedy (ignored otherwise) | `0.1` |
| `--epsilon-end` | End epsilon/temperature for scheduling | `0.05` |
| `--env` | Gymnasium env id | `CartPole-v1` |
| `--envs-count` | Number of vectorized envs | `4` |
| `--learning-rate` | Adam learning rate | `1e-3` |
| `--discount` | Discount factor gamma | `0.99` |
| `--collect-steps` | Env steps collected per epoch | `100` |
| `--replay-buffer-size` | Replay buffer capacity | `10000` |
| `--epochs` | Training epochs | `50` |
| `--optim-per-epoch`| Optimization steps per epoch | `200` |
| `--batch-size` | Mini-batch size | `64` |
| `--prewarm`| Initial random transitions before training | `1000` |
| `--run-name` | MLflow run name | `cartpole-run` |
| `--seed`| Optional random seed for policy/envs | `None` |

Start gridsearch

```bash

python src/gridsearch.py --env-name <ENV_NAME> --num-runs <NUMBER_OF_RUNS>

```

The gridsearch can be configured with the following command-line arguments. Other hyperparameters can be configured by editing the global variables in `src/gridsearch.py`.



| Argument | Description | Default |
|---|---|---|
| `--env-name` | The id of the gym environment | `LunarLander-v2` |
| `--num-runs` | The number of runs to execute, corresponds to the number of seeds | `4` |



Test a trained model

```bash

python src/test_model.py --model-path <PATH_TO_MODEL> --model-string <NETWORK> --env <ENV_ID>

```

This script runs a trained model in a given environment and records videos of the episodes.



| Argument | Description | Default |
|---|---|---|
| `--model-path` | Path to the saved model file (`.pt`). | (required) |
| `--model-string`| Network key from `dqn.networks.get_model` (must match the trained model). | (required) |
| `--env` | Gymnasium env id. | (required) |
| `--output` | Directory to save the recorded videos. | `videos` |
| `--not-interactive`| Disable the live preview window. | (not set) |
| `--episodes` | Number of episodes to run and record. | `5` |



## 4 Results