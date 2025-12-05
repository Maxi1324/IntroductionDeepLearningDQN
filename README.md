# IntroductionDeepLearningDQN

Minimal notes on how to set up and run the included DQN sample.

## Project layout (relevant bits)
- `src/dqn/agent.py` – wires together replay buffer, data collection, optimizer, policy, logging.
- `src/dqn/data_collection.py` – vectorized Gymnasium rollout + episode stats.
- `src/dqn/replay_buffer.py` – cyclic buffer with Torch-ready sampling.
- `src/dqn/optimizer.py` – DQN update (Double-Q action selection).
- `src/dqn/policies/` – greedy/epsilon-greedy/boltzmann and registry (`get_policy`).
- `src/dqn/networks/` – small MLPs / conv nets and registry (`get_model`).
- `src/dqn/utils/` – config validation, training logger, parameter dataclass.
- `src/training.py` – example entrypoint: trains CartPole with mlflow logging to `mlruns/`.

## Setup
```bash
cd IntroductionDeepLearningDQN          # enter project root
python -m venv .venv                    # create virtual environment
.\.venv\Scripts\Activate.ps1            # PowerShell; or source .venv/bin/activate on Unix
pip install -e .                        # install in editable mode (includes mlflow, gymnasium, torch)
```

## Run the sample
```bash
python -m src.training
```
This runs the CartPole example with mlflow logging to `mlflow.db` (SQLite backend).

Example with custom options (see `python -m src.training --help` for all):
```bash
python -m src.training 
  --env CartPole-v1 
  --network mlp_small 
  --policy epsilon_greedy 
  --epsilon 0.05 
  --device cpu 
  --epochs 100 
  --run-name cartpole-demo
```

## View logs (optional)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open http://localhost:5000 to browse metrics (loss, epoch_loss, avg_episode_length/reward).
