# Results for LunarLander-v3

This document summarizes the results for the LunarLander-v3 environment.

## Performance Graphs

<table>
  <tr>
    <td align="center"><b>Average Episode Length</b><br><img src="avg_episode_length.png" alt="Average Episode Length"></td>
    <td align="center"><b>Average Episode Reward</b><br><img src="avg_episode_reward.png" alt="Average Episode Reward"></td>
  </tr>
</table>

<p align="center">
  <img src="LunarLander-v3_mlp_large.gif" alt="Agent Performance GIF">
</p>

## Commands

### Test Model Interactively

Use the following command to test the agent interactively without creating a video.

```bash
python src/test_model.py --model-path results/LunarLander-v3/LunarLander-v3-model.pt --model-string mlp_large --env LunarLander-v3 --interactive --episodes 10
```

### Train Model

The following command can be used to train a new agent with a similar configuration as the one provided. The hyperparameters are based on the `gridsearch.py` configuration.

```bash
python src/training.py \
    --env LunarLander-v3 \
    --network mlp_large \
    --policy epsilon_greedy \
    --epsilon 1.0 \
    --epsilon-end 0.01 \
    --learning-rate 0.0001 \
    --device cuda \
    --envs-count 5 \
    --discount 0.99 \
    --collect-steps 200 \
    --replay-buffer-size 50000 \
    --epochs 300 \
    --optim-steps 250 \
    --batch-size 256 \
    --prewarm 5000 \
    --target-update-every 20 \
    --run-name LunarLander-v3-training
```

### Test Model and Generate Video

Use the following command to test the agent and generate a video of its performance.

```bash
python src/test_model.py --model-path results/LunarLander-v3/LunarLander-v3-model.pt --model-string mlp_large --env LunarLander-v3 --episodes 5 --output results/LunarLander-v3/LunarLander-v3_mlp_large.gif
```