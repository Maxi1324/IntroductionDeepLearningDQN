import argparse
import torch
import gymnasium as gym
import numpy as np
import imageio
from pathlib import Path

from dqn.networks import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-string", type=str, required=True)
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--output", type=str, default="videos", help="Video output directory")
    p.add_argument(
        "--episodes", type=int, default=5, help="Anzahl Episoden (Runs) pro Aufruf"
    )
    p.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode with live preview (no video saved)."
    )
    args = p.parse_args()

    render_mode = "human" if args.interactive else "rgb_array"
    env = gym.make(args.env, render_mode=render_mode)

    output_arg = Path(args.output)
    name_prefix = f"{args.env}_{args.model_string}"

    if str(output_arg).endswith(".mp4") or str(output_arg).endswith(".gif"):
        # User provided a file path
        output_file_base = output_arg.with_suffix('')
        output_path = output_arg.parent
    else:
        # User provided a directory
        output_path = output_arg
        output_file_base = output_path / name_prefix

    if not args.interactive:
        output_path.mkdir(parents=True, exist_ok=True)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n
    model = get_model(args.model_string, state_dim=state_dim, action_dim=action_dim)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    rewards = []
    all_frames = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                if args.interactive:
                    env.render()
                else:
                    all_frames.append(env.render())

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = model(obs_tensor).argmax(dim=1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            rewards.append(ep_reward)
            print(f"Episode {ep+1}/{args.episodes} finished. Reward: {ep_reward}")
    finally:
        env.close()

    if rewards:
        avg = sum(rewards) / len(rewards)
        print(f"Durchschnitts-Reward über {len(rewards)} Episoden: {avg}")

    if not args.interactive and all_frames:
        mp4_path = output_file_base.with_suffix(".mp4")
        gif_path = output_file_base.with_suffix(".gif")
        
        print(f"Saving {len(all_frames)} frames to {mp4_path}")
        imageio.mimsave(mp4_path, all_frames, fps=30)
        
        print(f"Saving {len(all_frames)} frames to {gif_path}")
        imageio.mimsave(gif_path, all_frames, fps=30)
        
        print(f"Videos gespeichert in: {output_path}, Prefix: {output_file_base.name}")


if __name__ == "__main__":
    main()
