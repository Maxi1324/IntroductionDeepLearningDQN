import argparse
import time
import torch
import gymnasium as gym
import numpy as np

from dqn.networks import get_model

def main():
    parser = argparse.ArgumentParser(description="Test a DQN model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model .pt file")
    parser.add_argument("--model-string", type=str, required=True, help="Model architecture string (e.g., mlp_large)")
    parser.add_argument("--env", type=str, required=True, help="Gymnasium environment name (e.g., CartPole-v1)")
    args = parser.parse_args()

    # Create the environment
    env = gym.make(args.env, render_mode="human")
    
    # Get state and action dimensions from the environment
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    # Instantiate the model
    model = get_model(args.model_string, state_dim=state_dim, action_dim=action_dim)

    # Load the state dict
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.eval() # Set the model to evaluation mode

    # Main loop
    while True:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0)

            # Get action from the model
            with torch.no_grad():
                action_values = model(obs_tensor)
                action = torch.argmax(action_values).item()

            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                done = True
            
            time.sleep(0.01) # Slow down rendering a bit
        
        print(f"Episode finished. Total reward: {total_reward}")

if __name__ == "__main__":
    main()
