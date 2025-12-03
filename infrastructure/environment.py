import gymnasium as gym

def make_env(env_id="CartPole-v1"):
    """
       Create and return a Gymnasium environment.
       Args:
           env_id (str): The ID of the environment to create.
                         Default is "CartPole-v1".
       Returns:
           env: A Gymnasium environment instance.
       Example:
           env = make_env("CartPole-v1")
           state, info = env.reset()
       """
    return gym.make(env_id)
