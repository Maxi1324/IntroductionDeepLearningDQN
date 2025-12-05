def collect_experience(env, policy_fn, replay_buffer, num_steps=1000):
    """
       Collect experiences from the environment using a given policy function
       and store them in the replay buffer.

       Args:
           env: The environment object (e.g., CartPole-v1)
           policy_fn: A function that takes the current state and returns an action
           replay_buffer: ReplayBuffer object to store experiences
           num_steps: Number of steps to collect
       """
    state, _ = env.reset()
    for _ in range(num_steps):
        action = policy_fn(state)
        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
