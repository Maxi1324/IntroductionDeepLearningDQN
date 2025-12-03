from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
        A replay buffer for storing experiences collected from the environment.
        Used in Deep Q-Learning (DQN) to sample past experiences randomly
        and stabilize training.
        """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size
