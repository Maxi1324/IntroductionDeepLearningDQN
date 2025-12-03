from infrastructure.environment import make_env
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.data_collection import collect_experience
import random

env = make_env()
buffer = ReplayBuffer(1000)

def random_policy(state):
    return random.randint(0, 1)

collect_experience(env, random_policy, buffer, num_steps=500)
print("Replay buffer size:", len(buffer))
