import torch
from gymnasium import spaces

from dqn.parameter import Parameter
from dqn.data_collection import DataCollect
from dqn.replay_buffer import ReplayBuffer
from dqn.optimizer import Optimizer
from dqn.online_predictor import OnlinePredictor

class Agent:
    
    def __init__(self, param: Parameter) -> None:
        self.param = param
        self.replayBuffer: ReplayBuffer = ReplayBuffer(param.replayBufferSize, param.Device)
        self.dataCollection:DataCollect = DataCollect(param.env,param.envsCount,self.replayBuffer,param.Policy)
        self.optimizer = Optimizer(param)
        onlinePredictor = OnlinePredictor(self.optimizer.getOnlineNetwork, param.Device)
        param.Policy.setOnlinePredictor(onlinePredictor)
        self._validateConfiguration()

    def train(self):
        self.dataCollection.collect(self.param.prewarm)
        for _ in range(0, self.param.epochs):
            self.dataCollection.collect(self.param.collectSteps)
                
            for _ in range(0,self.param.optimizationPerEpoch):
                states, actions, rewards, next_states, dones = self.replayBuffer.sample(self.param.batchSize)
                self.optimizer.optimize(states, actions, rewards, next_states, dones)
            
            self.optimizer.updateTargetNetwork()

    def _validateConfiguration(self) -> None:
        """Check basic parameter consistency and abort early with a short message."""
        p = self.param
        env = self.dataCollection.env
        states_np = self.dataCollection._states

        def fail(msg: str) -> None:
            print(f"Configuration error: {msg}")
            raise SystemExit(1)

        if not callable(p.Network):
            fail("Network must be a callable/constructor.")
        if p.learningRate <= 0:
            fail("learningRate must be > 0.")
        if not (0.0 <= p.discount <= 1.0):
            fail("discount must be in [0, 1].")
        if p.batchSize <= 0:
            fail("batchSize must be > 0.")
        if p.replayBufferSize < p.batchSize:
            fail("replayBufferSize must be >= batchSize.")
        if p.collectSteps <= 0:
            fail("collectSteps must be > 0.")
        if p.prewarm < p.batchSize:
            fail("prewarm must be >= batchSize for sampling.")
        if p.envsCount <= 0:
            fail("envsCount must be > 0.")
        if not p.env:
            fail("env id must be set.")
        if p.Policy.online_predictor is None:
            fail("Policy has no OnlinePredictor attached.")

        try:
            action_space = env.single_action_space
            obs_space = env.single_observation_space
        except Exception as e:
            fail(f"Could not inspect env spaces: {e}")

        if not isinstance(action_space, spaces.Discrete):
            fail("Action space must be Discrete.")

        if not isinstance(obs_space, spaces.Box) or obs_space.shape is None:
            fail("Observation space must be a Box with a defined shape.")

        expected_shape = (p.envsCount, *obs_space.shape)
        if states_np.shape != expected_shape:
            fail(f"Env reset states shape {states_np.shape} does not match expected {expected_shape}.")

        states_t = torch.as_tensor(states_np, device=p.Device, dtype=torch.float32)
        try:
            with torch.no_grad():
                logits = self.optimizer.online_network(states_t)
        except Exception as e:
            fail(f"Network forward failed: {e}")

        if logits.ndim < 2:
            fail("Network output must be at least 2D [batch, actions].")
        if logits.shape[0] != p.envsCount:
            fail(f"Network batch dimension {logits.shape[0]} != envsCount {p.envsCount}.")
        if logits.shape[-1] != action_space.n:
            fail(f"Network action dimension {logits.shape[-1]} != env action space {action_space.n}.")
