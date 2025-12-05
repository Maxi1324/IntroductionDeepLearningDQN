from torch.nn import Module
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

    def train(self):
        self.dataCollection.collect(self.param.prewarm)
        for _ in range(0, self.param.epochs):
            self.dataCollection.collect(self.param.collectSteps)
            
            for _ in range(0,self.param.optimizationPerEpoch):
                states, actions, rewards, next_states, dones = self.replayBuffer.sample(self.param.batchSize)
                self.optimizer.optimize(states, actions, rewards, next_states, dones)
            
            self.optimizer.updateTargetNetwork()