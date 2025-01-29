from Model.DQN import DQN
from Agent.Double_DQN_Trainer_Agent import Double_DQN_Trainer_Agent as Agent
#from Agent.DQN_Trainer_Agent import DQN_Trainer_Agent as Agent

from Environment.Environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

env = Environment()
dqn = DQN(6, 3).to(device)
dqn2 = DQN(6, 3).to(device)

agent = Agent(env=env, policy_model=dqn, target_model=dqn2, c=1000, visualize=True, buffer_capacity=10000, batch_size=125, lr=0.0001, eps_start=0.5, eps_end=0., decay_rate=10)
agent.init_pygame(60)

#agent = Agent(env=env, policy_model=dqn, visualize=False, buffer_capacity=10000, batch_size=125, lr=0.0001, eps_start=0.5, eps_end=0., decay_rate=10)


agent.train(200, "test_model")
