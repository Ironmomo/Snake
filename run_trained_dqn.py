from Environment.Environment import Environment
from Agent.AI_Agent import AI_Agent
from Model.DQN import DQN

device = "cpu"

env = Environment()
dqn = DQN(6, 3).to(device)
dqn.load_network("trained_models/DQN/DQN_v1.1.2")
agent = AI_Agent(env, dqn)
agent.init_pygame(fps=25)
while True:
    agent.run()