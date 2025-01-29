from Environment.Environment import Environment
from Agent.Human_Agent import Human_Agent

env = Environment()
agent = Human_Agent(env=env)

agent.init_pygame()

agent.run()