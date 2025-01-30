import argparse
from Environment.SimpleEnvironment import SimpleEnvironment
from Agent.Human_Agent import Human_Agent
from Agent.AI_Agent import AI_Agent
from Agent.Double_DQN_Trainer_Agent import Double_DQN_Trainer_Agent
from Agent.DQN_Trainer_Agent import DQN_Trainer_Agent
from Model.DQN import DQN

device = "cpu"

# env = SimpleEnvironment()
# agent = Human_Agent(env=env)

# agent.init_pygame()

# agent.run()

def main():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning - Snake Agent"
    )
    parser.add_argument(
        "-e", "--environment", type=str, help="Environment to use for training. Option 1: SimpleEnvironment"
    )
    parser.add_argument(
        "-n", "--network", type=str, help="Neuronal Network to use. Option 1: DQN (Shallow Neural Network)"
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Pretrained model to use. e.g. trained_models/DQN/DQN_v1.1.2"
    )
    parser.add_argument(
        "-a", "--agent", type=str, help="Agent to use.\nOption 1: HI (play snake yourself)\nOption 2: AI (played by pretrained model)\nOption 3: DDQN (train new model from scratch using Double Deep Q-Learning)\nOption 3: DQN (train new model from scratch using Deep Q-Learning)"
    )
    parser.add_argument(
        "-v", "--visual", type=bool, help="Render Environment with pygame. (Set it on false only if you want to increase training efficiency)"
    )
    parser.add_argument(
        "-r", "--episodes", type=int, help="Number of episods to train the model"
    )

    args = parser.parse_args()

    env = SimpleEnvironment()
    model = DQN(6, 3).to(device)
    model.load_network("trained_models/DQN/DQN_v1.1.2")
    agent = AI_Agent(env=env, model=model)
    episods = 200


    if args.model:
        model.load_network(args.model)
    if args.episodes:
        episods = args.episodes
    if args.agent == "HI":
        agent = Human_Agent(env=env)
        agent.init_pygame(fps=25)
        while True:
            agent.run()
    elif args.agent == "AI":
        agent = AI_Agent(env=env, model=model)
        if args.visual:
            agent.init_pygame(60)
        while True:
            agent.run()
    elif args.agent == "DDQN":
        p_model = DQN(6, 3).to(device)
        agent = Double_DQN_Trainer_Agent(env=env, policy_model=p_model, target_model=model, c=1000, visualize=args.visual, buffer_capacity=10000, batch_size=125, lr=0.0001, eps_start=0.5, eps_end=0., decay_rate=10)
        if args.visual:
            agent.init_pygame(60)
        agent.train(episods, "test_model")
    elif args.agent == "DQN":
        agent = DQN_Trainer_Agent(env=env, policy_model=model, visualize=args.visual, buffer_capacity=10000, batch_size=125, lr=0.0001, eps_start=0.5, eps_end=0., decay_rate=10)
        if args.visual:
            agent.init_pygame(60)
        agent.train(episods, "test_model")


if __name__ == "__main__":
    main()