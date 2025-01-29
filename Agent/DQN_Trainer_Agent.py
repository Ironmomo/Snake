from Environment.Environment import Environment, Direction
from Agent.Abstract_Agent import Agent
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import torch
from Model.DQN import DQN, ReplayMemory
import math
import random


# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

plt.ion()

# Agent
class DQN_Trainer_Agent(Agent):

    def __init__(self, env: Environment, policy_model: DQN, visualize: bool = False, buffer_capacity: int = 10000, batch_size: int = 125, lr: int = 0.1, max_steps: int = 2000, gamma: float = 0.9, eps_start: float = 0.1, eps_end: float = 0.001, decay_rate: int = 1000):
        super().__init__(env)
        self.policy_model = policy_model
        self.buffer = ReplayMemory(buffer_capacity)
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.score_buffer = []
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.lr = lr
        self.max_steps = max_steps
        self.visualize = visualize

    def __save_plot_and_hyper(self, path):
        pdffig = PdfPages(path)

        score_mean = [sum(self.score_buffer[idx-10:idx]) / 10 if idx >= 10 else sum(self.score_buffer[0:idx+1]) / (idx + 1) for idx, m in enumerate(self.score_buffer)]
       

        plt.clf()
        plt.plot(self.score_buffer, "blue", label="Score")
        plt.plot(score_mean, "orange", label="Mean")
        plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Training performance")
        plt.savefig(pdffig, format="pdf")

        metadata = pdffig.infodict()
        metadata['Title'] = 'Hyperparameters'
        metadata['buffer_capacity'] = self.buffer_capacity
        metadata['batch_size'] = self.batch_size
        metadata['eps_start'] = self.eps_start
        metadata['eps_end'] = self.eps_end
        metadata['decay_rate'] = self.decay_rate
        metadata['gamma'] = self.gamma
        metadata['lr'] = self.lr

        pdffig.close()

    def __plot_training(self):
        score_mean = [sum(self.score_buffer[idx-10:idx]) / 10 if idx >= 10 else sum(self.score_buffer[0:idx+1]) / (idx + 1) for idx, m in enumerate(self.score_buffer)]
       
        plt.clf()
        plt.plot(self.score_buffer, "blue", label="Score")
        plt.plot(score_mean, "orange", label="Mean")
        plt.legend()
        plt.pause(0.001)
        plt.show()


    def __calc_epsilon(self):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * len(self.score_buffer) / self.decay_rate)
        return eps_threshold   


    def __get_action(self, state):

        action = self.env.direction

        if random.random() <= self.__calc_epsilon():
            action_idx = random.sample([0,1,2], 1)[0]
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).float().to(device)
                action_tensor = self.policy_model(state_tensor)
                action_idx = torch.argmax(action_tensor).item()

        # right 
        if action_idx == 1:
            action = (-1 * action.value[1], action.value[0])
        # left
        if action_idx == 2:
            action = (action.value[1], -1 * action.value[0])

        action = Direction.get_direction(action)
        
        return action, action_idx
    

    def __update(self):

        if len(self.buffer) < self.batch_size:
            return
        
        # get batch from ReplayBuffer
        state_batch, action_batch, next_state_batch, reward_batch = zip(*self.buffer.sample(self.batch_size))
        # get index of non_final samples
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=device, dtype=torch.bool)
        non_final_next_state_batch = [n_s for n_s in next_state_batch if n_s]

        # batch to tensor
        state_tensor = torch.tensor(state_batch).float().to(device)
        action_tensor = torch.tensor(action_batch).to(device).view(-1,1)
        next_state_val_tensor = torch.zeros(self.batch_size, device=device)
        reward_tensor = torch.tensor(reward_batch).to(device)

        # get non final next states
        non_final_next_state_tensor = torch.tensor(non_final_next_state_batch).float().to(device)

        # get state prediction
        state_val_tensor = self.policy_model(state_tensor).gather(1, action_tensor).view(-1)
        #state_val_tensor = state_val_tensor.gather(1, action_tensor).view(-1)
        # get next_state prediction
        next_state_val_tensor[non_final_mask] = self.policy_model(non_final_next_state_tensor).max(1).values

        # calculate TD
        expected_next_state_val = next_state_val_tensor * self.gamma + reward_tensor

        # Optimize the model
        criterion = torch.nn.SmoothL1Loss()

        loss = criterion(state_val_tensor, expected_next_state_val)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    
    def train(self, num_episodes: int, model_name):
        for e in range(num_episodes):
            # Reset
            state, reward, done, self.score = self.env.reset()
            self.env.direction = Direction.UP
            num_steps = 0

            while not done and num_steps < self.max_steps:
                
                num_steps += 1

                # Select Action        
                action, action_idx = self.__get_action(state)

                # Step
                next_state, reward, done, self.score = self.env.step(action)

                # Save to buffer
                self.buffer.push(state, action_idx, next_state, reward)

                state = next_state

                # Update Model
                self.__update()                 

                # Render
                if not done and self.visualize:

                    # Event
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            self.policy_model.save_network(model_name)
                            pygame.quit()
                            sys.exit()

                    self.env.render(self.screen)

                    self.fpsClock.tick(self.fps)
            
            self.score_buffer.append(self.score)
            self.__plot_training()

        self.policy_model.save_network(model_name)
        self.__save_plot_and_hyper("figure.pdf")



            