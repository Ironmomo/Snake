from Environment.AbstractEnvironment import Environment, Direction
from Agent.Abstract_Agent import Agent
import pygame
from pygame.locals import *
import sys
import torch
from Model.DQN import DQN

# Check Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Agent
class AI_Agent(Agent):

    def __init__(self, env: Environment, model: DQN):
        super().__init__(env)
        self.model = model

    def __get_action(self, state):
        action = self.env.direction

        with torch.no_grad():
            state_tensor = torch.tensor(state).float().to(device)
            action_tensor = self.model(state_tensor)
            action_idx = torch.argmax(action_tensor)
            
        # right 
        if action_idx == 1:
            action = (-1 * action.value[1], action.value[0])
        # left
        if action_idx == 2:
            action = (action.value[1], -1 * action.value[0])

        action = Direction.get_direction(action)
        
        return action, action_idx
    
    def run(self):

        # Reset
        state, reward, done, _ = self.env.reset()
        self.env.direction = Direction.UP
        
        while not done:

            # Event
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.env.reset()
                        self.env.direction = Direction.UP

            # Select Action        
            action, _ = self.__get_action(state)

            # Step
            state, reward, done, _ = self.env.step(action)

            # Render
            if not done:
                self.env.render(self.screen)

            self.fpsClock.tick(self.fps)
            