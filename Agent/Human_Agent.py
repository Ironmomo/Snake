from Environment.AbstractEnvironment import Environment, Direction
from Agent.Abstract_Agent import Agent
import pygame
from pygame.locals import *
import sys

class Human_Agent(Agent):

    def __init__(self, env: Environment):
        super().__init__(env)

    def __get_action(self, state):
        action = None
        # Event
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
            # Action
            if event.type == KEYDOWN:
                if event.key == pygame.K_UP:
                    action = Direction.UP
                
                if event.key == pygame.K_RIGHT:
                    action = Direction.RIGHT
                
                if event.key == pygame.K_DOWN:
                    action = Direction.DOWN

                if event.key == pygame.K_LEFT:
                    action = Direction.LEFT
                
                if event.key == pygame.K_RETURN:
                    self.env.reset()
        
        return action
    
    def run(self):
        state, reward, done, _ = self.env.reset()

        while not done:
            action = self.__get_action(state)

            state, reward, done, _ = self.env.step(action)

            if not done:
                self.env.render(self.screen)

            self.fpsClock.tick(self.fps)
            