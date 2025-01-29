from Environment.Environment import Environment
import pygame
from pygame.locals import *


class Agent():

    def __init__(self, env: Environment):
        self.env: Environment = env
    

    def init_pygame(self, fps: int = 5):
        # Pygame init
        pygame.init()

        self.fps = fps
        self.fpsClock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.env.width, self.env.height))

    def __get_action(self, state):
        pass
    
    def run(self):
        pass
            