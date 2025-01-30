from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
import pygame

# Init
Point = namedtuple('Point', ['x', 'y'])

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    def get_direction(x):
        if x in list(Direction):
            return x
        for direction in Direction:
            if direction.value == x:
                return direction

    def get_index(enum_member):
        return list(Direction).index(enum_member)

class Environment(ABC):
  
    def __init__(self, width: int = 1200, height: int = 1200, tile_size: int = 40):

        # Game
        self.width: int = width
        self.height: int = height
        self.tile_size: int = tile_size
        self.score: int = 0
        self.done: bool = False

        # Snake
        self.snake = [Point((width / tile_size / 2 - 1) * tile_size, (height / tile_size / 2 - 1) * tile_size)]
        self.direction: Direction = None

        self._place_food()

    
    @abstractmethod
    def _get_state(self):
        pass


    def reset(self):
        self.score = 0
        self.done = False

        self.snake = [Point((self.width / self.tile_size / 2 - 1) * self.tile_size, (self.height / self.tile_size / 2 - 1) * self.tile_size)]
        self.direction = None

        self._place_food()

        return self._get_state(), 0, self.done, self.score


    @abstractmethod
    def step(self, action: Direction):
        pass



    def render(self, screen):
        screen.fill((0, 0, 0))
        # Snake
        for head in self.snake:
            head_snake = pygame.Rect(head.x + 1, head.y + 1, self.tile_size - 1, self.tile_size - 1)
            pygame.draw.rect(screen, (0,190,0), head_snake)

        # Food
        if isinstance(self.food, list):
            for cookie in self.food:
                food_rect = pygame.Rect(cookie.x + 1, cookie.y + 1, self.tile_size - 1, self.tile_size - 1)
                pygame.draw.rect(screen, (250,0,0), food_rect)
        else:
            food_rect = pygame.Rect(self.food.x + 1, self.food.y + 1, self.tile_size - 1, self.tile_size - 1)
            pygame.draw.rect(screen, (250,0,0), food_rect)

        # Score
        text = pygame.font.SysFont('arial', 25).render("Score: " + str(self.score), True, (255,255,255))
        screen.blit(text, [0, 0])

        pygame.display.flip()


    def _check_collision(self, point: Point):
        return point in self.snake[:-1] or point.x < 0 or point.x > self.width - self.tile_size or point.y < 0 or point.y > self.height - self.tile_size
    

    @abstractmethod
    def _place_food(self):
        pass
