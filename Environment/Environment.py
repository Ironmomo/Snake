from collections import namedtuple
from enum import Enum
import random
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

class Environment():
  
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

        self.food: Point = self.__place_food()

    def reset(self):
        self.score = 0
        self.done = False

        self.snake = [Point((self.width / self.tile_size / 2 - 1) * self.tile_size, (self.height / self.tile_size / 2 - 1) * self.tile_size)]
        self.direction = None

        self.food = self.__place_food()

        return self.__get_state(), 0, self.done, self.score


    def step(self, action: Direction):
        reward = 0
        
        if action == None:
            action = self.direction
        else:
            self.direction = action
        
        head = self.snake[-1]

        if self.direction:
            # move
            head = Point(head.x + self.direction.value[0] * self.tile_size, head.y + self.direction.value[1] * self.tile_size)
            self.snake.append(head)
            self.snake.pop(0)

        # check collision
        self.done = self.__check_collision(head)
        if self.done:
            reward = -10
            
        # food
        if not self.done and head == self.food:
            self.score += 1
            reward = 10
            self.snake.append(head)
            self.food = self.__place_food()
        
        return self.__get_state(), reward, self.done, self.score



    def render(self, screen):
        screen.fill((0, 0, 0))
        # Snake
        for head in self.snake:
            head_snake = pygame.Rect(head.x + 1, head.y + 1, self.tile_size - 1, self.tile_size - 1)
            pygame.draw.rect(screen, (0,190,0), head_snake)

        # Food
        food_rect = pygame.Rect(self.food.x + 1, self.food.y + 1, self.tile_size - 1, self.tile_size - 1)
        pygame.draw.rect(screen, (250,0,0), food_rect)

        # Score
        text = pygame.font.SysFont('arial', 25).render("Score: " + str(self.score), True, (255,255,255))
        screen.blit(text, [0, 0])

        pygame.display.flip()


    def __check_collision(self, point: Point):
        return point in self.snake[:-1] or point.x < 0 or point.x > self.width - self.tile_size or point.y < 0 or point.y > self.height - self.tile_size

    
    def __get_state(self):
        if self.done:
            return None
        
        head = self.snake[-1]
        point_t = Point(head.x, head.y - 1 * self.tile_size)
        point_r = Point(head.x + 1 * self.tile_size, head.y)
        point_b = Point(head.x, head.y + 1 * self.tile_size)
        point_l = Point(head.x - 1 * self.tile_size, head.y)

        state = [
            # Danger Straight
            self.direction == Direction.UP and self.__check_collision(point_t) or
            self.direction == Direction.RIGHT and self.__check_collision(point_r) or
            self.direction == Direction.DOWN and self.__check_collision(point_b) or
            self.direction == Direction.LEFT and self.__check_collision(point_l),
            # Danger Right
            self.direction == Direction.UP and self.__check_collision(point_r) or
            self.direction == Direction.RIGHT and self.__check_collision(point_b) or
            self.direction == Direction.DOWN and self.__check_collision(point_l) or
            self.direction == Direction.LEFT and self.__check_collision(point_t),
            # Danger Left
            self.direction == Direction.UP and self.__check_collision(point_l) or
            self.direction == Direction.RIGHT and self.__check_collision(point_t) or
            self.direction == Direction.DOWN and self.__check_collision(point_r) or
            self.direction == Direction.LEFT and self.__check_collision(point_b),

            # Food Straight
            self.direction == Direction.UP and self.food.y < head.y or
            self.direction == Direction.RIGHT and self.food.x > head.x or
            self.direction == Direction.DOWN and self.food.y > head.y or
            self.direction == Direction.LEFT and self.food.x < head.x,
            # Food Right
            self.direction == Direction.UP and self.food.x > head.x or
            self.direction == Direction.RIGHT and self.food.y > head.y or
            self.direction == Direction.DOWN and self.food.x < head.x or
            self.direction == Direction.LEFT and self.food.y < head.y,
            #Food Left
            self.direction == Direction.UP and self.food.x < head.x or
            self.direction == Direction.RIGHT and self.food.y < head.y or
            self.direction == Direction.DOWN and self.food.x > head.x or
            self.direction == Direction.LEFT and self.food.y > head.y,

        ]

        return state
    

    def __place_food(self):
        while True:
            x = random.randint(0, (self.width - self.tile_size) // self.tile_size) * self.tile_size
            y = random.randint(0, (self.height - self.tile_size) // self.tile_size) * self.tile_size  
            new_food = Point(x, y)

            if new_food not in self.snake:
                break

        return new_food
