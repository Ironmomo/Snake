from Environment.AbstractEnvironment import Environment, Direction, Point
import random

class SimpleEnvironment(Environment):
  
    def __init__(self, width: int = 1200, height: int = 1200, tile_size: int = 40):
        super().__init__(width, height, tile_size)
    
    def _get_state(self):
        if self.done:
            return None
        
        head = self.snake[-1]
        point_t = Point(head.x, head.y - 1 * self.tile_size)
        point_r = Point(head.x + 1 * self.tile_size, head.y)
        point_b = Point(head.x, head.y + 1 * self.tile_size)
        point_l = Point(head.x - 1 * self.tile_size, head.y)

        state = [
            # Danger Straight
            self.direction == Direction.UP and self._check_collision(point_t) or
            self.direction == Direction.RIGHT and self._check_collision(point_r) or
            self.direction == Direction.DOWN and self._check_collision(point_b) or
            self.direction == Direction.LEFT and self._check_collision(point_l),
            # Danger Right
            self.direction == Direction.UP and self._check_collision(point_r) or
            self.direction == Direction.RIGHT and self._check_collision(point_b) or
            self.direction == Direction.DOWN and self._check_collision(point_l) or
            self.direction == Direction.LEFT and self._check_collision(point_t),
            # Danger Left
            self.direction == Direction.UP and self._check_collision(point_l) or
            self.direction == Direction.RIGHT and self._check_collision(point_t) or
            self.direction == Direction.DOWN and self._check_collision(point_r) or
            self.direction == Direction.LEFT and self._check_collision(point_b),

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
        self.done = self._check_collision(head)
        if self.done:
            reward = -10
            
        # food
        if not self.done and head == self.food:
            self.score += 1
            reward = 10
            self.snake.append(head)
            self._place_food()
        
        return self._get_state(), reward, self.done, self.score


    def _place_food(self):
        while True:
                x = random.randint(0, (self.width - self.tile_size) // self.tile_size) * self.tile_size
                y = random.randint(0, (self.height - self.tile_size) // self.tile_size) * self.tile_size  
                new_food = Point(x, y)

                if new_food not in self.snake:
                    self.food = new_food
                    break