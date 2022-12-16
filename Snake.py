import numpy as np

from Snake_game import Snake_game


class Snake(Snake_game):
    def __init__(
        self,
        game_size,
        l1_size,
        l2_size,
        n_food,
        input_type,
        max_snake_coords_input_size,
        cheat: bool,
        display_freq: bool,
    ):
        super().__init__(
            game_size,
            l1_size,
            l2_size,
            n_food,
            input_type,
            max_snake_coords_input_size,
        )

        self.x1 = int(self.game_size / 2)
        self.y1 = int(self.game_size / 2)

        self.n_food = n_food
        self.food = np.random.randint(1, self.game_size - 1, size=(n_food, 2))

        self.snake_List = np.array([self.x1, self.y1])
        self.Length_of_snake = 1
        self.last_move_x = [0]
        self.last_move_y = [0]

        self.UP = (self.y1 + 1,)
        self.DOWN = (self.y1 - 1,)
        self.RIGHT = (self.x1 + 1,)
        self.LEFT = self.x1 - 1

        self.move_counter = 0
        self.matrix_coords_of_snake_and_food = np.vstack((self.snake_List.copy(), self.food))
        self.cheat = cheat
        self.display_freq = display_freq

        self.game_over = False
        self.is_final_snake = None
        self.move_list = []
        self.move_dist = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
        }

    def run(self):
        self.gameLoop()

    def update_location(self):
        self.x1 += self.last_move_x[-1]
        self.y1 += self.last_move_y[-1]

        self.snake_List = np.vstack((self.snake_List, np.array([self.x1, self.y1])))

    def grow_or_shift(self):
        # Remove first block so self doesn't grow infinitely
        if len(self.snake_List) > self.Length_of_snake:
            self.snake_List = np.delete(self.snake_List, 0, axis=0)

    def check_body_collision(self):
        if len(self.snake_List) > 5:
            for x in self.snake_List[:-1]:
                if np.isin(self.snake_List[:-1], x).sum() > 2:
                    self.game_close = True

    def check_food_found(self):
        if np.isin(self.food, np.array([self.x1, self.y1])).sum() > 2:
            self.create_food()
            self.Length_of_snake += 1

    def choose_move(self):

        if self.move_counter % 20 != 0:

            self.get_vector()
            move = self.compute()

        else:
            # Cyclical_check
            if np.abs(np.max(self.last_move_x[-5:]) - np.min(self.last_move_x[-5:])) < 2:

                if np.abs(np.max(self.last_move_y[-5:]) - np.min(self.last_move_y[-5:])) < 2:

                    move = np.random.randint(5)

                else:
                    self.get_vector()
                    move = self.compute()

            else:
                self.get_vector()
                move = self.compute()

        self.move_counter += 1
        # LEFT
        if move == 0:
            x1_change = -1
            y1_change = 0

        # RIGHT
        elif move == 1:
            x1_change = 1
            y1_change = 0

        # UP
        elif move == 2:
            y1_change = -1
            x1_change = 0

            # DOWN
        elif move == 3:
            y1_change = 1
            x1_change = 0

        # CONTINUE
        elif move == 4:
            y1_change = self.last_move_y[-1]
            x1_change = self.last_move_x[-1]

        if self.cheat is True:
            x1_change, y1_change = self.test_boundary_violation(x1_change, y1_change)

        self.move_list.append(move)
        self.move_dist[str(move)] += 1

        self.last_move_x.append(x1_change)
        self.last_move_y.append(y1_change)

    def get_dists_to_boundary(self):
        self.dist_x_r = (self.game_size - self.x1) / self.game_size
        self.dist_x_l = (self.x1) / self.game_size
        self.dist_y_u = (self.y1) / self.game_size
        self.dist_y_d = (self.game_size - self.y1) / self.game_size

        return np.array([self.dist_x_r, self.dist_x_l, self.dist_y_u, self.dist_y_d])

    def check_each_direction(self):
        UP, DOWN, RIGHT, LEFT = self.y1 + 1, self.y1 - 1, self.x1 + 1, self.x1 - 1

        directions = [UP, DOWN, RIGHT, LEFT]

        clear = []
        for i in directions:
            if i >= self.game_size or i <= 0:
                clear.append(0)
            else:
                clear.append(1)

        return np.array(clear)

    # def heuristics(self):

    #     clearness_thresh = 1 #
    #     if self.

    #     heuristics_list = []

    #     # Is it clear straight ahead

    #     # Is it clear to the left

    #     # Is it clear to the right

    #     # Is food straight ahead

    #     # Is food to the right

    #     # Is food to the left

    #     self.heuristics_array = np.array()
