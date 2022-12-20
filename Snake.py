import numpy as np
from numba import jit
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
        output_size: int,
        max_move_cycle: int,
    ):
        super().__init__(
            game_size,
            l1_size,
            l2_size,
            n_food,
            input_type,
            max_snake_coords_input_size,
            output_size,
            # max_move_cycle
        )

        # self.x1 = int(self.game_size / 2)
        # self.y1 = int(self.game_size / 2)

        self.location = np.array([int(self.game_size / 2), int(self.game_size / 2)])
        self.direction = np.array([1, 0])
        self.past_directions = self.direction
        self.max_move_cycle = max_move_cycle

        self.n_food = n_food
        self.food = np.random.randint(1, self.game_size - 1, size=(n_food, 2))

        self.snake_List = self.location
        self.Length_of_snake = 1

        ## Check where this is used and replace!!
        # self.last_move_x = [0]
        # self.last_move_y = [0]

        self.move_counter = 0
        self.matrix_coords_of_snake_and_food = np.vstack((self.snake_List.copy(), self.food))
        self.cheat = cheat
        self.display_freq = display_freq

        self.game_over = False
        self.is_final_snake = None
        self.moves = []
        self.move_dist = {
            "left": 0,
            "right": 0,
            "continue": 0,
        }

    def run(self):
        self.gameLoop()

    def update_location(self):
        self.location = self.location + self.direction
        self.snake_List = np.vstack((self.snake_List, self.location))
        self.past_directions = np.vstack((self.past_directions, self.direction))

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
        for i in range(len(self.food)):
            if (self.food[i] == self.location).sum() == 2:
                self.food = np.delete(self.food, i, axis=0)
                self.create_food()
                self.Length_of_snake += 1
            else:
                pass

    def choose_move(self):

        self.get_vector()
        self.move = self.compute()
        
        self.move_counter += 1

        # Check if snake is going around in circles burning silicon
        if self.move_counter % 5 == 0:
            #print('Checking cyclical behaviour move: {}'.format(self.move_counter))
            self.check_and_correct_cyclical_behaviour()

        self.moves.append(self.move)

        # Change move if it will result in a boundary collision
        if self.cheat is True:
            self.check_each_move_for_boundary_violation()

        # LEFT
        if self.move == 0:
            self.left_rotation()
            self.move_dist["left"] += 1

        # RIGHT
        elif self.move == 1:
            self.right_rotation()
            self.move_dist["right"] += 1

        # CONTINUE
        elif move == 4:
            y1_change = self.last_move_y[-1]
            x1_change = self.last_move_x[-1]

    @jit(forceobj=True)
    def right_rotation(self):
        self.direction = np.dot(self.direction, np.array([[0, 1], [-1, 0]]))

    @jit(forceobj=True)
    def left_rotation(self):
        self.direction = np.dot(self.direction, np.array([[0, -1], [1, 0]]))

    def get_dists_to_boundary(self):
        self.dist_x_r = self.game_size - self.location[0]
        self.dist_x_l = self.location[0]
        self.dist_y_u = self.location[1]
        self.dist_y_d = self.game_size - self.location[1]

        return np.array([self.dist_x_r, self.dist_x_l, self.dist_y_u, self.dist_y_d])

    def check_each_direction(self):
        UP, DOWN, RIGHT, LEFT = self.location[1] + 1, self.location[1] - 1, self.location[0] + 1, self.location[0] - 1

        directions = [UP, DOWN, RIGHT, LEFT]

        clear = []
        for i in directions:
            if i >= self.game_size or i <= 0:
                clear.append(0)
            else:
                clear.append(1)

        return np.array(clear)

    def check_each_move_for_boundary_violation(self):
        """
        Checks possible next moves for a boundary violation in x and y directions.
        bad_x and bad_y contain move numbers which would lead to game over, otherwise are empty.
        [0, 1, 2] -> (LEFT, RIGHT, CONTINUE)
        For Example:
        - If bad_x contains 0, then going left will result in a boundary colission and game over
        - If bad_y contains [0, 1, 2], i.e the set of all possible moves, then you're fucked. Thankfully corners only have two sides so this is impossible.
        Usage: For cheating - can prevent snake from ever making a boundary collision by changing the chosen move if bad_x or bad_y are ever non-empty.

        """
        loc = self.location.copy()
        possible_moves = [0, 1, 2]

        # Sets of possible next x and y locations after a move
        test_x = np.array(
            [
                (loc + current_direction)[0],
                (loc + np.dot(current_direction, left_rotation))[0],
                (loc + np.dot(current_direction, right_rotation))[0],
            ]
        )
        test_y = np.array(
            [
                (loc + current_direction)[1],
                (loc + np.dot(current_direction, left_rotation))[1],
                (loc + np.dot(current_direction, right_rotation))[1],
            ]
        )

        test_boundary_violation = lambda x: True if x < 0 or x > self.game_size else False

        bad_x = np.where(np.array(list(map(test_bnd, test_x))) == True)[0]
        bad_y = np.where(np.array(list(map(test_bnd, test_y))) == True)[0]

        if len(bad_x) > 0:
            possible_moves.remove(bad_x[0])
            if len(bad_y) > 0:
                possible_moves.remove(bad_y[0])

        self.move = np.random.choice(possible_moves)

    def check_and_correct_cyclical_behaviour(self):
        possible_moves = [0, 1, 2]

        if np.all(np.array(self.moves[-self.max_move_cycle :]) == self.move) == True:
            possible_moves.remove(self.move)
            chose_other_move = np.random.choice(possible_moves)
            #print('Snake was choosing same move every time, changing from {} to {}'.format(self.move, chose_other_move))

            self.move = chose_other_move

    def log_and_return_images(self):
        if self.move_counter == 1:
            self.images = []

        display = np.zeros((self.game_size+1, self.game_size+1))
        snake_arr = self.snake_List.copy()

        for snack in self.food:
            display[snack[0]][snack[1]] = "255"

        # if len(snake_arr) > 1:
        for coords in list(snake_arr):
            display[coords[0]][coords[1]] = "100"

        self.images.append(display)

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
