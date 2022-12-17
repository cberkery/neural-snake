from Snake_computation import Snake_computation
import numpy as np


class Snake_game(Snake_computation):
    def __init__(self, game_size, l1_size, l2_size, n_food, input_type, max_snake_coords_input_size, output_size):

        super().__init__(
            game_size,
            l1_size,
            l2_size,
            # n_food,
            input_type,
            max_snake_coords_input_size,
            output_size,
        )
        # self.n_food = n_food
        self.game_size = game_size

    def gameLoop(self):

        self.game_over = False
        self.game_close = False

        self.iteration = 0

        self.assign_vector_creation_type()
        self.create_food()
        self.initialise_weights()

        while self.game_over is not True:

            # self.create_food()
            if self.is_final_snake is True:
                print("final snake move count", self.move_counter)

            self.choose_move()

            # Move head of self in chosen direction
            self.update_location()

            # Check self hasn't made boundary violation
            self.check_boundary_violation()

            self.check_food_found()

            self.grow_or_shift()

            if self.move_counter > 800:
                self.game_over = True
                # print("Game went on for too many iterations")

            if self.is_final_snake is True:
                if self.game_over is False:
                    self.display()

        return self.Length_of_snake

    def check_boundary_violation(self):
        if (
            self.location[0] >= self.game_size
            or self.location[0] <= 0
            or self.location[1] >= self.game_size
            or self.location[1] <= 0
        ):
            # print("hit boundary")
            self.game_over = True

    def test_boundary_violation(self, x1_change, y1_change):  # Of a move
        test_x = self.x1 + x1_change
        test_y = self.y1 + y1_change

        if self.cheat is False:
            if test_x >= self.game_size or test_x < 0:
                # print("Snake hit the boundary")
                self.game_over = True

            if test_y >= self.game_size or test_y < 0:
                # print("Snake hit the boundary")
                self.game_over = True

        # Remove cheating feature soon
        if self.cheat is True:
            if test_x >= self.game_size or test_x < 0:
                x1_change = -x1_change

            if test_y >= self.game_size or test_y < 0:
                y1_change = -y1_change

        return x1_change, y1_change

    def create_food(self):
        # # remove found self.food
        # if self.n_food is None:
        #     print("n_food swas somehow set to None")
        #     self.n_food = 10

        if len(self.food) < self.n_food:
            for i in range(int(np.abs(self.n_food - len(self.food)))):
                new_food = np.random.randint(1, self.game_size - 1, size=(1, 2))
                self.food = np.vstack((self.food, np.random.randint(1, self.game_size - 1, size=2)))

        else:
            pass

    def display(self):

        if self.move_counter % self.display_freq == 0:
            print("displaying")
            display = [[" " for i in range(self.game_size)] for i in range(self.game_size)]
            snake_arr = self.snake_List.copy()

            for snack in self.food:
                # print('Snack ', snack)
                display[snack[0]][snack[1]] = "X"

            # if len(snake_arr) > 1:
            for coords in list(snake_arr):
                # print(row[0], row[1])
                # print('snake_arr[i]', snake_arr[i])
                display[coords[0]][coords[1]] = "0"

            # else:
            #     print(snake_arr[0],snake_arr[1])
            #     display[snake_arr[0][0]][snake_arr[0][1]] = "0"

            for row in display:
                print(row)
                print()

            print()
            print()
            print()
        else:
            pass
