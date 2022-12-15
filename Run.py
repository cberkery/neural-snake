from Snake import Snake
from Evolution import population
import numpy as np


def run(iterations):

    pop = population(
        population_size=100,
        iterations=10,
        mutation_rate=0.01,
        tournament_type="round-robin",
        selection_proportion=0.01,
        keep_best=True,
    )
    pop.initialise_population()

    for i in range(iterations):
        print("Starting iteration: ", i)
        pop.selection_loop()

    print("Running game with best snake")
    best_snake = pop.inherit_weights_from_best_sol()
    best_snake.is_final_snake = True
    best_snake.display_freq = 1
    best_snake.gameLoop()
    print('Final snake score on last run', best_snake.Length_of_snake)

    return pop


run(iterations=200)
