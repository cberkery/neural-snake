from Snake import Snake
from Population import Population
import numpy as np
from Visualise import visualise

from CLargs import CLargs
from Population import Population
from Snake import Snake

population_size, iterations, mutation_rate, selection_proportion = CLargs().return_args()


def run(iterations):
    """
    TODO
    -Input size is a placeholder string until it is determined on initialisation, since other input parmeters
    affect it's length, and I'm too lazy to work out a formula for it now.
    - String get's overwritten in Snake_computation.assign_vector_creation_type().
    """ 
    pop = Population(
        population_size=100,
        iterations=iterations,
        mutation_rate=0.03,
        tournament_type="round-robin",
        selection_proportion=0.1,
        keep_best=True,
        output_size=3,
        max_move_cycle=30,
        network_dimensions=[
            "Input_size",
            10,
            10,
            3,
        ],
    )
    pop.initialise_population()

    for i in range(iterations):
        print("Starting iteration: ", i)
        pop.selection_loop()

        if i % 10 == 0:

            print("Running game with best snake")
            best_snake = pop.inherit_weights_from_best_sol()
            best_snake.is_final_snake = True
            best_snake.display_freq = 1
            best_snake.gameLoop()
            visualise(best_snake.images)
            print("Final snake score on last run", best_snake.Length_of_snake)

    #return pop


run(iterations=50)
