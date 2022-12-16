import numpy as np

from CLargs import CLargs
from Population import Population
from Snake import Snake

population_size, iterations, mutation_rate, selection_proportion = CLargs().return_args()


def run(iterations):

    pop = Population(
        population_size=population_size,
        iterations=iterations,
        mutation_rate=mutation_rate,
        tournament_type="round-robin",
        selection_proportion=selection_proportion,
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
    print("Final snake score on last run", best_snake.Length_of_snake)

    return pop


run(iterations=iterations)
