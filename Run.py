from Snake import Snake
from Population import Population
import numpy as np
from Visualise import visualise


def run(iterations):

    pop = Population(
        population_size=40,
        iterations=10,
        mutation_rate=0.01,
        tournament_type="round-robin",
        selection_proportion=0.2,
        keep_best=True,
        output_size=3,
        max_move_cycle=5,
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
    visualise(best_snake.images)
    #print("Final snake score on last run", best_snake.Length_of_snake)

    #return pop


run(iterations=10)
