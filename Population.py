from copy import deepcopy

import numpy as np

from Snake import Snake


class Population(Snake):
    def __init__(
        self,
        population_size: int,
        iterations: int,
        mutation_rate: float,
        tournament_type: str,
        selection_proportion: float,
        keep_best: bool,
        output_size: int,
        max_move_cycle: int,
        network_dimensions: list,
        
    ):

        self.population_size = population_size
        self.selection_proportion = selection_proportion
        self.keep_best = keep_best
        self.output_size = output_size
        self.mutation_rate = mutation_rate
        self.max_move_cycle = max_move_cycle
        self.network_dimensions = network_dimensions


        self.Best_score = 0

        self.population = None
        self.BestSolution = None
        self.num_matrix_params = None
        self.num_idx_to_mutate = None
        self.parents_set = None
        self.idx_of_best = None
        self.population_scores = None


    def initialise_population(self):
        self.population = [
            Snake(
                game_size=20,
                l1_size=30,
                l2_size=30,
                n_food=4,
                input_type="simple",
                max_snake_coords_input_size=10,
                output_size=self.output_size,
                max_move_cycle=self.max_move_cycle,
                cheat=False,
                display_freq=0,
                network_dimensions=self.network_dimensions,
                mutation_rate=self.mutation_rate
            )
            for i in range(self.population_size)
        ]

    def get_num_idx_to_mutate(self):

        self.num_matrix_params = (
            len(self.population[0].W1.flatten())
            + len(self.population[0].W2.flatten())
            + len(self.population[0].W3.flatten())
        )

        self.num_idx_to_mutate = int(self.num_matrix_params * self.mutation_rate)

    def selection_loop(self):
        self.run_and_get_idx_of_best()
        self.get_parents()
        # if self.num_idx_to_mutate is None:
        #     self.get_num_idx_to_mutate()
        self.create_new_population()

    def run_and_get_idx_of_best(self):
        self.population_scores = [snake.gameLoop() for snake in self.population]
        if len(np.argsort(np.array(self.population_scores))[::-1]) > 1:
            self.idx_of_best = np.argsort(np.array(self.population_scores))[::-1][
                : int(self.selection_proportion * self.population_size)
            ]

            if self.population_scores[self.idx_of_best[0]] > self.Best_score:
                print("New highest score:", self.population_scores[self.idx_of_best[0]])
                self.BestSolution = self.population[self.idx_of_best[0]]
                self.Best_score = self.population_scores[self.idx_of_best[0]]
                self.print_move_dist_best_solution()
                
        else:
            self.idx_of_best = [i for i in range(10)]

    def create_new_population(self):

        new_population = []
        if self.keep_best is True:
            best = self.population[self.idx_of_best[0]]
            best = self.inherit_weights_from_best_sol()
            mutated_best = deepcopy(best)
            mutated_best.mutate()

            new_population.append(best)  # Ensure best organism survives unmutated + a mutated one
            new_population.append(mutated_best)

        while len(new_population) < self.population_size:
            parents = np.random.choice(self.parents_set, size=2, replace=True)
            parent1, parent2 = parents[0], parents[1]
            children = self.recombine_from_parents(parent1, parent2)
            child1, child2 = children[0], children[1]
            child1.mutate()
            child1.mutate()
            new_population.append(child1)
            new_population.append(child2)

        assert len(self.population) == len(new_population)

        self.population = new_population

    def get_parents(self):
        population_as_array = np.array(self.population.copy())
        self.parents_set = population_as_array[self.idx_of_best]

    def recombine_from_parents(self, parent1, parent2):
        """
        Create two children from each pair of parents. Child 1 inherits more weights from parent 1, Child 2 get's more from parent 2.
        TODO
        - Create children from vertical slices of weight matrices from each parent, rather than taking whole layers. 
        - 
        """
        children = [Snake(
            game_size=20,
            l1_size=30,
            l2_size=30,
            n_food=4,
            input_type="simple",
            max_snake_coords_input_size=10,
            output_size=self.output_size,
            max_move_cycle=self.max_move_cycle,
            cheat=False,
            display_freq=0,
            network_dimensions=self.network_dimensions,
            mutation_rate=self.mutation_rate
        ) for i in range(2)]

        for child in children:
            child.initialse_weights_and_biases()
            genes = np.random.choice([parent1, parent2], len(child.layers) + len(child.biases))
            for i in range(len(child.layers)):
                child.layers[i] = genes[i].layers[i]
                child.biases[i] = genes[i].biases[i]

        return children

    

    # def mutate(self, child):

    #     flat_weights = np.concatenate((child.W1.flatten(), child.W2.flatten(), child.W3.flatten()))
    #     # print("Len flat weights:", len(flat_weights))
    #     mutation_idx = np.random.randint(len(flat_weights), size=self.num_idx_to_mutate)
    #     flat_weights[mutation_idx] = np.random.uniform(low=-1, high=1.0, size=len(mutation_idx))

    #     w1_slice = child.W1.shape[0] * child.W1.shape[1]
    #     w2_slice = w1_slice + (child.W2.shape[0] * child.W2.shape[1])

    #     # print("Size w1", child.W1.shape, "total_entries:", len(child.W1.flatten()))
    #     # print("Size w2", child.W2.shape, "total_entries:", len(child.W2.flatten()))
    #     # print("Size w3", child.W3.shape, "total_entries:", len(child.W3.flatten()))

    #     w1 = flat_weights[:w1_slice]
    #     w2 = flat_weights[w1_slice:w2_slice]
    #     w3 = flat_weights[w2_slice:]

    #     # print("Length after flattening w1: ", len(w1))
    #     # print("Length after flattening w2: ", len(w2))
    #     # print("Length after flattening w3: ", len(w3))

    #     w1 = w1.reshape(child.W1.shape[0], child.W1.shape[1])
    #     w2 = w2.reshape(child.W2.shape[0], child.W2.shape[1])
    #     w3 = w3.reshape(child.W3.shape[0], child.W3.shape[1])

    #     child.W1 = w1
    #     child.W2 = w2
    #     child.W3 = w3

    def inherit_weights_from_best_sol(self):
        """
        Instead of using the same BestSolution object and manually resetting every changing attribute for the next selection round.
        """
        new_snake = Snake(
            game_size=20,
            l1_size=30,
            l2_size=30,
            n_food=4,
            input_type="simple",
            max_snake_coords_input_size=10,
            output_size=self.output_size,
            max_move_cycle=self.max_move_cycle,
            cheat=False,
            display_freq=0,
            network_dimensions=self.network_dimensions,
            mutation_rate=self.mutation_rate
        )

        new_snake.layers = self.BestSolution.layers.copy()
        new_snake.biases = self.BestSolution.biases.copy()

        return new_snake

    def select(self):
        pass

    def print_move_dist_best_solution(self):
        print("Move distribution of best solution:", self.BestSolution.move_dist)

    def enfore_structure(self):
        assert self.population == self.population_size

    def return_next_generation(self):
        pass
