import numpy as np
from numba import jit


class Snake_computation(object):
    def __init__(
        self, game_size, l1_size, l2_size, input_type: str, max_snake_coords_input_size, output_size, network_dimensions, use_bias=True, output="softmax"
    ):

        self.game_size = game_size
        self.game_area_length = self.game_size * self.game_size
        self.area_repr = np.zeros(game_size * game_size)

        self.input_type = input_type
        self.output_size = output_size
        self.max_snake_coords_input_size = max_snake_coords_input_size
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.input_type= 'simple'

        self.layers = None
        self.biases = None

        self.use_bias = use_bias
        self.network_dimensions = network_dimensions
        self.output_func = 'softmax'

    #@jit(forceobj=True)
    def initialse_weights_and_biases(self):
        """
        Uses weight/bias initialisation heuristic from Glorot (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
        """
        self.layers = []
        self.biases = []
        self.output = self._activation(self.output_func)
        for i in range(len(self.network_dimensions) - 1):
            shape = (self.network_dimensions[i], self.network_dimensions[i + 1])
            std = np.sqrt(2 / sum(shape))
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1, self.network_dimensions[i + 1])) * self.use_bias
            self.layers.append(layer)
            self.biases.append(bias)

    #@jit(forceobj=True)
    def _activation(self, output):
        if output == "softmax":
            return lambda X: np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == "sigmoid":
            return lambda X: (1 / (1 + np.exp(-X)))
        if output == "linear":
            return lambda X: X
    
    @jit(forceobj=True)
    def predict(self, X):
        # if not X.ndim == 2:
        #     raise ValueError(f"Input has {X.ndim} dimensions, expected 2")
        X = X.reshape((1, self.layers[0].shape[0]))
        # if not X.shape[1] == self.layers[0].shape[0]:
        #     X = X.reshape((1, self.layers[0].shape[0]))
            #raise ValueError(f"Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}")
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output(X)  # output activation
            else:
                X = np.clip(X, 0, np.inf)  # ReLU
        return X

    @jit(forceobj=True)
    def predict_choice(self, X, deterministic=True):
        probabilities = self.predict(X)
        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))
        if any(np.sum(probabilities, axis=1) != 1):
            raise ValueError(f'Output values must sum to 1 to use deterministic=False')
        if any(probabilities < 0):
            raise ValueError(f'Output values cannot be negative to use deterministic=False')
        choices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            U = np.random.rand(X.shape[0])
            c = 0
            while U > probabilities[i, c]:
                U -= probabilities[i, c]
                c += 1
            else:
                choices[i] = c
        return choices.reshape((-1,1))

    # def initialise_weights(self):
    #     self.W1 = self.layers[0]
    #     self.W2 = self.layers[1]
    #     self.W3 = self.layers[2]


    # def initialise_biases(self):
        
    #     self.B1 = 
    #     self.B2 = 
    #     self.B3 = 

    # def multiply_biases(self):
    #     X = X @ layer + np.ones((X.shape[0], 1)) @ bias

    # def set_weights(self, W1, W2, W3):
    #     # self.W1
    #     # self.W2
    #     # self.W3
    #     pass

    # # @jit(forceobj=True)
    # def compute(self):

    #     A1 = self.input_vec
    #     A2 = ReLU(np.dot(self.W1, A1))
    #     A3 = ReLU(np.dot(self.W2, A2))
    #     A3 = np.transpose(A3)
    #     A4 = ReLU(np.dot(self.W3, A3))

    #     return int(np.argmax(A4))

    @jit(forceobj=True)
    def jit_convert2(self, rec_array=None):
        coord_map = lambda array: array[1] + array[2] * array[0]

        if rec_array is not None:
            return coord_map(rec_array.T)
        else:
            return coord_map(self.matrix_coords_of_snake_and_food.T).astype(np.int64)

    @jit(forceobj=True)
    def update_coords_jit(self):
        self.area_repr.fill(0)
        self.area_repr[self.jit_convert2()] = 1
        assert len(self.area_repr) == self.input_size - 4

    @jit(forceobj=True)
    def create_input_vector(self):

        # Add snake + food to array of coords to be converted to array idx
        self.matrix_coords_of_snake_and_food = np.vstack((self.snake_List.copy(), self.food))
        self.get_snake_matrix_coords()
        self.update_coords_jit()

        # Add boundary info to area_repr
        self.get_dists_to_boundary()

        self.input_vec = np.concatenate(
            (
                self.area_repr.copy(),
                np.array([self.dist_x_l, self.dist_x_r, self.dist_y_d, self.dist_y_u]),
            ),
            axis=0,
        )

    #@jit(forceobj=True)
    def create_simple_input_vector(self):

        # Get relevant_snake
        relevant_snake_list = self.snake_List.copy()[-self.max_snake_coords_input_size :]
        try:
            relevant_snake_list = relevant_snake_list.reshape(
                min(self.Length_of_snake, self.max_snake_coords_input_size), 2
            )
        except ValueError:
            print("Problem")

        for i in range(np.abs(self.max_snake_coords_input_size - len(relevant_snake_list))):
            relevant_snake_list = np.insert(relevant_snake_list, 0, np.zeros(2), axis=0)

        if relevant_snake_list.shape != (self.max_snake_coords_input_size, 2):
            relevant_snake_list.reshape(self.max_snake_coords_input_size, 2)

        # Add food
        coords_to_be_converted = np.vstack((relevant_snake_list, self.food))

        # Transform to array idx
        ready_for_conversion = self.get_matrix_coords_rec_array(rec_array=coords_to_be_converted)
        converted = self.jit_convert2(rec_array=ready_for_conversion)

        # Add boundary info
        input_vec = np.concatenate((converted, self.get_dists_to_boundary()), axis=0)
        input_vec = np.concatenate((input_vec, self.check_each_direction()), axis=0)

        self.input_vec = input_vec

    #@jit(forceobj=True)
    def get_matrix_coords_rec_array(self, rec_array):
        return np.concatenate((rec_array, np.ones((len(rec_array), 1)) * self.game_size), axis=1)

    #@jit(forceobj=True)
    def get_snake_matrix_coords(self):
        self.matrix_coords_of_snake_and_food = np.concatenate(
            (
                self.matrix_coords_of_snake_and_food,
                np.ones((len(self.matrix_coords_of_snake_and_food), 1)) * self.game_size,
            ),
            axis=1,
        )

    def assign_vector_creation_type(self):
        if self.input_type == "simple":

            self.create_simple_input_vector()
            self.get_vector = self.create_simple_input_vector
            self.input_size = len(self.input_vec)

        elif self.input_type == "complex":
            self.input_size = len(self.area_repr) + 4  # Boundary distances
            self.get_vector = self.create_input_vector()

        self.network_dimensions[0] = self.input_size


# Activation functions
def ReLU(x):
    return np.maximum(0, x)


def dReLU(self, x):
    return 1 * (x > 0)


def softmax(self, z):
    z = z - np.max(z, axis=1).reshape(z.shape[0], 1)
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(z.shape[0], 1)
