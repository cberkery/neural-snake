import numpy as np


class Organism:
    def __init__(self, dimensions, use_bias=True, output="softmax"):
        self.layers = []
        self.biases = []
        self.use_bias = use_bias
        self.output = self._activation(output)
        for i in range(len(dimensions) - 1):
            shape = (dimensions[i], dimensions[i + 1])
            std = np.sqrt(2 / sum(shape))
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1, dimensions[i + 1])) * use_bias
            self.layers.append(layer)
            self.biases.append(bias)

    def _activation(self, output):
        if output == "softmax":
            return lambda X: np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == "sigmoid":
            return lambda X: (1 / (1 + np.exp(-X)))
        if output == "linear":
            return lambda X: X

    def predict(self, X):
        if not X.ndim == 2:
            raise ValueError(f"Input has {X.ndim} dimensions, expected 2")
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f"Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}")
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output(X)  # output activation
            else:
                X = np.clip(X, 0, np.inf)  # ReLU
        return X


organisms = []
for i in range(10):
    organisms.append(Organism(dimensions=[2, 10, 2]))
