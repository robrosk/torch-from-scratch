from abc import ABC, abstractmethod

import numpy as np

from .ActivationFunctions import ActivationFunction


class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def backward(self, inputs: np.ndarray, output_gradients: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement backward()")

    @abstractmethod
    def update(self, learning_rate: float):
        raise NotImplementedError("Subclasses must implement update()")

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_weights()")

    @abstractmethod
    def get_biases(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_biases()")


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
        # W in output_size x input_size
        self.weights = 0.01 * np.random.randn(output_size, input_size)
        # b in output_size x 1
        self.biases = np.zeros((output_size, 1))
        # dL/dW in output_size x input_size
        self.weights_gradient = np.zeros_like(self.weights)
        # dL/db in output_size x 1
        self.biases_gradient = np.zeros_like(self.biases)
        self.activation_function = activation_function

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation_function.forward(np.dot(self.weights, inputs) + self.biases)

    def backward(self, inputs: np.ndarray, output_gradients: np.ndarray) -> np.ndarray:
        return 0

    def update(self, learning_rate: float):
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_biases(self) -> np.ndarray:
        return self.biases


