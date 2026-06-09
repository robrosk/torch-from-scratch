from abc import ABC, abstractmethod

import numpy as np

from src.utilities import Tensor

from .activation import ActivationFunction


class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def update(self, learning_rate: float):
        raise NotImplementedError("Subclasses must implement update()")

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        raise NotImplementedError("Subclasses must implement parameters()")

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError("Subclasses must implement zero_grad()")

    @abstractmethod
    def get_weights(self) -> Tensor:
        raise NotImplementedError("Subclasses must implement get_weights()")

    @abstractmethod
    def get_biases(self) -> Tensor:
        raise NotImplementedError("Subclasses must implement get_biases()")


class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation_function: ActivationFunction):
        # W in output_size x input_size
        self.weights = Tensor(0.01 * np.random.randn(output_size, input_size), requires_grad=True)
        # b in output_size x 1
        self.biases = Tensor(np.zeros((output_size, 1)), requires_grad=True)
        self.activation_function = activation_function

    def forward(self, inputs: Tensor) -> Tensor:
        # NOTE: + self.biases broadcasts (out, batch) + (out, 1); its backward
        # must sum the bias grad over axis=1 (keepdims=True) to undo the broadcast.
        return self.activation_function.forward(self.weights @ inputs + self.biases)

    def update(self, learning_rate: float):
        # plain SGD step on .data — outside the graph, safe to mutate in place
        if self.weights.grad is not None:
            self.weights.data -= learning_rate * self.weights.grad
        if self.biases.grad is not None:
            self.biases.data -= learning_rate * self.biases.grad

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.biases]

    def zero_grad(self):
        self.weights.grad = None
        self.biases.grad = None

    def get_weights(self) -> Tensor:
        return self.weights

    def get_biases(self) -> Tensor:
        return self.biases
