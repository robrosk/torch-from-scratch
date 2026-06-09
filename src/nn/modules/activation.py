from abc import ABC, abstractmethod

from src.utilities import Tensor

from .. import functional as F


class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def activate(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement activate()")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activate(inputs)


class ReLU(ActivationFunction):
    def activate(self, inputs: Tensor) -> Tensor:
        return F.relu(inputs)


class LeakyReLU(ActivationFunction):
    def activate(self, inputs: Tensor) -> Tensor:
        return F.leaky_relu(inputs)


class Sigmoid(ActivationFunction):
    def activate(self, inputs: Tensor) -> Tensor:
        return F.sigmoid(inputs)


class Tanh(ActivationFunction):
    def activate(self, inputs: Tensor) -> Tensor:
        return F.tanh(inputs)


class Softmax(ActivationFunction):
    def activate(self, inputs: Tensor) -> Tensor:
        return F.softmax(inputs)
