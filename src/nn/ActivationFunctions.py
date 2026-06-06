from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def activate(self, inputs):
        raise NotImplementedError("Subclasses must implement activate()")

    def forward(self, inputs):
        return self.activate(inputs)


class ReLU(ActivationFunction):
    def activate(self, inputs):
        return np.maximum(np.zeros_like(inputs), inputs)


class LeakyReLU(ActivationFunction):
    def activate(self, inputs):
        return np.maximum(0.01 * inputs, inputs)


class Sigmoid(ActivationFunction):
    def activate(self, inputs):
        # numerically stable sigmoid
        out = np.empty_like(inputs)
        pos = inputs >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-inputs[pos]))
        exp_x = np.exp(inputs[neg])  # safe because inputs[neg] < 0
        out[neg] = exp_x / (1 + exp_x)
        return out


class Tanh(ActivationFunction):
    def activate(self, inputs):
        # numpy.tanh is more numerically stable than exp-based alternatives
        return np.tanh(inputs)


class Softmax(ActivationFunction):
    def activate(self, inputs):
        # softmax over classes (axis=0 aka along the columns)
        shifted = inputs - np.max(inputs, axis=0, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=0, keepdims=True)


