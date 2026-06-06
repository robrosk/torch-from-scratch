"""
Neural network building blocks for the src package.

Exports the most commonly-used symbols for a clean import surface:
    from src.nn import DenseLayer, NeuralNetwork, ReLU, MeanSquaredError
"""

from .modules import (
    ActivationFunction,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Layer,
    DenseLayer,
    LossFunction,
    MeanSquaredError,
    CrossEntropyLoss,
)
from .neural_network import NeuralNetwork

__all__ = [
    "ActivationFunction",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Layer",
    "DenseLayer",
    "LossFunction",
    "MeanSquaredError",
    "CrossEntropyLoss",
    "neural_network",
]


