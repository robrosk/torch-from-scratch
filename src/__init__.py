"""
Public API is re-exported from subpackages for convenient imports:
    from src.nn import NeuralNetwork, DenseLayer, ReLU
"""

from . import linalg, nn, stats

__all__ = ["nn", "stats", "linalg"]


