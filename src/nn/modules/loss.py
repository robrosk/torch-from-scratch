from abc import ABC, abstractmethod

from src.utilities import Tensor

from .. import functional as F


class LossFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement calculate_loss()")


class MeanSquaredError(LossFunction):
    def calculate_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true)


class CrossEntropyLoss(LossFunction):
    def calculate_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return F.cross_entropy(y_pred, y_true)
