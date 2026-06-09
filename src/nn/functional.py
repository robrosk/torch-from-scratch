"""
Functional ops on Tensors (mirrors torch.nn.functional).

Each function is a single primitive graph node: forward is computed with
numpy on .data, and _parents/_op/requires_grad are wired so the graph is
traversable. The _backward closures are left as TODOs.
"""

import numpy as np

from src.utilities import Tensor


def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad, _parents=(x,), _op="relu")
    # TODO(you): out._backward — grad passes where x.data > 0, else 0
    return out


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    out = Tensor(np.maximum(alpha * x.data, x.data), requires_grad=x.requires_grad, _parents=(x,), _op="leaky_relu")
    # TODO(you): out._backward — grad * 1 where x.data > 0, else grad * alpha
    return out


def sigmoid(x: Tensor) -> Tensor:
    # numerically stable sigmoid
    data = np.empty_like(x.data)
    pos = x.data >= 0
    neg = ~pos
    data[pos] = 1 / (1 + np.exp(-x.data[pos]))
    exp_x = np.exp(x.data[neg])  # safe because x.data[neg] < 0
    data[neg] = exp_x / (1 + exp_x)
    out = Tensor(data, requires_grad=x.requires_grad, _parents=(x,), _op="sigmoid")
    # TODO(you): out._backward — d/dx sigmoid = s * (1 - s), reuse out.data
    return out


def tanh(x: Tensor) -> Tensor:
    out = Tensor(np.tanh(x.data), requires_grad=x.requires_grad, _parents=(x,), _op="tanh")
    # TODO(you): out._backward — d/dx tanh = 1 - tanh(x)^2, reuse out.data
    return out


def softmax(x: Tensor) -> Tensor:
    # softmax over classes (axis=0 aka along the columns)
    shifted = x.data - np.max(x.data, axis=0, keepdims=True)
    exps = np.exp(shifted)
    data = exps / np.sum(exps, axis=0, keepdims=True)
    out = Tensor(data, requires_grad=x.requires_grad, _parents=(x,), _op="softmax")
    # TODO(you): out._backward — per column: J = diag(s) - s s^T, so
    # dL/dx = s * (g - sum(g * s, axis=0, keepdims=True)), reuse out.data
    return out


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    out = Tensor(
        np.mean(np.square(y_true.data - y_pred.data)),
        requires_grad=y_pred.requires_grad or y_true.requires_grad,
        _parents=(y_pred, y_true),
        _op="mse",
    )
    # TODO(you): out._backward — dL/dy_pred = 2 * (y_pred - y_true) / N
    return out


def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    out = Tensor(
        -np.sum(y_true.data * np.log(y_pred.data) + (1 - y_true.data) * np.log(1 - y_pred.data)),
        requires_grad=y_pred.requires_grad or y_true.requires_grad,
        _parents=(y_pred, y_true),
        _op="ce",
    )
    # TODO(you): out._backward — dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
    return out
