import numpy as np
from .topo import topo_reverse_sort

class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None                  # accumulated dL/d(self)
        self.requires_grad = requires_grad
        self._backward = lambda: None     # pushes grad to parents
        # NOTE: set dedup means x + x stores only ONE parent. When writing
        # _backward, distribute grads via the operands captured in the closure
        # (self, other) with += accumulation — never by iterating _parents.
        self._parents = set(_parents)     # who created me
        self._op = _op                    # label, for debugging/viz

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def __add__(self, other):  # returns a new Tensor + wires up _backward
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="+")
        # TODO(you): out._backward — grad flows unchanged to both operands;
        # if shapes were broadcast (e.g. (out,batch) + (out,1) bias), the
        # broadcast operand needs its grad summed back to its own shape.
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="@")
        # TODO(you): out._backward — dL/dA = G @ B.T, dL/dB = A.T @ G
        return out
    
    def backward(self):         # topo-sort parents, run _backward in reverse
        topo = topo_reverse_sort(self)

        for node in topo:
            node._backward()


def _assert_tensor(other):
    if not isinstance(other, Tensor):
        raise TypeError(f"Expected a Tensor, but got {type(other)}")
    


    