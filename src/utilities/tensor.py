import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None                  # accumulated dL/d(self)
        self.requires_grad = requires_grad
        self._backward = lambda: None     # pushes grad to parents
        self._parents = set(_parents)     # who created me
        self._op = _op                    # label, for debugging/viz

    def __add__(self, other):  # returns a new Tensor + wires up _backward
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="+")

        return out
    
    def __matmul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="@")

        return out
    
    def backward(self):         # topo-sort parents, run _backward in reverse
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)  # seed with dL/dself = 1
        for v in reversed(topo):
            v._backward()
