## torch-from-scratch (mini PyTorch, for learning)

This repo is my **living “from-scratch” reimplementation of a tiny subset of PyTorch**, built to document and reinforce my learning in machine learning over time. Expect frequent iteration: APIs may change, pieces may be incomplete, and implementations will steadily improve.

## What’s here right now

- **`src`**: a small Python package at `src/`
  - **`src.nn`**: core neural-network building blocks (layers, activations, losses, simple network wrapper)
  - **`src.linalg`**: linear-algebra utilities (matrix decompositions, PSD helpers)
  - **`src.stats`**: sample statistics (mean, variance, covariance, correlation)
  - **`src.calculus`**: calculus utilities — Jacobian, Hessian, etc. (work in progress)
  - **`src.autograd`**: automatic differentiation engine (planned)
  - **`src.graph`**: computation-graph primitives backing autograd (work in progress)

## Repository structure

```text
src/
  __init__.py
  nn/
    __init__.py
    neural_network.py        # composes layers into a trainable network
    modules/                 # building blocks, grouped by family (mirrors torch.nn.modules)
      __init__.py
      activation.py          # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
      layers.py              # Layer, DenseLayer
      loss.py                # MeanSquaredError, CrossEntropyLoss
  linalg/
    __init__.py
    decomp.py
    psd.py
  stats/
    __init__.py
    sample.py
    covariance.py
  calculus/                  # calculus utilities — Jacobian, Hessian (work in progress)
    __init__.py
    jacobian.py
    hessian.py
  autograd/                  # automatic differentiation engine (planned)
    __init__.py
  graph/                     # computation graph backing autograd (work in progress)
    __init__.py
    graph.py
    modules/
      nodes.py
      edges.py
```

## Roadmap (informal)

- **Autograd** (tensors with gradients, graph, backprop)
- **More layers** (Dropout, BatchNorm, Conv, etc.)
- **Optimizers** (SGD, Adam, weight decay)
- **Losses + metrics** (stable cross-entropy, accuracy, etc.)
- **Tests and examples** (small training scripts, notebooks)
- **Packaging** (editable install, versioning, clearer API guarantees)

## Notes

- This is a **learning project**, not a production-ready framework.
- I’m intentionally prioritizing clarity and iteration over completeness.