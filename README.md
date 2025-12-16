## mini-torch (mini PyTorch, for learning)

This repo is my **living “from-scratch” reimplementation of a tiny subset of PyTorch**, built to document and reinforce my learning in machine learning over time. Expect frequent iteration: APIs may change, pieces may be incomplete, and implementations will steadily improve.

## What’s here right now

- **`mini_torch`**: a small Python package under `src/mini_torch/`
  - **`mini_torch.nn`**: core neural-network building blocks (layers, activations, losses, simple network wrapper)
  - **`mini_torch.math`**: placeholder for math utilities (will grow over time)

## Repository structure

```text
src/
  mini_torch/
    __init__.py
    nn/
      __init__.py
      ActivationFunctions.py
      Layers.py
      LossFunctions.py
      NeuralNetwork.py
    math/
      __init__.py
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