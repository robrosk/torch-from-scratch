import numpy as np
from math import prod

"""
axis=0 means "go down the rows" (operate column-wise)
- you collapse the row dimension, so you get one value per column.
axis=1 means "go across the columns" (operate row-wise)
- you collapse the column dimension, so you get one value per row.
"""

def _assert_ndarray(X):
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy.ndarray, got {type(X).__name__}")
    return np.asarray(X, dtype=np.float64)

def _count_along_axes(X, axis):
    if axis is None:
        return X.size
    axes = (axis,) if isinstance(axis, int) else axis
    return prod(X.shape[a] for a in axes)

def sample_mean(X, axis=None, keepdims=False):
    """
    Compute the sample mean of X.
    - axis=None: mean over all entries (scalar)
    - axis=int/tuple: mean over those axes
    - keepdims=True: keeps reduced axes as size-1 for broadcasting
    """
    X = _assert_ndarray(X)
    n = _count_along_axes(X, axis)
    return X.sum(axis=axis, keepdims=keepdims) / n

def center(X, axis=None):
    """
    Returns the centered version of the input array.
    """
    X = _assert_ndarray(X)
    return X - sample_mean(X, axis=axis, keepdims=True)

def sample_variance(X, axis=None, keepdims=False, correction=1):
    """
    Returns the sample variance of the input array.

    The unbiased sample variance when correction = 1, 
    the maximum likelihood estimate when correction = 0.
    """
    X = _assert_ndarray(X)
    n = _count_along_axes(X, axis)
    denom = n - correction
    if denom <= 0:
        raise ValueError(f"Need n > correction (got n={n}, correction={correction}).")

    centered = center(X, axis=axis)
    return (centered ** 2).sum(axis=axis, keepdims=keepdims) / denom 

def sample_std(X, axis=0, keepdims=False, correction=1, eps=0.0):
    """
    Returns the sample standard deviation of the input array.
    """
    X = _assert_ndarray(X)
    var = sample_variance(X, axis=axis, keepdims=keepdims, correction=correction)
    return np.sqrt(var + eps)

def standardize(X, axis=None, keepdims=False, eps=1e-8, correction=1):
    """
    Z-score standardization along `axis`:
      Z = (X - mean) / (std + eps)

    axis: dimension(s) to compute mean/std over (int or tuple)
    keepdims: keep reduced dims for broadcasting
    correction: passed to sample_std (0 for MLE, 1 for unbiased)
    """
    _assert_ndarray(X)
    mu = sample_mean(X, axis=axis, keepdims=True)
    sigma = sample_std(X, axis=axis, keepdims=True, correction=correction)
    return (X - mu) / (sigma + eps)