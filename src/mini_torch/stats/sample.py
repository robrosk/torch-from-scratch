import numpy as np
from math import prod

"""
axis=0 means "go down the rows" (operate column-wise)
- you collapse the row dimension, so you get one value per column.
axis=1 means "go across the columns" (operate row-wise)
- you collapse the column dimension, so you get one value per row.
"""

def _assert_ndarray(X):
    assert isinstance(X, np.ndarray), f"X must be a numpy.ndarray, got {type(X).__name__}"
    return X

def sample_mean(X, axis=None, keepdims=False):
    """
    Compute the sample mean of X.
    - axis=None: mean over all entries (scalar)
    - axis=int/tuple: mean over those axes
    - keepdims=True: keeps reduced axes as size-1 for broadcasting
    """
    _assert_ndarray(X)
    if axis is None:
        n = X.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        n = prod(X.shape[a] for a in axes)

    X.astype(np.float64)
    
    return X.sum(axis=axis, keepdims=keepdims) / n
    

def center(X, axis=0):
    """
    Returns the centered version of the input array.
    """
    _assert_ndarray(X)
    return X - sample_mean(X, axis=axis, keepdims=True)

def sample_variance(X, axis=0, keepdims=False):
    """
    Returns the sample variance of the input array.
    """
    _assert_ndarray(X)
    raise NotImplementedError("sample_variance is not implemented yet")

def sample_std(X, axis=0, keepdims=False):
    """
    Returns the sample standard deviation of the input array.
    """
    _assert_ndarray(X)
    raise NotImplementedError("sample_std is not implemented yet")

def sample_cov(X, axis=0, keepdims=False):
    """
    Returns the sample covariance of the input array.
    """
    _assert_ndarray(X)
    # Placeholder: true covariance returns a matrix; keeping as TODO for now.
    raise NotImplementedError("sample_cov is not implemented yet")

def standardize(X):
    """
    Returns the standardized version of the input array.
    - z-score standardization: subtract the mean and divide by the standard deviation.
    """
    _assert_ndarray(X)
    mu = sample_mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, ddof=1, keepdims=True)
    return (X - mu) / sigma