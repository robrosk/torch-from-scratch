import numpy as np

from .sample import _assert_ndarray, _count_along_axes, sample_mean, center, sample_variance, sample_std

def cov(X, Y=None, *,  rowvar=True, correction=1):
    """
    Sample covariance.

    If Y is None:
        - X is 1D -> returns Var(x)
        - X is 2D -> returns covariance matrix of the variables in X
    if Y is provided:
        - X and Y are 1D -> returns Cov(X, Y) (scalar)
        
    """
    X = _assert_ndarray(X)

    if Y is None:
        if X.ndim == 1:
            return sample_variance(X, axis=None, keepdims=False, correction=correction)
        elif X.ndim == 2:
            return sample_variance(X, axis=1, keepdims=True, correction=correction)
        else:
            raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")

    Y = _assert_ndarray(Y)

    pass

def corrcoef(X):
    """
    Returns the correlation coefficient matrix (Pearson correlations), i.e. covariance with units removed.
    - Output is a d x d matrix R with entries [-1, 1], and ones on the diagonal.
    - Conceptually, this is "normalize covariance by standard deviations," so it tells you 
    strength of linear relationship without being distorted by units.
    """
    raise NotImplementedError("corrcoef is not implemented yet")


