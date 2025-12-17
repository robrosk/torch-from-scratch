import numpy as np

from .sample import _assert_ndarray, _count_along_axes, sample_mean, center, sample_variance, sample_std

def cov(X, Y=None, *, rowvar=False, correction=1):
    """
    Sample covariance.

    If Y is None:
      - X is 1D -> returns Var(X) (scalar)
      - X is 2D -> returns covariance matrix among variables in X

    If Y is provided:
      - X and Y are 1D -> returns Cov(X, Y) (scalar)
      - X and Y are 2D -> returns cross-covariance matrix
    """
    X = _assert_ndarray(X)

    if Y is None:
        if X.ndim == 1:
            return sample_variance(X, axis=None, keepdims=False, correction=correction)

        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")

        # covariance matrix
        if rowvar:
            n = X.shape[1]                # observations
            Xc = center(X, axis=1)        # center each row-variable over columns
            denom = n - correction
            if denom <= 0:
                raise ValueError(f"Need n > correction (got n={n}, correction={correction}).")
            return (Xc @ Xc.T) / denom    # (d,d)
        else:
            n = X.shape[0]                # observations
            Xc = center(X, axis=0)        # center each col-variable over rows
            denom = n - correction
            if denom <= 0:
                raise ValueError(f"Need n > correction (got n={n}, correction={correction}).")
            return (Xc.T @ Xc) / denom    # (d,d)

    Y = _assert_ndarray(Y)

    if X.ndim != Y.ndim:
        raise ValueError(f"X and Y must have same ndim. X: {X.ndim}, Y: {Y.ndim}")

    # 1D: scalar covariance
    if X.ndim == 1:
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same length for 1D covariance.")
        n = X.shape[0]
        denom = n - correction
        if denom <= 0:
            raise ValueError(f"Need n > correction (got n={n}, correction={correction}).")
        x = center(X, axis=None)
        y = center(Y, axis=None)
        return (x @ y) / denom

    # 2D: cross-covariance
    if X.ndim == 2:
        if rowvar:
            if X.shape[1] != Y.shape[1]:
                raise ValueError("Rowvar=True: X and Y must share observation count (columns).")
            n = X.shape[1]
            Xc = center(X, axis=1)
            Yc = center(Y, axis=1)
            denom = n - correction
            if denom <= 0:
                raise ValueError(f"Need n > correction (got n={n}, correction={correction}).")
            return (Xc @ Yc.T) / denom
        else:
            if X.shape[0] != Y.shape[0]:
                raise ValueError("Rowvar=False: X and Y must share observation count (rows).")
            n = X.shape[0]
            Xc = center(X, axis=0)
            Yc = center(Y, axis=0)
            denom = n - correction
            if denom <= 0:
                raise ValueError(f"Need n > correction (got n={n}, correction={correction}).")
            return (Xc.T @ Yc) / denom

    raise ValueError(f"X must be 1D or 2D, got {X.ndim}D")

def corrcoef(X):
    """
    Returns the correlation coefficient matrix (Pearson correlations), i.e. covariance with units removed.
    - Output is a d x d matrix R with entries [-1, 1], and ones on the diagonal.
    - Conceptually, this is "normalize covariance by standard deviations," so it tells you 
    strength of linear relationship without being distorted by units.
    """
    raise NotImplementedError("corrcoef is not implemented yet")


