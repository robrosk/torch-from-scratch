import numpy as np

def cov(X, correction=1, rowvar=True):
    """
    Returns the covariance matrix of the variables in X.

    Paramters:
        - X: array_like
            A 1-D or 2-D array containing multiple variables and observations.
            Each row of X represents a variable, and each column represents an
            observation of all variables.
        - correction: int, optional
            - the degrees-of-freedom correction in the denominator. The adjustment factor. The default is 1.
        - rowvar: bool, optional
            If rowvar is True (default), then treat rows as variables and columns as observations;
            otherwise, treat columns as variables and rows as observations.
    Returns:
    out: ndarray
        The covariance matrix of the variables.
        If rowvar is True, the covariance matrix is (N,N), where N is
        the number of variables.
        If rowvar is False, the covariance matrix is (N,N), where N is
        the number of observations.
    """
    raise NotImplementedError("cov is not implemented yet")


def corrcoef(X):
    """
    Returns the correlation coefficient matrix (Pearson correlations), i.e. covariance with units removed.
    - Output is a d x d matrix R with entries [-1, 1], and ones on the diagonal.
    - Conceptually, this is "normalize covariance by standard deviations," so it tells you 
    strength of linear relationship without being distorted by units.
    """
    raise NotImplementedError("corrcoef is not implemented yet")


