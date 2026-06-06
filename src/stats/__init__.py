"""Statistics utilities for mini_torch."""

from .sample import (
    sample_mean,
    center,
    sample_variance,
    sample_std,
    standardize,
    _assert_ndarray,
    _count_along_axes,
)

from .covariance import cov, corrcoef

__all__ = [
    "sample_mean",
    "center",
    "sample_variance",
    "sample_std",
    "standardize",
    "_assert_ndarray",
    "_count_along_axes",
    "cov",
    "corrcoef",
]




