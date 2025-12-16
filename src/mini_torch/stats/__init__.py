"""Statistics utilities for mini_torch."""

from .sample import (
    sample_mean,
    center,
    sample_variance,
    sample_std,
    sample_cov,
    standardize,
)

from .covariance import cov, corrcoef

__all__ = [
    "sample_mean",
    "center",
    "sample_variance",
    "sample_std",
    "sample_cov",
    "standardize",
    "cov",
    "corrcoef",
]


