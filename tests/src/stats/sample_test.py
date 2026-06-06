import os
import importlib.util

import numpy as np


def _load_sample_module():
    """
    Load `src/mini_torch/stats/sample.py` directly (without importing the package),
    so unfinished modules in package __init__ files don't break test collection.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sample_path = os.path.join(repo_root, "src", "mini_torch", "stats", "sample.py")

    spec = importlib.util.spec_from_file_location("mini_torch_stats_sample", sample_path)
    assert spec and spec.loader, "Could not load sample.py module spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_mean_matches_numpy_random():
    sample = _load_sample_module()
    sample_mean = sample.sample_mean

    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5, 3))

    assert sample_mean(X) == np.mean(X)
    np.testing.assert_allclose(sample_mean(X, axis=0), np.mean(X, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        sample_mean(X, axis=(0, 2), keepdims=True),
        np.mean(X, axis=(0, 2), keepdims=True),
        rtol=0,
        atol=1e-12,
    )
