import numpy as np

from geometry_utils import normalize_geometry


def test_feature_permutation_preserves_normalization_and_variance_mask():
    geometry = np.array([5.0, 9.0, -3.0, 4.0], dtype=np.float32)
    mean = np.array([1.0, 2.0, 1.0, 0.0], dtype=np.float32)
    std = np.array([2.0, 0.0, 4.0, 1.0], dtype=np.float32)
    order = np.array([2, 0, 3, 1])
    baseline = normalize_geometry(geometry, mean, std)
    permuted = normalize_geometry(geometry[order], mean[order], std[order])
    np.testing.assert_allclose(permuted, baseline[order], rtol=1e-6, atol=1e-7)
    assert permuted[-1] == 0.0
    assert permuted.dtype == np.float32
