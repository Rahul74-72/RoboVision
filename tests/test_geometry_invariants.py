import numpy as np

from geometry_utils import normalize_geometry


def test_empty_geometry_vectors_return_empty_float32_result():
    result = normalize_geometry([], [], [])
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_normalized_direction_is_invariant_to_positive_common_std_scaling():
    geometry = np.array([4.0, -5.0, 8.0], dtype=np.float32)
    mean = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    std = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    baseline = normalize_geometry(geometry, mean, std)
    scaled = normalize_geometry(geometry, mean, std * 8.0)
    np.testing.assert_allclose(scaled, baseline, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(scaled), 1.0, rtol=1e-6)
