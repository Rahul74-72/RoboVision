import numpy as np

from geometry_utils import normalize_geometry


def test_normalize_geometry_handles_extreme_finite_values_without_overflow():
    geometry = np.array([3e38, -3e38], dtype=np.float32)
    mean = np.array([-3e38, 3e38], dtype=np.float32)
    std = np.ones(2, dtype=np.float32)

    result = normalize_geometry(geometry, mean, std)

    expected = np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.allclose(result, expected)
