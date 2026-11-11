import numpy as np
import pytest

from geometry_utils import normalize_geometry


@pytest.mark.parametrize("std_value", [1.0, 2e-6])
def test_extreme_finite_geometry_preserves_direction_without_overflow(std_value):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = normalize_geometry(
            np.array([3e38, -3e38], dtype=np.float32),
            np.array([-3e38, 3e38], dtype=np.float32),
            np.full(2, std_value, dtype=np.float32),
        )
    expected = np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.allclose(result, expected)


def test_tiny_nonzero_geometry_is_preserved_below_normalization_threshold():
    geometry = np.array([1e-10, -1e-10], dtype=np.float32)
    result = normalize_geometry(geometry, np.zeros(2), np.ones(2))
    assert result.dtype == np.float32
    assert np.array_equal(result, geometry)


def test_extreme_geometry_with_masked_variance_returns_zero():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = normalize_geometry([3e38], [-3e38], [0.0])
    assert np.array_equal(result, np.zeros(1, dtype=np.float32))
