import numpy as np
import pytest

from geometry_utils import normalize_geometry


def test_normalize_geometry_ignores_zero_standard_deviation():
    geometry = np.array([3.0, 5.0, 9.0], dtype=np.float32)
    mean = np.array([1.0, 5.0, 1.0], dtype=np.float32)
    std = np.array([1.0, 0.0, 2.0], dtype=np.float32)

    result = normalize_geometry(geometry, mean, std)

    assert np.isfinite(result).all()
    assert result[1] == 0.0
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_normalize_geometry_returns_zero_for_no_valid_variance():
    result = normalize_geometry(
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([1.0, 3.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )

    assert np.array_equal(result, np.zeros(2, dtype=np.float32))


def test_normalize_geometry_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        normalize_geometry(
            np.array([2.0, 4.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        )


def test_normalize_geometry_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        normalize_geometry(
            np.array([2.0, np.nan], dtype=np.float32),
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        )
