import numpy as np

from geometry_utils import normalize_geometry


def test_common_translation_of_geometry_and_mean_preserves_result():
    geometry = np.array([5.0, -3.0, 8.0], dtype=np.float32)
    mean = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    std = np.array([2.0, 0.0, 3.0], dtype=np.float32)
    offset = np.array([16.0, -8.0, 32.0], dtype=np.float32)
    expected = normalize_geometry(geometry, mean, std)
    actual = normalize_geometry(geometry + offset, mean + offset, std)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual[1] == 0.0


def test_reflecting_geometry_around_mean_reverses_direction():
    geometry = np.array([5.0, -3.0], dtype=np.float32)
    mean = np.array([1.0, 1.0], dtype=np.float32)
    std = np.array([2.0, 4.0], dtype=np.float32)
    expected = normalize_geometry(geometry, mean, std)
    actual = normalize_geometry(2 * mean - geometry, mean, std)
    np.testing.assert_allclose(actual, -expected, rtol=1e-6, atol=1e-7)
