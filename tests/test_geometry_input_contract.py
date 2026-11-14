import numpy as np

from geometry_utils import normalize_geometry


def test_std_threshold_masks_boundary_but_accepts_next_float32_value():
    threshold = np.float32(1e-6)
    above = np.nextafter(threshold, np.float32(np.inf))
    result = normalize_geometry([1.0, 1.0], [0.0, 0.0], [threshold, above])
    np.testing.assert_array_equal(result, np.array([0.0, 1.0], dtype=np.float32))


def test_normalization_accepts_readonly_views_without_mutating_inputs():
    arrays = [
        np.array([3.0, 99.0, 5.0, 99.0], dtype=np.float32),
        np.array([1.0, 99.0, 1.0, 99.0], dtype=np.float32),
        np.array([1.0, 99.0, 2.0, 99.0], dtype=np.float32),
    ]
    originals = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False

    result = normalize_geometry(*(array[::2] for array in arrays))

    np.testing.assert_allclose(result, np.array([1.0, 1.0]) / np.sqrt(2.0))
    assert result.dtype == np.float32
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)
        assert not np.shares_memory(result, array)
