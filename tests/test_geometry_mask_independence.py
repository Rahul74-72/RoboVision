import numpy as np

from geometry_utils import normalize_geometry


def test_masked_feature_values_cannot_change_valid_feature_direction():
    std = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    baseline = normalize_geometry([2.0, 0.0, 4.0], [0.0, 0.0, 0.0], std)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        noisy = normalize_geometry(
            [2.0, 3e38, 4.0], [0.0, -3e38, 0.0], std
        )
    np.testing.assert_array_equal(noisy, baseline)
    assert noisy[1] == 0.0
    np.testing.assert_allclose(np.linalg.norm(noisy), 1.0, rtol=1e-6)
