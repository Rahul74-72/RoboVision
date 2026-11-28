import pytest

from geometry_utils import normalize_geometry


@pytest.mark.parametrize("field", ["geometry", "mean", "std"])
def test_length_one_vectors_cannot_broadcast_across_features(field):
    inputs = {
        "geometry": [2.0, 4.0],
        "mean": [0.0, 0.0],
        "std": [1.0, 2.0],
    }
    inputs[field] = [1.0]
    with pytest.raises(ValueError, match="same shape"):
        normalize_geometry(**inputs)
