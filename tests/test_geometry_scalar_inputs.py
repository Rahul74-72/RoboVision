import pytest

from geometry_utils import normalize_geometry


@pytest.mark.parametrize("field", ["geometry", "mean", "std"])
def test_scalar_inputs_are_rejected_instead_of_broadcast(field):
    inputs = {"geometry": [2.0], "mean": [0.0], "std": [1.0]}
    inputs[field] = 1.0
    with pytest.raises(ValueError, match="1-D vectors"):
        normalize_geometry(**inputs)
