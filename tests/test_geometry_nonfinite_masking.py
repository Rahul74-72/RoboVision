import numpy as np
import pytest

from geometry_utils import normalize_geometry


@pytest.mark.parametrize("field", ["geometry", "mean", "std"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_rejected_even_in_masked_dimensions(field, invalid):
    inputs = {
        "geometry": np.array([1.0, 2.0], dtype=np.float32),
        "mean": np.array([0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 0.0], dtype=np.float32),
    }
    inputs[field][1] = invalid
    with pytest.raises(ValueError, match="finite"):
        normalize_geometry(**inputs)
