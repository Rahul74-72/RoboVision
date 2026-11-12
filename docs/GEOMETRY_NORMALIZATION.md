# Geometry normalization

RoboVision combines FaceNet embeddings with normalized facial-geometry features during recognition.

## Current behavior

The runtime uses `geometry_utils.normalize_geometry()` as the shared normalization boundary before comparing geometry with the stored prototypes. The helper converts `geometry`, `mean`, and `std` to NumPy `float32` arrays so downstream geometry scoring receives a consistent dtype.

A training set can contain a geometry feature with zero variance. The shared helper protects this case by skipping standard-deviation entries at or below `1e-6` instead of dividing by zero.

## Input contract

The helper requires `geometry`, `mean`, and `std` to be one-dimensional vectors with identical shapes and finite values. Standard deviations must also be non-negative. Invalid dimensions, mismatched shapes, non-finite values, or negative standard deviations raise a clear `ValueError` instead of allowing malformed data into recognition scoring.

## Output behavior

Dimensions with standard deviation above `1e-6` are z-score normalized; all other dimensions contribute `0`. Subtraction, division, and norm calculation use `float64` intermediates to prevent overflow for extreme finite `float32` inputs. The result is converted back to `float32`.

If the z-score vector's L2 norm is greater than `1e-8`, it is scaled to unit length. At or below that threshold, the z-score vector is returned without L2 scaling: tiny nonzero values are preserved, while an exactly zero vector remains zero.

## Regression coverage

The lightweight geometry tests cover zero and near-zero standard deviation, all-invalid variance, shape mismatches, non-finite inputs, and the `float32` output contract. `tests/test_geometry_extreme_values.py` additionally covers extreme finite values, masked variance with extreme inputs, and preservation of tiny nonzero vectors. The production recognition tests also verify that `3_robot.py` calls this shared helper.
