# Geometry normalization

RoboVision combines FaceNet embeddings with normalized facial-geometry features during recognition.

## Current behavior

The runtime uses `geometry_utils.normalize_geometry()` as the shared normalization boundary before comparing geometry with the stored prototypes. The helper converts `geometry`, `mean`, and `std` to NumPy `float32` arrays so downstream geometry scoring receives a consistent dtype.

A training set can contain a geometry feature with zero variance. The shared helper protects this case by skipping standard-deviation entries at or below `1e-6` instead of dividing by zero.

## Input contract

The helper requires `geometry`, `mean`, and `std` to be one-dimensional vectors with identical shapes and finite values. Standard deviations must also be non-negative. Invalid dimensions, mismatched shapes, non-finite values, or negative standard deviations raise a clear `ValueError` instead of allowing malformed data into recognition scoring.

## Output behavior

Valid dimensions are z-score normalized and then L2-normalized. Zero or near-zero variance dimensions contribute `0`. If the normalized vector has a near-zero norm, the helper returns the zero vector rather than `NaN` or infinity. The returned vector uses `float32` dtype, including for zero-vector results.

## Regression coverage

The lightweight geometry tests cover zero and near-zero standard deviation, all-invalid variance, shape mismatches, non-finite inputs, and the `float32` output contract. The production recognition tests also verify that `3_robot.py` calls this shared helper.
