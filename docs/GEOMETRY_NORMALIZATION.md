# Geometry normalization

RoboVision combines FaceNet embeddings with normalized facial-geometry features during recognition.

## Current behavior

The runtime uses `geometry_utils.normalize_geometry()` as the shared normalization boundary before comparing geometry with the stored prototypes.

A training set can contain a geometry feature with zero variance. The shared helper protects this case by skipping standard-deviation entries whose magnitude is at or below the numerical tolerance instead of dividing by zero.

## Input contract

The helper requires `geometry`, `mean`, and `std` to have identical shapes and to contain finite values. Invalid shapes or non-finite values raise a clear `ValueError` instead of allowing malformed data to enter recognition scoring.

## Output behavior

Valid dimensions are z-score normalized and then L2-normalized. Zero or near-zero variance dimensions contribute `0`. If no dimensions can be normalized, the helper returns a zero vector rather than `NaN` or infinity.

## Regression coverage

The lightweight geometry tests cover zero and near-zero standard deviation, all-invalid variance, shape mismatches, and non-finite inputs. The production recognition tests also verify that `3_robot.py` calls this shared helper.
