# Geometry normalization

RoboVision combines FaceNet embeddings with normalized facial-geometry features during recognition.

## Current behavior

The runtime currently standardizes geometry with the stored `geometry_mean` and `geometry_std` values before comparing it with the geometry prototypes.

A training set can contain a geometry feature with zero variance. In that case its standard deviation is `0`, and direct division by that value can produce `NaN` or infinite values during recognition. Those invalid values can contaminate the geometry score and, in turn, the combined recognition score.

## Expected behavior

A zero-variance geometry dimension should not contribute to the normalized geometry signal. The normalization should therefore:

1. Calculate `geometry - geometry_mean`.
2. Divide only where the corresponding standard deviation is greater than a small numerical tolerance.
3. Set zero-variance dimensions to `0` rather than allowing division by zero.
4. Continue using the embedding score normally when geometry normalization is unavailable or invalid.

## Regression test target

The recognition test suite should include a case where at least one `geometry_std` entry is zero and verify that recognition produces finite scores and does not emit `NaN` or infinity.

This document records the failure mode so the runtime fix and its regression test can be implemented together without changing the intended embedding/geometry weighting model.
