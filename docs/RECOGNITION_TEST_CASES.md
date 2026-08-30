# Recognition Regression Cases

This document records the small, repeatable cases to verify when changing the RoboVision recognition pipeline.

## Geometry guard

`geometry_features()` should return `None` when any of the three base face measurements (`fw`, `ed`, or `fh`) is effectively zero. This prevents invalid ratios from entering recognition.

## Missing geometry

`recognize()` should still return a valid result when geometry is unavailable. In that case the geometry score is zero and the embedding score remains usable.

## Single prototype

With one enrolled prototype, the margin comparison has no second candidate. The implementation should use the existing fallback margin behavior rather than indexing a missing second score.

## Tracking stability

A track should not be assigned a stable identity until it has at least `MIN_VOTES` non-`Unknown` votes. `Unknown` observations must not count toward the vote threshold.

## TTS queue pressure

When the TTS queue is full, the recognition loop should skip the greeting rather than blocking face processing. The track should only be marked as greeted after the greeting is successfully queued.

## Next runtime regression

The next numerical fix should cover zero or near-zero entries in `geometry_std` before z-score normalization. The expected result is a finite geometry score rather than `NaN` or infinity.
