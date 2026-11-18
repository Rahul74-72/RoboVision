# Lightweight geometry regression tests

Run from the repository root in a separate virtual environment:

```sh
python -m pip install numpy==1.26.4 pytest
python -m pytest tests/test_geometry_utils.py tests/test_geometry_extreme_values.py tests/test_geometry_input_contract.py tests/test_geometry_nonfinite_masking.py
```

These tests exercise `geometry_utils.normalize_geometry` without loading camera hardware, TensorFlow, MediaPipe, speech, or the interactive robot program. NumPy matches the version in the runtime requirements. Keep this environment separate from an existing application installation.

Coverage includes float32 output, extreme finite values, tiny vectors, the standard-deviation threshold, read-only strided inputs, and rejection of nonfinite values even in masked dimensions.

Passing this focused suite verifies the shared helper only. It does not establish that camera capture, model inference, tracking, or speech work. Run the production integration tests and hardware checks separately when changing those paths.
