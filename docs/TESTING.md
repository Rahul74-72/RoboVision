# RoboVision Testing

RoboVision combines camera input, MediaPipe landmarks, FaceNet inference, geometry scoring, and text-to-speech. The core runtime therefore depends on hardware and heavyweight ML packages that are not suitable for every automated test.

## Current lightweight tests

`tests/test_robot_tracking_source.py` uses Python's `ast` module to inspect `3_robot.py` without importing the camera, MediaPipe, FaceNet, or TTS dependencies.

The tests currently protect:

- one authoritative definition of the face-tracking helpers;
- vote-based stable recognition;
- greeting generation for time-of-day and role/name handling.

## Running the tests

From the RoboVision repository root:

```bash
python -m pytest tests/test_robot_tracking_source.py
```

These tests are intentionally lightweight. Full runtime verification should be performed separately on the target machine with the camera and required ML/TTS dependencies installed.
