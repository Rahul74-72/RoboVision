# RoboVision Recognition Pipeline

RoboVision currently combines FaceNet embeddings with facial geometry to recognize known people.

## Training flow

1. `1_capture.py` collects face images into the `dataset/` directory.
2. `2_train.py` loads each image and uses MediaPipe Face Mesh to locate one face.
3. Images with unusable geometry or insufficient visual quality are skipped.
4. The face is aligned and cropped before being resized to `160x160` RGB pixels.
5. `InceptionResnetV1` pretrained on VGGFace2 produces a 512-dimensional FaceNet embedding.
6. A 20-feature geometric representation is calculated from facial landmarks.
7. Per-person embedding and geometry prototypes are stored in `encodings.pickle`, together with the normalization statistics and training metadata.

## Runtime flow

`3_robot.py` loads the saved prototypes and processes faces from the camera:

1. MediaPipe detects multiple faces in each frame.
2. Each detected face is assigned its own track so recognition votes do not mix between people.
3. A 512-dimensional FaceNet embedding and 20-feature geometry vector are calculated for each face.
4. Embedding and geometry similarity scores are combined using the stored weights (currently `0.80` and `0.20`).
5. A match must pass both the confidence threshold and the margin over the second-best candidate.
6. Recognition results are accumulated in a per-track vote window before a greeting is triggered.

## Important artifacts

- `dataset/` — captured training images
- `people.json` — display names and roles associated with captured people
- `encodings.pickle` — generated recognition database
- `1_capture.py` — data capture
- `2_train.py` — FaceNet and geometry training
- `3_robot.py` — live recognition, tracking, and greeting

## Re-training after capture changes

When people or training images change, run `2_train.py` again so that `encodings.pickle` reflects the current dataset. The runtime script expects the FaceNet/VGGFace2 database format produced by the trainer.
