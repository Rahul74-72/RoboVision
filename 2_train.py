import os
import cv2
import json
import pickle
import numpy as np
import mediapipe as mp
import torch
from facenet_pytorch import InceptionResnetV1

DATASET_PATH = "dataset"
ENCODINGS_PATH = "encodings.pickle"
PEOPLE_PATH = "people.json"

MIN_FACE_SIZE = 80
BLUR_THRESHOLD = 45.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FaceNet/InceptionResNetV1 pretrained on VGGFace2.
facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

LANDMARKS = {
    "left_eye_outer": 33, "left_eye_inner": 133,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "nose_top": 6, "nose_tip": 1,
    "nose_left": 98, "nose_right": 327,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top": 13, "mouth_bottom": 14,
    "left_cheek": 234, "right_cheek": 454,
    "left_jaw": 172, "right_jaw": 397,
    "chin": 152, "forehead": 10,
}


def l2(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def point(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def geometry_features(landmarks):
    p = {name: point(landmarks, idx) for name, idx in LANDMARKS.items()}
    face_width = dist(p["left_cheek"], p["right_cheek"])
    eye_distance = dist(p["left_eye_outer"], p["right_eye_outer"])
    face_height = dist(p["forehead"], p["chin"])

    if min(face_width, eye_distance, face_height) < 1e-6:
        return None

    features = [
        face_width / face_height,
        eye_distance / face_width,
        eye_distance / face_height,
        dist(p["left_eye_outer"], p["left_eye_inner"]) / face_width,
        dist(p["right_eye_inner"], p["right_eye_outer"]) / face_width,
        dist(p["left_eye_top"], p["left_eye_bottom"]) / face_width,
        dist(p["right_eye_top"], p["right_eye_bottom"]) / face_width,
        dist(p["nose_left"], p["nose_right"]) / face_width,
        dist(p["nose_top"], p["nose_tip"]) / face_height,
        dist(p["nose_tip"], p["mouth_top"]) / face_height,
        dist(p["mouth_left"], p["mouth_right"]) / face_width,
        dist(p["mouth_top"], p["mouth_bottom"]) / face_height,
        dist(p["left_jaw"], p["right_jaw"]) / face_width,
        dist(p["left_jaw"], p["chin"]) / face_height,
        dist(p["right_jaw"], p["chin"]) / face_height,
        dist(p["chin"], p["mouth_bottom"]) / face_height,
        dist(p["left_cheek"], p["nose_tip"]) / face_width,
        dist(p["right_cheek"], p["nose_tip"]) / face_width,
        dist(p["left_eye_inner"], p["mouth_left"]) / face_width,
        dist(p["right_eye_inner"], p["mouth_right"]) / face_width,
    ]
    return np.asarray(features, dtype=np.float32)


def align_face(image_rgb, landmarks):
    h, w = image_rgb.shape[:2]
    left, right = landmarks[33], landmarks[263]

    x1, y1 = left.x * w, left.y * h
    x2, y2 = right.x * w, right.y * h

    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_center_face(aligned):
    h, w = aligned.shape[:2]
    return aligned[
        int(h * 0.12):int(h * 0.88),
        int(w * 0.18):int(w * 0.82),
    ]


@torch.inference_mode()
def get_embedding(face_rgb):
    face_rgb = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)

    # FaceNet/InceptionResnetV1 expects 160x160 RGB tensors in [-1, 1].
    x = torch.from_numpy(
        face_rgb.astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    x = (x - 127.5) / 128.0
    embedding = facenet(x).cpu().numpy()[0]

    return l2(embedding)


def quality_ok(face):
    if face is None or face.size == 0:
        return False

    h, w = face.shape[:2]
    if min(h, w) < MIN_FACE_SIZE:
        return False

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_THRESHOLD


def load_people():
    try:
        with open(PEOPLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


people = load_people()

person_embeddings = {}
person_geometry = {}
roles = {}

all_embeddings = []
all_names = []

stats_all = {}
total_images = 0
total_trained = 0
total_skipped = 0

print("\n" + "=" * 72)
print("ROBOT DOG - FACENET TRAINING")
print("=" * 72)
print("Pretrained model : FaceNet / InceptionResNetV1")
print("Pretraining data : VGGFace2")
print(f"Device           : {DEVICE}")
print("Embedding size   : 512")
print("Geometry features: 20")
print("=" * 72)

if not os.path.isdir(DATASET_PATH):
    raise SystemExit("Dataset folder not found. Run 1_capture.py first.")

for folder in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(person_path):
        continue

    display_name = people.get(folder, {}).get("name", folder)
    role = people.get(folder, {}).get("role", "Unknown")

    files = [
        f for f in sorted(os.listdir(person_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    stats = {
        "total": len(files),
        "trained": 0,
        "skipped": 0,
        "no_face": 0,
        "low_quality": 0,
        "geometry_failed": 0,
        "embedding_error": 0,
    }

    total_images += len(files)
    embeddings = []
    geometries = []

    print(f"\n[PERSON] {display_name} | {role}")
    print(f"Images found: {len(files)}")

    for filename in files:
        path = os.path.join(person_path, filename)
        bgr = cv2.imread(path)

        if bgr is None:
            stats["skipped"] += 1
            stats["embedding_error"] += 1
            print(f"  [SKIP] unreadable: {filename}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            stats["skipped"] += 1
            stats["no_face"] += 1
            print(f"  [SKIP] no face: {filename}")
            continue

        landmarks = result.multi_face_landmarks[0].landmark
        geometry = geometry_features(landmarks)

        if geometry is None:
            stats["skipped"] += 1
            stats["geometry_failed"] += 1
            print(f"  [SKIP] geometry failed: {filename}")
            continue

        aligned = align_face(rgb, landmarks)
        face = crop_center_face(aligned)

        if not quality_ok(face):
            stats["skipped"] += 1
            stats["low_quality"] += 1
            print(f"  [SKIP] low quality: {filename}")
            continue

        try:
            emb = get_embedding(face)
            embeddings.append(emb)
            geometries.append(geometry)
            all_embeddings.append(emb)
            all_names.append(display_name)
            stats["trained"] += 1
            print(f"  [TRAINED] {filename}")
        except Exception as e:
            stats["skipped"] += 1
            stats["embedding_error"] += 1
            print(f"  [SKIP] FaceNet error: {filename} -> {e}")

    total_trained += stats["trained"]
    total_skipped += stats["skipped"]
    stats_all[display_name] = stats

    print(
        f"  SUMMARY: Total={stats['total']} | "
        f"Trained={stats['trained']} | "
        f"Skipped={stats['skipped']}"
    )

    if embeddings:
        person_embeddings[display_name] = l2(np.mean(embeddings, axis=0))
        person_geometry[display_name] = np.mean(
            np.stack(geometries), axis=0
        ).astype(np.float32)
        roles[display_name] = role

if not person_embeddings:
    raise SystemExit("No usable faces found.")

geometry_matrix = np.stack(
    [person_geometry[name] for name in person_embeddings]
)

geometry_mean = np.mean(geometry_matrix, axis=0)
geometry_std = np.std(geometry_matrix, axis=0)
geometry_std[geometry_std < 1e-5] = 1.0

geometry_prototypes = {
    name: l2(
        (person_geometry[name] - geometry_mean) /
        geometry_std
    )
    for name in person_embeddings
}

data = {
    "version": 5,
    "recognition_type": "facenet_vggface2_plus_geometry",
    "model": "FaceNet_InceptionResNetV1_VGGFace2",
    "embedding_dimension": 512,
    "embeddings": np.stack(all_embeddings).astype(np.float32),
    "names": all_names,
    "prototypes": {
        name: vec.astype(np.float32)
        for name, vec in person_embeddings.items()
    },
    "geometry_prototypes": geometry_prototypes,
    "geometry_mean": geometry_mean.astype(np.float32),
    "geometry_std": geometry_std.astype(np.float32),
    "roles": roles,
    "embedding_weight": 0.80,
    "geometry_weight": 0.20,
    "threshold_hint": 0.55,
    "margin_hint": 0.05,
    "training_stats": stats_all,
}

with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

face_mesh.close()

print("\n" + "=" * 72)
print("TRAINING SUMMARY")
print("=" * 72)

for name, s in stats_all.items():
    print(
        f"{name}: Total={s['total']} | "
        f"Trained={s['trained']} | "
        f"Skipped={s['skipped']}"
    )

print("-" * 72)
print(f"TOTAL IMAGES FOUND   : {total_images}")
print(f"TOTAL IMAGES TRAINED : {total_trained}")
print(f"TOTAL IMAGES SKIPPED : {total_skipped}")
print(f"TOTAL EMBEDDINGS     : {len(all_embeddings)}")
print(f"PEOPLE               : {len(person_embeddings)}")
print("MODEL                : FaceNet InceptionResNetV1")
print("EMBEDDING DIMENSION  : 512")
print("=" * 72)
print(f"Database saved: {ENCODINGS_PATH}")