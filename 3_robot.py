import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
import queue
import uuid
import datetime
import os
import torch
from collections import deque, Counter
from facenet_pytorch import InceptionResnetV1

from gtts import gTTS
import pygame

ENCODINGS_PATH = "encodings.pickle"
CAMERA_INDEX = 0

EMBEDDING_WEIGHT = 0.80
GEOMETRY_WEIGHT = 0.20
CONFIDENCE_THRESHOLD = 0.55
MATCH_MARGIN = 0.05

VOTE_WINDOW = 7
MIN_VOTES = 5
GREETING_COOLDOWN = 30
MIN_FACE_SIZE = 70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

if data.get("model") != "FaceNet_InceptionResNetV1_VGGFace2":
    print("[WARNING] Database was not created with the new FaceNet trainer.")
    print("[WARNING] Delete encodings.pickle and run 2_train.py again.")

prototype_names = list(data.get("prototypes", {}).keys())
prototype_vectors = np.asarray(
    [data["prototypes"][n] for n in prototype_names],
    dtype=np.float32,
)

geometry_vectors = np.asarray(
    [data["geometry_prototypes"][n] for n in prototype_names],
    dtype=np.float32,
)

geometry_mean = np.asarray(
    data.get("geometry_mean", []),
    dtype=np.float32,
)

geometry_std = np.asarray(
    data.get("geometry_std", []),
    dtype=np.float32,
)

roles = data.get("roles", {})
EMBEDDING_WEIGHT = float(data.get("embedding_weight", EMBEDDING_WEIGHT))
GEOMETRY_WEIGHT = float(data.get("geometry_weight", GEOMETRY_WEIGHT))

# FaceNet pretrained model.
facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(DEVICE)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
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
    fw = dist(p["left_cheek"], p["right_cheek"])
    ed = dist(p["left_eye_outer"], p["right_eye_outer"])
    fh = dist(p["forehead"], p["chin"])

    if min(fw, ed, fh) < 1e-6:
        return None

    return np.asarray([
        fw / fh, ed / fw, ed / fh,
        dist(p["left_eye_outer"], p["left_eye_inner"]) / fw,
        dist(p["right_eye_inner"], p["right_eye_outer"]) / fw,
        dist(p["left_eye_top"], p["left_eye_bottom"]) / fw,
        dist(p["right_eye_top"], p["right_eye_bottom"]) / fw,
        dist(p["nose_left"], p["nose_right"]) / fw,
        dist(p["nose_top"], p["nose_tip"]) / fh,
        dist(p["nose_tip"], p["mouth_top"]) / fh,
        dist(p["mouth_left"], p["mouth_right"]) / fw,
        dist(p["mouth_top"], p["mouth_bottom"]) / fh,
        dist(p["left_jaw"], p["right_jaw"]) / fw,
        dist(p["left_jaw"], p["chin"]) / fh,
        dist(p["right_jaw"], p["chin"]) / fh,
        dist(p["chin"], p["mouth_bottom"]) / fh,
        dist(p["left_cheek"], p["nose_tip"]) / fw,
        dist(p["right_cheek"], p["nose_tip"]) / fw,
        dist(p["left_eye_inner"], p["mouth_left"]) / fw,
        dist(p["right_eye_inner"], p["mouth_right"]) / fw,
    ], dtype=np.float32)


def align_face(image_rgb, landmarks):
    h, w = image_rgb.shape[:2]
    left, right = landmarks[33], landmarks[263]

    angle = np.degrees(
        np.arctan2(
            right.y * h - left.y * h,
            right.x * w - left.x * w,
        )
    )

    matrix = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_face(aligned):
    h, w = aligned.shape[:2]
    return aligned[
        int(h * 0.12):int(h * 0.88),
        int(w * 0.18):int(w * 0.82),
    ]


@torch.inference_mode()
def get_embedding(face_rgb):
    face_rgb = cv2.resize(
        face_rgb,
        (160, 160),
        interpolation=cv2.INTER_AREA,
    )

    x = torch.from_numpy(
        face_rgb.astype(np.float32)
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    x = (x - 127.5) / 128.0
    emb = facenet(x).cpu().numpy()[0]

    return l2(emb)


def recognize(embedding, geometry):
    if not prototype_names:
        return "Unknown", 0.0, 0.0, 0.0

    embedding_scores = prototype_vectors @ embedding

    if geometry is not None and len(geometry_mean):
        z = l2((geometry - geometry_mean) / geometry_std)
        geometry_scores = geometry_vectors @ z
    else:
        geometry_scores = np.zeros(
            len(prototype_names),
            dtype=np.float32,
        )

    combined = (
        EMBEDDING_WEIGHT * embedding_scores +
        GEOMETRY_WEIGHT * geometry_scores
    )

    order = np.argsort(combined)[::-1]
    best = int(order[0])

    best_combined = float(combined[best])
    second = float(combined[order[1]]) if len(order) > 1 else -1.0
    margin = best_combined - second

    best_embedding = float(embedding_scores[best])
    best_geometry = float(geometry_scores[best])

    if (
        best_combined < CONFIDENCE_THRESHOLD or
        margin < MATCH_MARGIN
    ):
        return "Unknown", best_combined, best_embedding, best_geometry

    return (
        prototype_names[best],
        best_combined,
        best_embedding,
        best_geometry,
    )


def greeting_for(name, role):
    hour = datetime.datetime.now().hour

    if 5 <= hour < 12:
        time_word = "Good morning"
    elif 12 <= hour < 17:
        time_word = "Good afternoon"
    elif 17 <= hour < 21:
        time_word = "Good evening"
    else:
        time_word = "Hello"

    return (
        f"{time_word}, {role} {name}"
        if role and role != "Unknown"
        else f"{time_word}, {name}"
    )


def speak(text):
    # Unique file per greeting prevents Windows file-lock conflicts.
    filename = os.path.abspath(
        f"robot_greeting_{uuid.uuid4().hex}.mp3"
    )

    try:
        gTTS(
            text=text,
            lang="en"
        ).save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    except Exception as e:
        print("[TTS ERROR]", e)

    finally:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        # Windows may keep the MP3 locked for a short time.
        for _ in range(15):
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                break
            except PermissionError:
                time.sleep(0.1)
            except OSError:
                break


tts_queue = queue.Queue(maxsize=5)

def tts_worker():
    while True:
        text = tts_queue.get()
        try:
            if text is None: return
            speak(text)
        finally:
            tts_queue.task_done()

pygame.mixer.init()
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


# ============================================================
# PER-FACE TRACKING + GREETING
# ============================================================

face_tracks = {}
next_track_id = 0
TRACK_MAX_DISTANCE = 140
TRACK_MAX_MISSED = 8


def face_center(box):
    x1,y1,x2,y2=box
    return ((x1+x2)/2.0,(y1+y2)/2.0)


def assign_track(box):
    global next_track_id

    cx,cy=face_center(box)
    best=None
    best_dist=float("inf")

    for tid,t in face_tracks.items():
        if t["missed"] > TRACK_MAX_MISSED:
            continue

        tx,ty=t["center"]
        d=((cx-tx)**2+(cy-ty)**2)**0.5
        x1,y1,x2,y2=box
        size=max(x2-x1,y2-y1)
        allowed=max(TRACK_MAX_DISTANCE,size*1.25)

        if d<allowed and d<best_dist:
            best=tid
            best_dist=d

    if best is None:
        best=next_track_id
        next_track_id+=1
        face_tracks[best]={
            "center":(cx,cy),
            "votes":deque(maxlen=VOTE_WINDOW),
            "name":"Unknown",
            "missed":0,
            "greeted":False
        }

    t=face_tracks[best]
    t["center"]=(cx,cy)
    t["missed"]=0
    return best,t


def stable_track_name(track):
    votes=[v for v in track["votes"] if v!="Unknown"]

    if len(votes)<MIN_VOTES:
        return None

    name,count=Counter(votes).most_common(1)[0]

    return name if count>=MIN_VOTES else None


def cleanup_tracks(active):
    for tid in list(face_tracks):
        if tid not in active:
            face_tracks[tid]["missed"]+=1

            if face_tracks[tid]["missed"]>TRACK_MAX_MISSED:
                del face_tracks[tid]


def queue_track_greeting(tid,name,combined,embedding,geometry):
    track=face_tracks.get(tid)

    if track is None or track["greeted"]:
        return

    role=roles.get(name,"Unknown")
    greeting=greeting_for(name,role)

    try:
        tts_queue.put_nowait(greeting)
    except queue.Full:
        print(f"[TTS] Queue full - skipped Face {tid}: {name}")
        return

    print(
        f"[GREETING] Face {tid} | {greeting} | "
        f"combined={combined:.3f} "
        f"embedding={embedding:.3f} "
        f"geometry={geometry:.3f}"
    )

    track["greeted"]=True


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise SystemExit("Could not open camera.")

print("\n" + "=" * 60)
print("ROBOT DOG - FACENET FACE AI")
print("=" * 60)
print("Model: FaceNet / InceptionResNetV1")
print("Pretrained on: VGGFace2")
print("Embedding: 512 dimensions")
print("Device:", DEVICE)
print("Embedding weight:", EMBEDDING_WEIGHT)
print("Geometry weight:", GEOMETRY_WEIGHT)
print("Press Q to quit.")
print("=" * 60)


while True:
    ret,frame=cap.read()
    if not ret:
        break

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    detections=[]

    # Detect every face.
    if results.multi_face_landmarks:
        for fl in results.multi_face_landmarks:
            lm=fl.landmark

            xs=[p.x*frame.shape[1] for p in lm]
            ys=[p.y*frame.shape[0] for p in lm]

            x1=max(0,int(min(xs)))
            y1=max(0,int(min(ys)))
            x2=min(frame.shape[1],int(max(xs)))
            y2=min(frame.shape[0],int(max(ys)))

            if x2>x1 and y2>y1:
                detections.append({
                    "box":(x1,y1,x2,y2),
                    "landmarks":lm
                })

    active=set()
    current=[]

    # Each face gets its OWN track and OWN FaceNet embedding.
    for det in detections:
        box=det["box"]
        landmarks=det["landmarks"]

        tid,track=assign_track(box)
        active.add(tid)

        geometry=geometry_features(landmarks)
        aligned=align_face(rgb,landmarks)
        face=crop_face(aligned)

        if face is None or face.size==0 or min(face.shape[:2])<MIN_FACE_SIZE:
            name="Unknown"
            combined=emb_score=geo_score=0.0
        else:
            try:
                embedding=get_embedding(face)
                name,combined,emb_score,geo_score=recognize(
                    embedding,geometry
                )
            except Exception as e:
                print(f"[RECOGNITION ERROR] Face {tid}: {e}")
                name="Unknown"
                combined=emb_score=geo_score=0.0

        # Vote ONLY inside this face's track.
        track["votes"].append(name)

        stable=stable_track_name(track)

        if stable is not None:
            track["name"]=stable

        display_name=track["name"]

        current.append({
            "track":tid,
            "name":display_name,
            "combined":combined,
            "embedding":emb_score,
            "geometry":geo_score
        })

        if stable is not None:
            queue_track_greeting(
                tid,
                stable,
                combined,
                emb_score,
                geo_score
            )

        x1,y1,x2,y2=box
        color=(0,255,0) if display_name!="Unknown" else (0,0,255)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            frame,
            f"Face {tid}",
            (x1,max(48,y1-8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .45,
            color,
            1
        )

        cv2.putText(
            frame,
            f"C:{combined:.2f} E:{emb_score:.2f} G:{geo_score:.2f}",
            (x1,min(frame.shape[0]-8,y2+18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .42,
            color,
            1
        )

    cleanup_tracks(active)

    names=list(dict.fromkeys(
        x["name"] for x in current
        if x["name"]!="Unknown"
    ))

    if names:
        summary="People: "+", ".join(names)
    elif current:
        summary="People: Unknown"
    else:
        summary="People: None"

    cv2.putText(
        frame,summary,(10,30),
        cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,0),2
    )

    cv2.putText(
        frame,f"Faces detected: {len(current)}",
        (10,58),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,0),2
    )

    cv2.putText(
        frame,f"TTS queue: {tts_queue.qsize()}",
        (10,84),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,0),2
    )

    cv2.imshow(
        "Robot Dog - FaceNet Multi-Face AI",
        frame
    )

    if cv2.waitKey(1)&0xFF==ord("q"):
        break


cap.release()
face_mesh.close()
try:
    tts_queue.put_nowait(None)
except queue.Full:
    pass
tts_thread.join(timeout=1.0)
pygame.mixer.quit()
cv2.destroyAllWindows()