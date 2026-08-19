import cv2
import os
import time
import json
import numpy as np

# ============================================================
# ROBOT DOG - YuNet Face Capture
# Uses OpenCV YuNet ONNX for face detection.
# No MediaPipe is required.
# ============================================================

DATASET_PATH = "dataset"
PEOPLE_PATH = "people.json"
CAMERA_INDEX = 0

# Put the YuNet ONNX file in the same folder as this script.
MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# Detection settings
INPUT_SIZE = (320, 320)
CONF_THRESHOLD = 0.75
NMS_THRESHOLD = 0.30
TOP_K = 5000

# 120 useful images with guided pose diversity.
POSE_PLAN = [
    ("CENTER", 25, 0, 0),
    ("SLIGHT LEFT", 20, -15, 0),
    ("SLIGHT RIGHT", 20, 15, 0),
    ("LEFT", 20, -30, 0),
    ("RIGHT", 20, 30, 0),
    ("UP", 15, 0, -15),
]

TOTAL_TARGET = sum(p[1] for p in POSE_PLAN)

MIN_FACE_SIZE = 110
BLUR_THRESHOLD = 80.0
HOLD_TIME = 0.00
CAPTURE_DELAY = 0.45
DUPLICATE_THRESHOLD = 0.997
START_DELAY = 2.0

CAPTURE_BOX_SIZE = 300
CAPTURE_BOX_TOLERANCE = 0.20


def load_people():
    if os.path.exists(PEOPLE_PATH):
        try:
            with open(PEOPLE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_people(data):
    with open(PEOPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_name(name):
    return "".join(c for c in name if c.isalnum() or c in " _-").strip()


def instruction(pose):
    return {
        "CENTER": "LOOK STRAIGHT AT CAMERA",
        "SLIGHT LEFT": "TURN SLIGHTLY LEFT  <-",
        "SLIGHT RIGHT": "TURN SLIGHTLY RIGHT  ->",
        "LEFT": "TURN FACE LEFT  <<",
        "RIGHT": "TURN FACE RIGHT  >>",
        "UP": "LOOK UP  ^",
    }[pose]


def crop_face(frame, face):
    """
    YuNet face format:
    [x, y, w, h, right_eye_x, right_eye_y,
     left_eye_x, left_eye_y, nose_x, nose_y,
     right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y,
     confidence]
    """
    h, w = frame.shape[:2]

    x, y, fw, fh = face[:4].astype(int)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + fw)
    y2 = min(h, y + fh)

    if x2 <= x1 or y2 <= y1:
        return None, None

    # Add context around the detected face.
    px = int((x2 - x1) * 0.12)
    py = int((y2 - y1) * 0.12)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def quality_check(face):
    if face is None or face.size == 0:
        return False, "NO FACE"

    h, w = face.shape[:2]

    if min(h, w) < MIN_FACE_SIZE:
        return False, "MOVE CLOSER"

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    if sharpness < BLUR_THRESHOLD:
        return False, "TOO BLURRY"

    return True, "GOOD"


def signature(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32))
    small = cv2.equalizeHist(small).astype(np.float32).reshape(-1)
    norm = np.linalg.norm(small)
    return small / norm if norm > 1e-8 else small


def get_landmarks(face):
    """Return the five YuNet landmarks."""
    # YuNet indices:
    # right eye = 4:6
    # left eye = 6:8
    # nose      = 8:10
    # right mouth = 10:12
    # left mouth  = 12:14
    return {
        "right_eye": face[4:6].astype(float),
        "left_eye": face[6:8].astype(float),
        "nose": face[8:10].astype(float),
        "right_mouth": face[10:12].astype(float),
        "left_mouth": face[12:14].astype(float),
    }


def estimate_pose(face):
    """
    Lightweight head-pose estimate using YuNet's 5 landmarks.

    This is intentionally simpler than MediaPipe solvePnP.
    It is good enough for guided capture:
      yaw   > 0  -> face is toward the user's left side
      yaw   < 0  -> face is toward the user's right side
      pitch > 0  -> face is tilted upward
      pitch < 0  -> face is tilted downward

    Values are approximate degrees.
    """
    lm = get_landmarks(face)

    re = lm["right_eye"]
    le = lm["left_eye"]
    nose = lm["nose"]
    rm = lm["right_mouth"]
    lmouth = lm["left_mouth"]

    eye_mid = (re + le) / 2.0
    mouth_mid = (rm + lmouth) / 2.0

    eye_dist = np.linalg.norm(re - le)
    if eye_dist < 1e-6:
        return None

    # Horizontal nose displacement relative to the eye midpoint.
    yaw_ratio = (nose[0] - eye_mid[0]) / eye_dist

    # Normalize vertical structure using eye-to-mouth distance.
    vertical_dist = np.linalg.norm(mouth_mid - eye_mid)
    if vertical_dist < 1e-6:
        return None

    pitch_ratio = (nose[1] - eye_mid[1]) / vertical_dist

    # Approximate degrees. The scale is deliberately conservative
    # because this is guidance, not biometric pose measurement.
    yaw = yaw_ratio * 90.0

    # Centered faces are usually around ~0.5-0.6 in this ratio.
    pitch = (pitch_ratio - 0.52) * 100.0

    return float(yaw), float(pitch), 0.0


def draw_capture_area(frame, box=None, ready=False):
    h, w = frame.shape[:2]

    size = min(CAPTURE_BOX_SIZE, int(min(w, h) * 0.55))
    cx, cy = w // 2, h // 2

    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = cx + size // 2
    y2 = cy + size // 2

    color = (0, 255, 0) if ready else (0, 200, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        frame,
        "CAPTURE AREA",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), color, 1)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), color, 1)

    if box is not None:
        fx1, fy1, fx2, fy2 = box
        cv2.rectangle(
            frame,
            (fx1, fy1),
            (fx2, fy2),
            (0, 255, 0) if ready else (255, 200, 0),
            2,
        )

    return (x1, y1, x2, y2)


def face_in_capture_area(box, area):
    if box is None:
        return False

    fx1, fy1, fx2, fy2 = box
    ax1, ay1, ax2, ay2 = area

    aw = ax2 - ax1
    ah = ay2 - ay1

    tx = int(aw * CAPTURE_BOX_TOLERANCE)
    ty = int(ah * CAPTURE_BOX_TOLERANCE)

    rx1 = ax1 - tx
    ry1 = ay1 - ty
    rx2 = ax2 + tx
    ry2 = ay2 + ty

    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2

    return rx1 <= face_cx <= rx2 and ry1 <= face_cy <= ry2


def center_guidance(yaw, pitch):
    messages = []

    if yaw > 8:
        messages.append("MOVE FACE RIGHT -> CENTER")
    elif yaw < -8:
        messages.append("MOVE FACE LEFT <- CENTER")
    else:
        messages.append("YAW CENTERED")

    if pitch > 8:
        messages.append("MOVE FACE DOWN v CENTER")
    elif pitch < -8:
        messages.append("MOVE FACE UP ^ CENTER")

    return messages


# ============================================================
# Person setup
# ============================================================

person_name = input("Enter person name: ").strip()
if not person_name:
    raise SystemExit("Name cannot be empty.")

role = input(
    "Enter role/hierarchy (Student/Faculty/HOD/Principal/etc.): "
).strip() or "Unknown"

folder = safe_name(person_name)
if not folder:
    raise SystemExit("Invalid name.")

person_path = os.path.join(DATASET_PATH, folder)
os.makedirs(person_path, exist_ok=True)

people = load_people()
people[folder] = {"name": person_name, "role": role}
save_people(people)

# Fresh-person mode: clear old images for this person.
progress_file = os.path.join(person_path, "progress.json")

if os.path.exists(progress_file):
    try:
        os.remove(progress_file)
    except OSError:
        pass

for old_file in os.listdir(person_path):
    if old_file.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            os.remove(os.path.join(person_path, old_file))
        except OSError:
            pass

img_count = 0

# ============================================================
# Load YuNet
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"YuNet model not found:\n{MODEL_PATH}\n\n"
        "Put face_detection_yunet_2023mar.onnx in the same folder "
        "as this Python file."
    )

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    INPUT_SIZE,
    CONF_THRESHOLD,
    NMS_THRESHOLD,
    TOP_K,
)

# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise SystemExit("Could not open camera.")

stats = {
    "accepted": 0,
    "no_face": 0,
    "low_quality": 0,
    "too_small": 0,
    "wrong_pose": 0,
    "duplicate": 0,
}

print("\n" + "=" * 65)
print("       ROBOT DOG - YuNet FACE REGISTRATION")
print("=" * 65)
print(f"Person: {person_name} | Role: {role}")
print(f"Target: {TOTAL_TARGET} useful images")
print("Face detector: YuNet ONNX")
print("The program will guide you through different poses.")
print("Press Q to stop.")
print("=" * 65)

window_name = "Robot Dog - YuNet Face Capture"

try:
    import tkinter as tk

    screen = tk.Tk()
    screen.withdraw()
    SCREEN_W = screen.winfo_screenwidth()
    SCREEN_H = screen.winfo_screenheight()
    screen.destroy()
except Exception:
    SCREEN_W, SCREEN_H = 1280, 720

# Half-screen camera window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Use approximately half of the screen.
WINDOW_W = max(800, SCREEN_W // 2)
WINDOW_H = max(450, SCREEN_H // 2)

cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)
cv2.moveWindow(window_name, 20, 20)

session_start = time.time()
last_capture = 0.0
last_signature = None

# ============================================================
# Capture loop
# ============================================================

for pose_index, (pose_name, target, target_yaw, target_pitch) in enumerate(POSE_PLAN):
    step = pose_index + 1
    captured_for_pose = 0
    hold_start = None

    while captured_for_pose < target:
        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]

        # YuNet requires the current image size.
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)

        capture_area = draw_capture_area(frame, box=None, ready=False)

        status = "NO FACE"
        yaw = pitch = roll = 0.0
        box = None
        pose_ok = False
        in_area = False

        if faces is None or len(faces) == 0:
            stats["no_face"] += 1
            hold_start = None
        else:
            # Choose the largest detected face.
            faces = sorted(
                faces,
                key=lambda f: float(f[2] * f[3]),
                reverse=True,
            )

            face_detection = faces[0]

            face, box = crop_face(frame, face_detection)
            good, quality_status = quality_check(face)

            if not good:
                status = quality_status
                hold_start = None

                if quality_status == "MOVE CLOSER":
                    stats["too_small"] += 1
                else:
                    stats["low_quality"] += 1
            else:
                pose = estimate_pose(face_detection)

                if pose is None:
                    status = "POSE ERROR"
                    hold_start = None
                else:
                    yaw, pitch, roll = pose

                    # Wider tolerances make capture easier on a webcam.
                    yaw_tolerance = 18 if abs(target_yaw) < 20 else 16
                    pitch_tolerance = 18 if abs(target_pitch) < 10 else 16

                    pose_ok = (
                        abs(yaw - target_yaw) <= yaw_tolerance
                        and abs(pitch - target_pitch) <= pitch_tolerance
                    )

                    in_area = face_in_capture_area(
                        box,
                        capture_area,
                    )

                    if target_yaw == 0 and target_pitch == 0 and not pose_ok:
                        status = " | ".join(center_guidance(yaw, pitch))

                    elif pose_ok and not in_area:
                        status = "MOVE FACE INTO CENTER SQUARE"

                    elif pose_ok and in_area:
                        status = "POSE + AREA GOOD - HOLD"

                        if hold_start is None:
                            hold_start = time.time()

                        held = time.time() - hold_start
                        hold_left = max(0.0, HOLD_TIME - held)

                        cv2.putText(
                            frame,
                            f"HOLD... {hold_left:.1f}s",
                            (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 255, 0),
                            2,
                        )

                        if (
                            held >= HOLD_TIME
                            and time.time() - last_capture >= CAPTURE_DELAY
                        ):
                            sig = signature(face)

                            duplicate = (
                                last_signature is not None
                                and float(np.dot(last_signature, sig))
                                >= DUPLICATE_THRESHOLD
                            )

                            if duplicate:
                                stats["duplicate"] += 1
                                status = "TOO SIMILAR - MOVE A LITTLE"
                            else:
                                filename = f"{folder}_{img_count:03d}.jpg"
                                filepath = os.path.join(person_path, filename)

                                cv2.imwrite(
                                    filepath,
                                    face,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                                )

                                img_count += 1
                                captured_for_pose += 1
                                stats["accepted"] += 1
                                last_capture = time.time()
                                last_signature = sig
                                hold_start = None

                                print(
                                    f"[CAPTURED] {filename} | "
                                    f"{pose_name} "
                                    f"{captured_for_pose}/{target} | "
                                    f"Total {stats['accepted']}/{TOTAL_TARGET}"
                                )

                                status = "CAPTURED ✓"

                    else:
                        status = instruction(pose_name)
                        hold_start = None
                        stats["wrong_pose"] += 1

        ready_for_box = (
            box is not None
            and pose_ok
            and in_area
        )

        draw_capture_area(
            frame,
            box=box,
            ready=ready_for_box,
        )

        # ====================================================
        # Transparent text overlay - no black information panel
        # ====================================================
        def overlay_text(image, text, position, scale, color, thickness=1):
            cv2.putText(
                image,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                thickness + 3,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        overlay_text(
            frame,
            f"STEP {step}/{len(POSE_PLAN)}: {pose_name}",
            (10, 30),
            0.58,
            (0, 255, 255),
            2,
        )

        overlay_text(
            frame,
            instruction(pose_name),
            (10, 60),
            0.48,
            (0, 255, 0),
            2,
        )

        overlay_text(
            frame,
            f"STATUS: {status}",
            (10, 88),
            0.44,
            (255, 255, 0),
            2,
        )

        overlay_text(
            frame,
            f"Yaw:{yaw:+.1f} Pitch:{pitch:+.1f}  "
            f"Pose:{captured_for_pose}/{target}  "
            f"Total:{stats['accepted']}/{TOTAL_TARGET}",
            (10, 115),
            0.40,
            (255, 255, 255),
            1,
        )

        overlay_text(
            frame,
            "AUTO CAPTURE | Q = STOP",
            (w - 230, 30),
            0.40,
            (255, 255, 255),
            1,
        )

        # Startup countdown.
        if time.time() - session_start < START_DELAY:
            remaining = max(
                1,
                int(START_DELAY - (time.time() - session_start)) + 1,
            )

            cv2.putText(
                frame,
                f"GET READY: {remaining}",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Capture stopped.")

    print(
        f"[POSE COMPLETE] {pose_name}: "
        f"{captured_for_pose}/{target}"
    )

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 65)
print("REGISTRATION COMPLETE")
print("=" * 65)
print(f"Person: {person_name}")
print(f"Role: {role}")
print(f"Useful images: {stats['accepted']}")
print(f"No face: {stats['no_face']}")
print(f"Low quality: {stats['low_quality']}")
print(f"Too small: {stats['too_small']}")
print(f"Wrong pose: {stats['wrong_pose']}")
print(f"Near duplicates: {stats['duplicate']}")
print(f"Saved total in folder: {img_count}")
print(f"Dataset: {person_path}")
