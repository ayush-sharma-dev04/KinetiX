import time
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================================================================
# 1. Landmarker Setup
# =============================================================================

def create_landmarker(model_path="pose_landmarker_lite.task"):
    """Initialize MediaPipe Pose Landmarker for video mode."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)


# =============================================================================
# 2. Landmark Indices, Names & Connections
# =============================================================================

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Landmarks whose combined visibility gates whether computed features for
# a frame are trusted. Broader than a single exercise's needs since this
# is the shared feature layer other scripts (squat.py, pushup.py, ...) build on.
REQUIRED_LANDMARKS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
]

# Readable name for every one of the 33 pose landmarks (used to label raw
# x/y/z/visibility columns in the CSV).
LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# Standard 33-point pose skeleton connections (index pairs).
POSE_CONNECTIONS = [
    (conn.start, conn.end) for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS
]


# =============================================================================
# 3. Video Source & Configuration
# =============================================================================

# Live webcam (0). To use a recorded video file, set e.g. VIDEO_SOURCE = "video.mp4"
VIDEO_SOURCE = 0

# Visibility thresholds
VISIBILITY_THRESHOLD = 0.5       # Smoothed visibility required to trust computed features
DRAW_VISIBILITY_THRESHOLD = 0.2  # Landmarks/connections below this are not drawn (noise filter)

# Temporal smoothing
SMOOTHING_WINDOW = 5             # Frames for sliding-window moving-average filter (visibility gate)

# CSV logging configuration
# -----------------------------------------------------------------------------
# GENERIC FEATURE LAYER NOTE:
# This file is the common extraction layer: MediaPipe -> 33 landmarks ->
# angles/distances/ratios/symmetry/alignment -> CSV. It intentionally has no
# thresholds, states, or rep-counting logic. Per-exercise scripts (squat.py,
# pushup.py, lunge.py, ...) consume landmarks.csv and add their own logic.
# -----------------------------------------------------------------------------
CSV_PATH = "landmarks.csv"
FLUSH_EVERY = 30

RAW_LANDMARK_COLUMNS = [
    f"{name}_{suffix}"
    for name in LANDMARK_NAMES.values()
    for suffix in ("x", "y", "z", "visibility")
]

FEATURE_COLUMNS = [
    "left_knee_angle", "right_knee_angle", "avg_knee_angle",
    "left_hip_angle", "right_hip_angle", "avg_hip_angle",
    "left_elbow_angle", "right_elbow_angle", "avg_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle", "avg_shoulder_angle",
    "torso_angle",
    "shoulder_symmetry", "hip_symmetry", "shoulder_tilt_angle", "hip_tilt_angle",
    "shoulder_width", "hip_width", "stance_width", "foot_width",
    "left_upper_arm_length", "right_upper_arm_length",
    "left_forearm_length", "right_forearm_length",
    "left_thigh_length", "right_thigh_length",
    "left_shank_length", "right_shank_length",
    "torso_length",
    "shoulder_hip_ratio", "stance_shoulder_ratio",
    "left_thigh_shank_ratio", "right_thigh_shank_ratio",
    "left_arm_torso_ratio", "right_arm_torso_ratio",
    "left_leg_torso_ratio", "right_leg_torso_ratio",
    "neck_angle", "head_offset_x", "head_offset_ratio", "nose_to_mid_shoulder_dist",
]

CSV_COLUMNS = ["timestamp"] + RAW_LANDMARK_COLUMNS + ["pose_visibility"] + FEATURE_COLUMNS


# =============================================================================
# 4. Geometry Helpers
# =============================================================================

def _xy(lm):
    """Convert landmark to (x, y) numpy array."""
    return np.array([lm.x, lm.y])


def _midpoint(a, b):
    """Midpoint (x, y) of two landmarks."""
    return (_xy(a) + _xy(b)) / 2


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees [0, 180]."""
    a, b, c = _xy(a), _xy(b), _xy(c)

    ba = a - b
    bc = c - b

    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def calculate_distance(a, b):
    """Euclidean distance between two landmarks (x, y only)."""
    return float(np.linalg.norm(_xy(a) - _xy(b)))


def vertical_angle_of_vector(vec):
    """Angle (degrees) of a 2D vector relative to vertical (0 = upright)."""
    vertical_vec = np.array([0.0, -1.0])  # Upward in image coordinate space

    denom = np.linalg.norm(vec) * np.linalg.norm(vertical_vec)
    if denom < 1e-8:
        return 0.0

    cos_angle = np.dot(vec, vertical_vec) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def horizontal_angle_of_vector(vec):
    """Angle (degrees) of a 2D vector relative to horizontal (0 = level)."""
    horizontal_vec = np.array([1.0, 0.0])

    denom = np.linalg.norm(vec) * np.linalg.norm(horizontal_vec)
    if denom < 1e-8:
        return 0.0

    cos_angle = np.dot(vec, horizontal_vec) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def calculate_torso_angle(shoulder_l, shoulder_r, hip_l, hip_r):
    """
    Angle of the torso relative to vertical (0 = perfectly upright).
    Uses shoulder/hip midpoints to remain robust against torso rotation.
    """
    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    return vertical_angle_of_vector(shoulder_mid - hip_mid)


def safe_div(numerator, denominator):
    if denominator is None or np.isnan(denominator) or abs(denominator) < 1e-8:
        return np.nan
    return numerator / denominator


# =============================================================================
# 5. Feature Extraction (angles, distances, ratios, symmetry, alignment)
# =============================================================================

def extract_features(pose):
    """
    Given the 33 MediaPipe pose landmarks for one frame, compute a flat dict
    of reusable numerical features. Purely geometric — no exercise-specific
    thresholds, states, or logic live here; that belongs in per-exercise
    scripts that consume this CSV.
    """
    features = {}

    # --- Knee angles (hip - knee - ankle) ---
    left_knee = calculate_angle(pose[LEFT_HIP], pose[LEFT_KNEE], pose[LEFT_ANKLE])
    right_knee = calculate_angle(pose[RIGHT_HIP], pose[RIGHT_KNEE], pose[RIGHT_ANKLE])
    features["left_knee_angle"] = left_knee
    features["right_knee_angle"] = right_knee
    features["avg_knee_angle"] = (left_knee + right_knee) / 2

    # --- Hip angles (shoulder - hip - knee) ---
    left_hip = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_KNEE])
    right_hip = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_KNEE])
    features["left_hip_angle"] = left_hip
    features["right_hip_angle"] = right_hip
    features["avg_hip_angle"] = (left_hip + right_hip) / 2

    # --- Elbow angles (shoulder - elbow - wrist) ---
    left_elbow = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST])
    right_elbow = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST])
    features["left_elbow_angle"] = left_elbow
    features["right_elbow_angle"] = right_elbow
    features["avg_elbow_angle"] = (left_elbow + right_elbow) / 2

    # --- Shoulder angles (elbow - shoulder - hip) ---
    left_shoulder_ang = calculate_angle(pose[LEFT_ELBOW], pose[LEFT_SHOULDER], pose[LEFT_HIP])
    right_shoulder_ang = calculate_angle(pose[RIGHT_ELBOW], pose[RIGHT_SHOULDER], pose[RIGHT_HIP])
    features["left_shoulder_angle"] = left_shoulder_ang
    features["right_shoulder_angle"] = right_shoulder_ang
    features["avg_shoulder_angle"] = (left_shoulder_ang + right_shoulder_ang) / 2

    # --- Torso angle vs vertical ---
    features["torso_angle"] = calculate_torso_angle(
        pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER], pose[LEFT_HIP], pose[RIGHT_HIP]
    )

    mid_shoulder = _midpoint(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
    mid_hip = _midpoint(pose[LEFT_HIP], pose[RIGHT_HIP])

    # --- Symmetry: raw height diff + shoulder/hip line tilt vs horizontal ---
    features["shoulder_symmetry"] = float(abs(pose[LEFT_SHOULDER].y - pose[RIGHT_SHOULDER].y))
    features["hip_symmetry"] = float(abs(pose[LEFT_HIP].y - pose[RIGHT_HIP].y))
    features["shoulder_tilt_angle"] = horizontal_angle_of_vector(
        _xy(pose[RIGHT_SHOULDER]) - _xy(pose[LEFT_SHOULDER])
    )
    features["hip_tilt_angle"] = horizontal_angle_of_vector(
        _xy(pose[RIGHT_HIP]) - _xy(pose[LEFT_HIP])
    )

    # --- Widths ---
    shoulder_width = calculate_distance(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
    hip_width = calculate_distance(pose[LEFT_HIP], pose[RIGHT_HIP])
    stance_width = calculate_distance(pose[LEFT_ANKLE], pose[RIGHT_ANKLE])
    foot_width = calculate_distance(pose[LEFT_FOOT_INDEX], pose[RIGHT_FOOT_INDEX])
    features["shoulder_width"] = shoulder_width
    features["hip_width"] = hip_width
    features["stance_width"] = stance_width
    features["foot_width"] = foot_width

    # --- Limb / body segment lengths ---
    left_upper_arm = calculate_distance(pose[LEFT_SHOULDER], pose[LEFT_ELBOW])
    right_upper_arm = calculate_distance(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW])
    left_forearm = calculate_distance(pose[LEFT_ELBOW], pose[LEFT_WRIST])
    right_forearm = calculate_distance(pose[RIGHT_ELBOW], pose[RIGHT_WRIST])
    left_thigh = calculate_distance(pose[LEFT_HIP], pose[LEFT_KNEE])
    right_thigh = calculate_distance(pose[RIGHT_HIP], pose[RIGHT_KNEE])
    left_shank = calculate_distance(pose[LEFT_KNEE], pose[LEFT_ANKLE])
    right_shank = calculate_distance(pose[RIGHT_KNEE], pose[RIGHT_ANKLE])
    torso_length = float(np.linalg.norm(mid_shoulder - mid_hip))

    features["left_upper_arm_length"] = left_upper_arm
    features["right_upper_arm_length"] = right_upper_arm
    features["left_forearm_length"] = left_forearm
    features["right_forearm_length"] = right_forearm
    features["left_thigh_length"] = left_thigh
    features["right_thigh_length"] = right_thigh
    features["left_shank_length"] = left_shank
    features["right_shank_length"] = right_shank
    features["torso_length"] = torso_length

    # --- Normalized ratios (scale-invariant across body sizes) ---
    features["shoulder_hip_ratio"] = safe_div(shoulder_width, hip_width)
    features["stance_shoulder_ratio"] = safe_div(stance_width, shoulder_width)
    features["left_thigh_shank_ratio"] = safe_div(left_thigh, left_shank)
    features["right_thigh_shank_ratio"] = safe_div(right_thigh, right_shank)
    features["left_arm_torso_ratio"] = safe_div(left_upper_arm + left_forearm, torso_length)
    features["right_arm_torso_ratio"] = safe_div(right_upper_arm + right_forearm, torso_length)
    features["left_leg_torso_ratio"] = safe_div(left_thigh + left_shank, torso_length)
    features["right_leg_torso_ratio"] = safe_div(right_thigh + right_shank, torso_length)

    # --- Head / neck alignment (nose relative to mid-shoulder) ---
    nose_xy = _xy(pose[NOSE])
    features["neck_angle"] = vertical_angle_of_vector(nose_xy - mid_shoulder)
    features["head_offset_x"] = float(nose_xy[0] - mid_shoulder[0])
    features["head_offset_ratio"] = safe_div(nose_xy[0] - mid_shoulder[0], shoulder_width)
    features["nose_to_mid_shoulder_dist"] = float(np.linalg.norm(nose_xy - mid_shoulder))

    return features


# =============================================================================
# 6. Drawing Helpers (Pose Skeleton)
# =============================================================================

def draw_pose_skeleton(frame, pose):
    """
    Draws 33 pose landmarks and skeleton connections on the frame,
    color-coded by landmark visibility confidence.

    Green  = Confident (visibility >= VISIBILITY_THRESHOLD)
    Orange = Borderline (DRAW_VISIBILITY_THRESHOLD <= visibility < VISIBILITY_THRESHOLD)
    """
    h, w = frame.shape[:2]

    def point_px(lm):
        return int(lm.x * w), int(lm.y * h)

    def color_for(visibility):
        if visibility >= VISIBILITY_THRESHOLD:
            return (0, 255, 0)      # Green (BGR)
        return (0, 165, 255)        # Orange (BGR)

    # Draw connection lines first so landmark dots appear on top
    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = pose[start_idx]
        end_lm = pose[end_idx]

        if start_lm.visibility < DRAW_VISIBILITY_THRESHOLD or \
                end_lm.visibility < DRAW_VISIBILITY_THRESHOLD:
            continue

        edge_color = color_for(min(start_lm.visibility, end_lm.visibility))
        cv2.line(frame, point_px(start_lm), point_px(end_lm), edge_color, 2)

    # Draw landmark dots
    for lm in pose:
        if lm.visibility < DRAW_VISIBILITY_THRESHOLD:
            continue
        cv2.circle(frame, point_px(lm), 4, color_for(lm.visibility), -1)


# =============================================================================
# 7. Temporal Smoothing Filter
# =============================================================================

class MovingAverage:
    """Sliding-window moving average filter to denoise per-frame measurements."""

    def __init__(self, window_size):
        self.values = []
        self.window_size = window_size

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

    @property
    def ready(self):
        return len(self.values) > 0


# =============================================================================
# 8. CSV Logging Helper
# =============================================================================

buffer = []


def flush_buffer(rows, path):
    """Flush buffered frame rows to CSV dataset."""
    if not rows:
        return

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False
    )

    rows.clear()


def empty_row(timestamp_ms):
    """A row of all-None feature/landmark values, used when tracking is lost."""
    row = {col: None for col in CSV_COLUMNS}
    row["timestamp"] = timestamp_ms
    return row


# =============================================================================
# 9. HUD Rendering (Translucent Card Overlay)
# =============================================================================

def draw_hud(frame, features, pose_visibility, pose_detected):
    """
    Renders a translucent dark card HUD in the top-left corner showing the
    key computed features, for visual testing of this extraction layer.
    """
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (15, 15),
        (370, 345),
        (20, 20, 20),
        -1
    )
    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    cv2.putText(
        frame, "POSE FEATURES", (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )

    vis_color = (0, 255, 0) if (pose_visibility or 0) >= VISIBILITY_THRESHOLD else (0, 165, 255)
    vis_str = f"VISIBILITY: {pose_visibility:.2f}" if pose_visibility is not None else "VISIBILITY: --"
    cv2.putText(
        frame, vis_str, (28, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2, cv2.LINE_AA,
    )

    def fmt(key, label, suffix=""):
        value = None if features is None else features.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"{label}: --"
        return f"{label}: {value:.1f}{suffix}"

    lines = [
        fmt("avg_knee_angle", "KNEE", " deg"),
        fmt("avg_hip_angle", "HIP", " deg"),
        fmt("avg_elbow_angle", "ELBOW", " deg"),
        fmt("avg_shoulder_angle", "SHOULDER", " deg"),
        fmt("torso_angle", "TORSO", " deg"),
        fmt("neck_angle", "NECK", " deg"),
        fmt("shoulder_symmetry", "SHOULDER SYM", ""),
        fmt("hip_symmetry", "HIP SYM", ""),
        fmt("shoulder_width", "SHOULDER W", ""),
        fmt("stance_width", "STANCE W", ""),
    ]

    y = 104
    for line in lines:
        cv2.putText(
            frame, line, (28, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )
        y += 24

    if not pose_detected:
        cv2.putText(
            frame, "No person detected", (20, 375),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    cv2.putText(
        frame, "[q]=quit", (20, 402),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
    )


# =============================================================================
# 10. Main Execution Loop
# =============================================================================

def main():
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    visibility_smoother = MovingAverage(SMOOTHING_WINDOW)

    frames_with_pose = 0
    frames_with_reliable_features = 0

    session_start = time.time()

    print("Generic pose feature extraction running. Press [q] to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB for MediaPipe Tasks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        pose_detected = False
        pose_visibility = None
        features = None

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            pose_detected = True
            frames_with_pose += 1

            # Draw skeleton connections & landmarks
            draw_pose_skeleton(frame, pose)

            row = {"timestamp": timestamp_ms}

            # Raw x/y/z/visibility for all 33 landmarks — always logged
            for idx, landmark in enumerate(pose):
                name = LANDMARK_NAMES[idx]
                row[f"{name}_x"] = round(landmark.x, 4)
                row[f"{name}_y"] = round(landmark.y, 4)
                row[f"{name}_z"] = round(landmark.z, 4)
                row[f"{name}_visibility"] = round(landmark.visibility, 4)

            raw_visibility = min(pose[i].visibility for i in REQUIRED_LANDMARKS)
            pose_visibility = visibility_smoother.update(raw_visibility)
            row["pose_visibility"] = round(pose_visibility, 3)

            if pose_visibility >= VISIBILITY_THRESHOLD:
                # Reliable pose: compute the full generic feature set
                features = extract_features(pose)
                frames_with_reliable_features += 1

                for key in FEATURE_COLUMNS:
                    value = features.get(key)
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        row[key] = None
                    else:
                        row[key] = round(float(value), 4)
            else:
                # Low visibility: raw landmarks still logged, features left blank
                for key in FEATURE_COLUMNS:
                    row[key] = None
        else:
            # No person detected in frame
            pose_visibility = visibility_smoother.update(0.0)
            row = empty_row(timestamp_ms)
            row["pose_visibility"] = round(pose_visibility, 3)

        buffer.append(row)
        if len(buffer) >= FLUSH_EVERY:
            flush_buffer(buffer, CSV_PATH)

        draw_hud(frame, features, pose_visibility, pose_detected)

        cv2.imshow("Pose Feature Extraction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    flush_buffer(buffer, CSV_PATH)

    landmarker.close()

    # Session Summary
    session_duration = time.time() - session_start
    print("\n----- Session Summary -----")
    print(f"Frames processed: {frame_idx}")
    print(f"Session duration: {session_duration:.1f} sec")
    print(f"Frames with pose detected: {frames_with_pose}")
    print(f"Frames with reliable features: {frames_with_reliable_features}")
    print(f"Saved to: {CSV_PATH}")
    print("----------------------------")


if __name__ == "__main__":
    main()