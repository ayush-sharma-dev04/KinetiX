import os
import time
from collections import deque

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
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


# =============================================================================
# 2. Landmark Indices & Connections
# =============================================================================

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

REQUIRED_LANDMARKS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
]

# Standard 33-point pose skeleton connections (index pairs).
POSE_CONNECTIONS = [
    (conn.start, conn.end) for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS
]


# =============================================================================
# 3. Video Source & Configuration
# =============================================================================

# Live webcam (0). To use a recorded video file, set e.g. VIDEO_SOURCE = "downward_dog_video.mp4"
VIDEO_SOURCE = 0

POSE_NAME = "Downward-Facing Dog"
POSE_LABEL = "Downward Dog"

# Pose-Specific Form Thresholds (Downward Dog / Adho Mukha Svanasana)
# Inverted-V body geometry:
HIP_ANGLE_MIN = 55.0              # Min shoulder-hip-ankle fold angle (deg)
HIP_ANGLE_MAX = 110.0             # Max shoulder-hip-ankle fold angle (deg)
ELBOW_ANGLE_MIN = 155.0           # Min arm extension angle (deg)
SHOULDER_ANGLE_MIN = 150.0        # Min arm-torso overhead alignment angle (deg)
KNEE_ANGLE_MIN = 140.0            # Min leg extension angle (deg) (allows tight hamstring variant)
SYMMETRY_TILT_MAX = 15.0          # Max shoulder/hip tilt asymmetry (deg)

# Temporal smoothing
SMOOTHING_WINDOW = 5              # Frames for sliding-window moving-average filter

# Visibility thresholds
VISIBILITY_THRESHOLD = 0.5        # Smoothed visibility required to trust pose & drive hold timer
DRAW_VISIBILITY_THRESHOLD = 0.2   # Landmarks/connections below this are not drawn (noise filter)

# Common CSV Logging Configuration
# -----------------------------------------------------------------------------
# ML DATASET ARCHITECTURE NOTE:
# - All seven yoga pose files contribute correct pose samples to this shared CSV.
# - Columns represent the complete union of numerical features across all poses.
# - Unused/inapplicable features for a specific pose are populated with NaN.
# - Samples are recorded ONLY when the target pose is CORRECT.
# - No timestamps, frame counters, feedback strings, or hold timers are stored.
# -----------------------------------------------------------------------------
CSV_PATH = "yoga_pose_dataset.csv"
FLUSH_EVERY = 30

CSV_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "avg_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "avg_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "avg_shoulder_angle",
    "left_body_line_angle",
    "right_body_line_angle",
    "avg_body_line_angle",
    "torso_angle",
    "torso_to_horizontal",
    "left_thigh_to_horizontal",
    "right_thigh_to_horizontal",
    "left_arm_to_horizontal",
    "right_arm_to_horizontal",
    "shoulder_tilt_angle",
    "hip_tilt_angle",
    "shoulder_symmetry",
    "hip_symmetry",
    "shoulder_width",
    "hip_width",
    "stance_width",
    "stance_to_shoulder_ratio",
    "stance_to_hip_ratio",
    "shoulder_to_hip_ratio",
    "neck_angle",
    "head_offset_x",
    "head_offset_ratio",
    "nose_to_mid_shoulder_dist",
    "pose_visibility",
    "front_knee_angle",
    "back_knee_angle",
    "front_hip_angle",
    "back_hip_angle",
    "standing_knee_angle",
    "lifted_knee_angle",
    "standing_hip_angle",
    "lifted_hip_angle",
    "top_arm_shoulder_angle",
    "bottom_arm_shoulder_angle",
    "top_arm_elbow_angle",
    "bottom_arm_elbow_angle",
    "hip_sag_offset",
    "lifted_foot_to_standing_knee_dist",
    "front_knee_ankle_x_offset",
    "pose_label",
]


# =============================================================================
# 4. Geometry Helpers
# =============================================================================

def _xy(lm):
    """Convert landmark to (x, y) numpy array."""
    return np.array([lm.x, lm.y])


def _midpoint(a, b):
    """Midpoint (x, y) of two landmarks."""
    return (_xy(a) + _xy(b)) / 2.0


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees [0, 180]."""
    a, b, c = _xy(a), _xy(b), _xy(c)
    ba = a - b
    bc = c - b

    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


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
    """Angle (degrees) of a 2D vector relative to horizontal (0 = level, [0, 90])."""
    horizontal_vec = np.array([1.0, 0.0])
    denom = np.linalg.norm(vec) * np.linalg.norm(horizontal_vec)
    if denom < 1e-8:
        return 0.0

    cos_angle = np.dot(vec, horizontal_vec) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_angle)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def calculate_torso_angle(shoulder_l, shoulder_r, hip_l, hip_r):
    """
    Angle of the torso relative to vertical (0 = perfectly upright).
    Uses shoulder/hip midpoints to remain robust against torso rotation.
    """
    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    return vertical_angle_of_vector(shoulder_mid - hip_mid)


def calculate_torso_to_horizontal(shoulder_l, shoulder_r, hip_l, hip_r):
    """Angle of the torso relative to horizontal (0 = parallel to floor)."""
    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    return horizontal_angle_of_vector(shoulder_mid - hip_mid)


def safe_div(numerator, denominator):
    """Safe division returning np.nan on zero or invalid denominator."""
    if denominator is None or np.isnan(denominator) or abs(denominator) < 1e-8:
        return np.nan
    return float(numerator / denominator)


# =============================================================================
# 5. Feature Extraction
# =============================================================================

def extract_features(pose):
    """
    Given the 33 MediaPipe pose landmarks for one frame, compute a flat dict
    of reusable numerical features.
    """
    features = {}

    # Knee angles (hip - knee - ankle)
    left_knee = calculate_angle(pose[LEFT_HIP], pose[LEFT_KNEE], pose[LEFT_ANKLE])
    right_knee = calculate_angle(pose[RIGHT_HIP], pose[RIGHT_KNEE], pose[RIGHT_ANKLE])
    features["left_knee_angle"] = left_knee
    features["right_knee_angle"] = right_knee
    features["avg_knee_angle"] = (left_knee + right_knee) / 2.0

    # Hip angles (shoulder - hip - knee)
    left_hip = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_KNEE])
    right_hip = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_KNEE])
    features["left_hip_angle"] = left_hip
    features["right_hip_angle"] = right_hip
    features["avg_hip_angle"] = (left_hip + right_hip) / 2.0

    # Elbow angles (shoulder - elbow - wrist)
    left_elbow = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST])
    right_elbow = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST])
    features["left_elbow_angle"] = left_elbow
    features["right_elbow_angle"] = right_elbow
    features["avg_elbow_angle"] = (left_elbow + right_elbow) / 2.0

    # Shoulder angles (elbow - shoulder - hip)
    left_shoulder_ang = calculate_angle(pose[LEFT_ELBOW], pose[LEFT_SHOULDER], pose[LEFT_HIP])
    right_shoulder_ang = calculate_angle(pose[RIGHT_ELBOW], pose[RIGHT_SHOULDER], pose[RIGHT_HIP])
    features["left_shoulder_angle"] = left_shoulder_ang
    features["right_shoulder_angle"] = right_shoulder_ang
    features["avg_shoulder_angle"] = (left_shoulder_ang + right_shoulder_ang) / 2.0

    # Full body line / hip fold angles (shoulder - hip - ankle)
    left_body_line = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_ANKLE])
    right_body_line = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_ANKLE])
    features["left_body_line_angle"] = left_body_line
    features["right_body_line_angle"] = right_body_line
    features["avg_body_line_angle"] = (left_body_line + right_body_line) / 2.0

    # Torso angles
    features["torso_angle"] = calculate_torso_angle(
        pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER],
        pose[LEFT_HIP], pose[RIGHT_HIP]
    )
    features["torso_to_horizontal"] = calculate_torso_to_horizontal(
        pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER],
        pose[LEFT_HIP], pose[RIGHT_HIP]
    )

    # Thigh & arm orientations
    features["left_thigh_to_horizontal"] = horizontal_angle_of_vector(_xy(pose[LEFT_KNEE]) - _xy(pose[LEFT_HIP]))
    features["right_thigh_to_horizontal"] = horizontal_angle_of_vector(_xy(pose[RIGHT_KNEE]) - _xy(pose[RIGHT_HIP]))
    features["left_arm_to_horizontal"] = horizontal_angle_of_vector(_xy(pose[LEFT_WRIST]) - _xy(pose[LEFT_SHOULDER]))
    features["right_arm_to_horizontal"] = horizontal_angle_of_vector(_xy(pose[RIGHT_WRIST]) - _xy(pose[RIGHT_SHOULDER]))

    # Midpoints & symmetry
    mid_shoulder = _midpoint(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
    mid_hip = _midpoint(pose[LEFT_HIP], pose[RIGHT_HIP])
    mid_ankle = _midpoint(pose[LEFT_ANKLE], pose[RIGHT_ANKLE])
    mid_wrist = _midpoint(pose[LEFT_WRIST], pose[RIGHT_WRIST])

    features["shoulder_symmetry"] = float(abs(pose[LEFT_SHOULDER].y - pose[RIGHT_SHOULDER].y))
    features["hip_symmetry"] = float(abs(pose[LEFT_HIP].y - pose[RIGHT_HIP].y))
    features["shoulder_tilt_angle"] = horizontal_angle_of_vector(_xy(pose[RIGHT_SHOULDER]) - _xy(pose[LEFT_SHOULDER]))
    features["hip_tilt_angle"] = horizontal_angle_of_vector(_xy(pose[RIGHT_HIP]) - _xy(pose[LEFT_HIP]))

    # Widths & ratios
    shoulder_width = calculate_distance(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
    hip_width = calculate_distance(pose[LEFT_HIP], pose[RIGHT_HIP])
    stance_width = calculate_distance(pose[LEFT_ANKLE], pose[RIGHT_ANKLE])

    features["shoulder_width"] = shoulder_width
    features["hip_width"] = hip_width
    features["stance_width"] = stance_width
    features["stance_to_shoulder_ratio"] = safe_div(stance_width, shoulder_width)
    features["stance_to_hip_ratio"] = safe_div(stance_width, hip_width)
    features["shoulder_to_hip_ratio"] = safe_div(shoulder_width, hip_width)

    # Head / neck alignment
    nose_xy = _xy(pose[NOSE])
    features["neck_angle"] = vertical_angle_of_vector(nose_xy - mid_shoulder)
    features["head_offset_x"] = float(nose_xy[0] - mid_shoulder[0])
    features["head_offset_ratio"] = safe_div(nose_xy[0] - mid_shoulder[0], shoulder_width)
    features["nose_to_mid_shoulder_dist"] = float(np.linalg.norm(nose_xy - mid_shoulder))

    # Downward Dog specific relative heights (in image space, smaller y = higher)
    features["hip_mid_y"] = float(mid_hip[1])
    features["shoulder_mid_y"] = float(mid_shoulder[1])
    features["ankle_mid_y"] = float(mid_ankle[1])
    features["wrist_mid_y"] = float(mid_wrist[1])
    features["nose_y"] = float(pose[NOSE].y)

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
            return (0, 255, 0)
        return (0, 165, 255)

    # Draw connection lines first so landmark dots appear on top
    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = pose[start_idx]
        end_lm = pose[end_idx]

        if start_lm.visibility < DRAW_VISIBILITY_THRESHOLD or end_lm.visibility < DRAW_VISIBILITY_THRESHOLD:
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
        self.values = deque(maxlen=window_size)

    def update(self, value):
        if value is None or np.isnan(value):
            return None
        self.values.append(value)
        return sum(self.values) / len(self.values)

    @property
    def ready(self):
        return len(self.values) > 0


# =============================================================================
# 8. Pose Recognition & Form Validation
# =============================================================================

def is_downward_dog_detected(feat, pose):
    """
    Detects if the user is in an Inverted-V body shape (Downward Dog):
    - Hips are the highest point (hip_y < shoulder_y and hip_y < ankle_y)
    - Hands and feet are both on the ground (wrists and ankles lower than hips)
    - Distinct hip fold angle
    """
    hip_y = feat["hip_mid_y"]
    sh_y = feat["shoulder_mid_y"]
    ak_y = feat["ankle_mid_y"]
    wr_y = feat["wrist_mid_y"]

    # Inverted-V check: hips must be strictly higher in real space (smaller y in image space)
    # than shoulders and ankles
    if not (hip_y < sh_y and hip_y < ak_y):
        return False

    # Hands and feet grounded (well below hips)
    if not (wr_y > hip_y + 0.08 and ak_y > hip_y + 0.08):
        return False

    # Hip fold angle check (distinguishes from plank or standing)
    if feat["avg_body_line_angle"] > 135.0 and feat["avg_hip_angle"] > 135.0:
        return False

    return True


def evaluate_downward_dog_form(feat):
    """
    Validates Downward Dog form against researched biomechanical thresholds.
    Returns (is_correct, feedback_message, failed_rules).
    """
    issues = []

    # 1. Hip fold / Inverted-V angle
    hip_fold = feat["avg_body_line_angle"]
    if hip_fold > HIP_ANGLE_MAX:
        issues.append(f"Lift hips higher ({hip_fold:.1f}° > {HIP_ANGLE_MAX}°)")
    elif hip_fold < HIP_ANGLE_MIN:
        issues.append(f"Open hip angle ({hip_fold:.1f}° < {HIP_ANGLE_MIN}°)")

    # 2. Arm extension (elbow angle)
    if feat["avg_elbow_angle"] < ELBOW_ANGLE_MIN:
        issues.append(f"Straighten arms fully ({feat['avg_elbow_angle']:.1f}° < {ELBOW_ANGLE_MIN}°)")

    # 3. Shoulder overhead alignment (push chest toward thighs)
    if feat["avg_shoulder_angle"] < SHOULDER_ANGLE_MIN:
        issues.append(f"Open shoulders / push chest back ({feat['avg_shoulder_angle']:.1f}° < {SHOULDER_ANGLE_MIN}°)")

    # 4. Leg extension (knee angle)
    if feat["avg_knee_angle"] < KNEE_ANGLE_MIN:
        issues.append(f"Straighten legs ({feat['avg_knee_angle']:.1f}° < {KNEE_ANGLE_MIN}°)")

    # 5. Head / neck relaxation (nose hanging in line with or below shoulders)
    if feat["nose_y"] < feat["shoulder_mid_y"] - 0.06:
        issues.append("Relax neck (gaze toward navel, don't look up)")

    # 6. Bilateral symmetry / level weight distribution
    if feat["shoulder_tilt_angle"] > SYMMETRY_TILT_MAX or feat["hip_tilt_angle"] > SYMMETRY_TILT_MAX:
        issues.append("Distribute weight evenly on both sides")

    if not issues:
        return True, "Good Form! Hold Downward Dog!", []
    return False, issues[0], issues


# =============================================================================
# 9. Pose Hold Timer
# =============================================================================

class PoseHoldTimer:
    """
    Tracks static pose hold duration.
    Only increments hold timer when form is CORRECT.
    Resets timer to 0 immediately if form becomes INCORRECT/ADJUST or POSE NOT DETECTED.
    """

    def __init__(self):
        self.hold_start_time = None
        self.current_hold_time = 0.0
        self.max_hold_time = 0.0
        self.total_correct_samples = 0

    def update(self, is_correct):
        now = time.time()
        if is_correct:
            if self.hold_start_time is None:
                self.hold_start_time = now
                self.current_hold_time = 0.0
            else:
                self.current_hold_time = now - self.hold_start_time
            self.max_hold_time = max(self.max_hold_time, self.current_hold_time)
            self.total_correct_samples += 1
        else:
            self.hold_start_time = None
            self.current_hold_time = 0.0

        return self.current_hold_time

    def reset(self):
        self.hold_start_time = None
        self.current_hold_time = 0.0


# =============================================================================
# 10. CSV Logging Helper
# =============================================================================

buffer = []


def flush_buffer(rows, path=CSV_PATH):
    """Flush buffered correct pose frame rows to CSV dataset."""
    if not rows:
        return

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )
    rows.clear()


# =============================================================================
# 11. HUD Rendering (Translucent Card Overlay)
# =============================================================================

STATUS_COLORS = {
    "CORRECT": (0, 255, 0),         # Green
    "ADJUST FORM": (0, 215, 255),    # Yellow / Amber
    "POSE NOT DETECTED": (0, 0, 255),# Red
}


def draw_hud(frame, status, hold_time, max_hold, feedback, feat_display, visibility, pose_detected):
    """
    Renders a modern translucent dark card HUD in the top-left corner:
    ┌──────────────────────────────────────────┐
    │ POSE: DOWNWARD-FACING DOG                │
    │ STATUS: CORRECT                          │
    │ HOLD TIME: 05.4s  (MAX: 12.0s)           │
    │ FEEDBACK: Good Form! Hold Downward Dog!  │
    │                                          │
    │ HIP FOLD:        82.4°                   │
    │ SHOULDER ANGLE: 168.1°                   │
    │ AVG ELBOW:      174.5°                   │
    │ AVG KNEE:       165.0°                   │
    │ VISIBILITY:      0.94                    │
    └──────────────────────────────────────────┘
    """
    # Create translucent dark overlay box (top-left)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (15, 15),
        (440, 290),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame,
    )

    # Pose title
    cv2.putText(
        frame,
        f"POSE: {POSE_NAME.upper()}",
        (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Status indicator & dynamic color
    status_color = STATUS_COLORS.get(status, (255, 255, 255))
    cv2.putText(
        frame,
        f"STATUS: {status}",
        (28, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Hold Timer
    timer_color = (0, 255, 0) if status == "CORRECT" else (200, 200, 200)
    cv2.putText(
        frame,
        f"HOLD TIME: {hold_time:.1f}s   (MAX: {max_hold:.1f}s)",
        (28, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        timer_color,
        2,
        cv2.LINE_AA,
    )

    # Form feedback
    fb_color = (0, 255, 0) if status == "CORRECT" else (0, 215, 255) if status == "ADJUST FORM" else (160, 160, 160)
    cv2.putText(
        frame,
        f"FEEDBACK: {feedback}",
        (28, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        fb_color,
        1,
        cv2.LINE_AA,
    )

    # Display key angles/features
    y_pos = 168
    for label, val in feat_display.items():
        val_str = f"{val:.1f}°" if isinstance(val, (int, float)) and not np.isnan(val) else "--"
        cv2.putText(
            frame,
            f"{label}: {val_str}",
            (28, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y_pos += 26

    # Visibility score with color warning for low confidence
    if visibility is not None:
        vis_str = f"VISIBILITY: {visibility:.2f}"
        vis_col = (240, 240, 240) if visibility >= VISIBILITY_THRESHOLD else (0, 165, 255)
    else:
        vis_str = "VISIBILITY: --"
        vis_col = (0, 165, 255)

    cv2.putText(
        frame,
        vis_str,
        (28, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        vis_col,
        1,
        cv2.LINE_AA,
    )

    # Warning / Hint below the card
    if not pose_detected:
        cv2.putText(
            frame,
            "No person detected",
            (20, 318),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "[q]=quit   [r]=reset timer",
        (20, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


# =============================================================================
# 12. Main Execution Loop
# =============================================================================

def main():
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Initialize feature smoothers
    torso_smoother = MovingAverage(SMOOTHING_WINDOW)
    knee_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    knee_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    hip_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    hip_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    elbow_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    elbow_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    shoulder_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    shoulder_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    body_line_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    body_line_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    sh_tilt_smoother = MovingAverage(SMOOTHING_WINDOW)
    hp_tilt_smoother = MovingAverage(SMOOTHING_WINDOW)
    visibility_smoother = MovingAverage(SMOOTHING_WINDOW)

    hold_timer = PoseHoldTimer()
    session_start = time.time()

    print(f"Automatic {POSE_NAME} hold detection running. Press [q] to quit.")
    print(f"Logging correct samples to: {CSV_PATH}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB for MediaPipe Tasks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status = "POSE NOT DETECTED"
        feedback = "Position yourself in camera view"
        smoothed_visibility = None
        pose_detected = False
        feat_display = {
            "HIP FOLD": np.nan,
            "SHOULDER ANGLE": np.nan,
            "AVG ELBOW": np.nan,
            "AVG KNEE": np.nan,
        }

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            pose_detected = True

            # Draw skeleton connections & landmarks
            draw_pose_skeleton(frame, pose)

            raw_visibility = min(pose[i].visibility for i in REQUIRED_LANDMARKS)
            smoothed_visibility = visibility_smoother.update(raw_visibility)

            if smoothed_visibility >= VISIBILITY_THRESHOLD:
                raw_feat = extract_features(pose)

                # Smooth key angles
                feat = raw_feat.copy()
                feat["torso_angle"] = torso_smoother.update(raw_feat["torso_angle"])
                feat["left_knee_angle"] = knee_l_smoother.update(raw_feat["left_knee_angle"])
                feat["right_knee_angle"] = knee_r_smoother.update(raw_feat["right_knee_angle"])
                feat["avg_knee_angle"] = (feat["left_knee_angle"] + feat["right_knee_angle"]) / 2.0

                feat["left_hip_angle"] = hip_l_smoother.update(raw_feat["left_hip_angle"])
                feat["right_hip_angle"] = hip_r_smoother.update(raw_feat["right_hip_angle"])
                feat["avg_hip_angle"] = (feat["left_hip_angle"] + feat["right_hip_angle"]) / 2.0

                feat["left_elbow_angle"] = elbow_l_smoother.update(raw_feat["left_elbow_angle"])
                feat["right_elbow_angle"] = elbow_r_smoother.update(raw_feat["right_elbow_angle"])
                feat["avg_elbow_angle"] = (feat["left_elbow_angle"] + feat["right_elbow_angle"]) / 2.0

                feat["left_shoulder_angle"] = shoulder_l_smoother.update(raw_feat["left_shoulder_angle"])
                feat["right_shoulder_angle"] = shoulder_r_smoother.update(raw_feat["right_shoulder_angle"])
                feat["avg_shoulder_angle"] = (feat["left_shoulder_angle"] + feat["right_shoulder_angle"]) / 2.0

                feat["left_body_line_angle"] = body_line_l_smoother.update(raw_feat["left_body_line_angle"])
                feat["right_body_line_angle"] = body_line_r_smoother.update(raw_feat["right_body_line_angle"])
                feat["avg_body_line_angle"] = (feat["left_body_line_angle"] + feat["right_body_line_angle"]) / 2.0

                feat["shoulder_tilt_angle"] = sh_tilt_smoother.update(raw_feat["shoulder_tilt_angle"])
                feat["hip_tilt_angle"] = hp_tilt_smoother.update(raw_feat["hip_tilt_angle"])

                feat_display = {
                    "HIP FOLD": feat["avg_body_line_angle"],
                    "SHOULDER ANGLE": feat["avg_shoulder_angle"],
                    "AVG ELBOW": feat["avg_elbow_angle"],
                    "AVG KNEE": feat["avg_knee_angle"],
                }

                if is_downward_dog_detected(feat, pose):
                    is_correct, fb_msg, _ = evaluate_downward_dog_form(feat)
                    if is_correct:
                        status = "CORRECT"
                        feedback = fb_msg
                        hold_timer.update(True)

                        # Build ML dataset row with complete feature union
                        row = {
                            "left_knee_angle": round(feat["left_knee_angle"], 2),
                            "right_knee_angle": round(feat["right_knee_angle"], 2),
                            "avg_knee_angle": round(feat["avg_knee_angle"], 2),
                            "left_hip_angle": round(feat["left_hip_angle"], 2),
                            "right_hip_angle": round(feat["right_hip_angle"], 2),
                            "avg_hip_angle": round(feat["avg_hip_angle"], 2),
                            "left_elbow_angle": round(feat["left_elbow_angle"], 2),
                            "right_elbow_angle": round(feat["right_elbow_angle"], 2),
                            "avg_elbow_angle": round(feat["avg_elbow_angle"], 2),
                            "left_shoulder_angle": round(feat["left_shoulder_angle"], 2),
                            "right_shoulder_angle": round(feat["right_shoulder_angle"], 2),
                            "avg_shoulder_angle": round(feat["avg_shoulder_angle"], 2),
                            "left_body_line_angle": round(feat["left_body_line_angle"], 2),
                            "right_body_line_angle": round(feat["right_body_line_angle"], 2),
                            "avg_body_line_angle": round(feat["avg_body_line_angle"], 2),
                            "torso_angle": round(feat["torso_angle"], 2),
                            "torso_to_horizontal": round(feat["torso_to_horizontal"], 2),
                            "left_thigh_to_horizontal": round(feat["left_thigh_to_horizontal"], 2),
                            "right_thigh_to_horizontal": round(feat["right_thigh_to_horizontal"], 2),
                            "left_arm_to_horizontal": round(feat["left_arm_to_horizontal"], 2),
                            "right_arm_to_horizontal": round(feat["right_arm_to_horizontal"], 2),
                            "shoulder_tilt_angle": round(feat["shoulder_tilt_angle"], 2),
                            "hip_tilt_angle": round(feat["hip_tilt_angle"], 2),
                            "shoulder_symmetry": round(feat["shoulder_symmetry"], 4),
                            "hip_symmetry": round(feat["hip_symmetry"], 4),
                            "shoulder_width": round(feat["shoulder_width"], 4),
                            "hip_width": round(feat["hip_width"], 4),
                            "stance_width": round(feat["stance_width"], 4),
                            "stance_to_shoulder_ratio": round(feat["stance_to_shoulder_ratio"], 3),
                            "stance_to_hip_ratio": round(feat["stance_to_hip_ratio"], 3),
                            "shoulder_to_hip_ratio": round(feat["shoulder_to_hip_ratio"], 3),
                            "neck_angle": round(feat["neck_angle"], 2),
                            "head_offset_x": round(feat["head_offset_x"], 4),
                            "head_offset_ratio": round(feat["head_offset_ratio"], 3),
                            "nose_to_mid_shoulder_dist": round(feat["nose_to_mid_shoulder_dist"], 4),
                            "pose_visibility": round(smoothed_visibility, 3),
                            "front_knee_angle": np.nan,
                            "back_knee_angle": np.nan,
                            "front_hip_angle": np.nan,
                            "back_hip_angle": np.nan,
                            "standing_knee_angle": np.nan,
                            "lifted_knee_angle": np.nan,
                            "standing_hip_angle": np.nan,
                            "lifted_hip_angle": np.nan,
                            "top_arm_shoulder_angle": np.nan,
                            "bottom_arm_shoulder_angle": np.nan,
                            "top_arm_elbow_angle": np.nan,
                            "bottom_arm_elbow_angle": np.nan,
                            "hip_sag_offset": np.nan,
                            "lifted_foot_to_standing_knee_dist": np.nan,
                            "front_knee_ankle_x_offset": np.nan,
                            "pose_label": POSE_LABEL,
                        }
                        buffer.append(row)
                        if len(buffer) >= FLUSH_EVERY:
                            flush_buffer(buffer, CSV_PATH)
                    else:
                        status = "ADJUST FORM"
                        feedback = fb_msg
                        hold_timer.update(False)
                else:
                    status = "ADJUST FORM"
                    feedback = "Lift hips and ground hands/feet (Inverted V)"
                    hold_timer.update(False)
            else:
                status = "POSE NOT DETECTED"
                feedback = "Low landmark visibility"
                hold_timer.update(False)
        else:
            smoothed_visibility = visibility_smoother.update(0.0)
            status = "POSE NOT DETECTED"
            feedback = "No person detected in frame"
            hold_timer.update(False)

        # Render translucent card HUD overlay
        draw_hud(
            frame,
            status=status,
            hold_time=hold_timer.current_hold_time,
            max_hold=hold_timer.max_hold_time,
            feedback=feedback,
            feat_display=feat_display,
            visibility=smoothed_visibility,
            pose_detected=pose_detected,
        )

        cv2.imshow(f"Yoga Pose Tracker - {POSE_NAME}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            hold_timer.reset()

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    flush_buffer(buffer, CSV_PATH)
    landmarker.close()

    # Session Summary
    session_duration = time.time() - session_start
    print("\n----- Session Summary -----")
    print(f"Pose: {POSE_NAME}")
    print(f"Frames processed: {frame_idx}")
    print(f"Session duration: {session_duration:.1f} sec")
    print(f"Max continuous hold: {hold_timer.max_hold_time:.1f} sec")
    print(f"Correct samples logged: {hold_timer.total_correct_samples}")
    print(f"Dataset path: {CSV_PATH}")
    print("----------------------------\n")


if __name__ == "__main__":
    main()
