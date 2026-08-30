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

# Live webcam (0). To use a recorded video file, set e.g. VIDEO_SOURCE = "plank_video.mp4"
VIDEO_SOURCE = 0

POSE_NAME = "Plank"
POSE_LABEL = "Plank"

# Pose-Specific Form Thresholds (Plank / Utthita Chaturanga Dandasana)
# Straight horizontal body line from shoulders to ankles:
BODY_LINE_MIN = 168.0             # Min shoulder-hip-ankle angle (deg) (straight spine line)
HIP_ANGLE_MIN = 168.0             # Min shoulder-hip-knee angle (deg)
KNEE_STRAIGHT_MIN = 168.0         # Min knee extension angle (deg)
TORSO_HORIZONTAL_MAX = 15.0       # Max torso angle relative to horizontal (deg)
HIGH_PLANK_ELBOW_MIN = 155.0      # Min elbow extension for high plank (deg)
FOREARM_ELBOW_MIN = 75.0          # Min elbow angle for forearm plank (deg)
FOREARM_ELBOW_MAX = 105.0         # Max elbow angle for forearm plank (deg)
SAG_PIKE_OFFSET_MAX = 0.035       # Vertical displacement tolerance of hips from shoulder-ankle line
SHOULDER_TILT_MAX = 10.0          # Max shoulder tilt from horizontal (deg)
HIP_TILT_MAX = 10.0               # Max hip tilt from horizontal (deg)
WRIST_SHOULDER_STACK_MAX = 0.14   # Max horizontal offset between wrists and shoulders (normalized)

# Temporal smoothing
SMOOTHING_WINDOW = 6              # Sliding-window moving average window size

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
    """Angle (degrees) of a 2D vector relative to horizontal (0 = level)."""
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
    """Angle of the torso relative to vertical (0 = perfectly upright)."""
    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    return vertical_angle_of_vector(shoulder_mid - hip_mid)


def calculate_torso_to_horizontal(shoulder_l, shoulder_r, hip_l, hip_r):
    """Angle of the torso relative to horizontal (0 = parallel to floor)."""
    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    return horizontal_angle_of_vector(shoulder_mid - hip_mid)


def calculate_hip_sag_offset(shoulder_mid, hip_mid, ankle_mid):
    """
    Computes signed vertical offset of the hip from the shoulder-ankle reference line.
    In image space (y increases downwards):
      > 0: hip is below the shoulder-ankle line (sagging down / lumbar collapse)
      < 0: hip is above the shoulder-ankle line (piking up / rear elevated)
    """
    sh_x, sh_y = shoulder_mid[0], shoulder_mid[1]
    ak_x, ak_y = ankle_mid[0], ankle_mid[1]
    hp_x, hp_y = hip_mid[0], hip_mid[1]

    dx = ak_x - sh_x
    if abs(dx) < 1e-5:
        # Nearly vertical: use horizontal offset from line
        return float(hp_x - sh_x)

    # Linear interpolation along shoulder-ankle line at hip's x-coordinate
    t = (hp_x - sh_x) / dx
    expected_line_y = sh_y + t * (ak_y - sh_y)

    return float(hp_y - expected_line_y)


def safe_div(numerator, denominator):
    """Safe division returning np.nan on zero or invalid denominator."""
    if denominator is None or np.isnan(denominator) or abs(denominator) < 1e-8:
        return np.nan
    return float(numerator / denominator)


# =============================================================================
# 5. Feature Extraction
# =============================================================================

def extract_features(pose):
    """Extract standard geometric features from 33 MediaPipe pose landmarks."""
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

    # Full body line angles (shoulder - hip - ankle) (critical plank line)
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

    # Signed Hip Sag / Pike Offset
    features["hip_sag_offset"] = calculate_hip_sag_offset(mid_shoulder, mid_hip, mid_ankle)

    # Wrist under shoulder stacking offset
    features["wrist_shoulder_x_offset"] = float(abs(mid_wrist[0] - mid_shoulder[0]))

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

    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = pose[start_idx]
        end_lm = pose[end_idx]

        if start_lm.visibility < DRAW_VISIBILITY_THRESHOLD or end_lm.visibility < DRAW_VISIBILITY_THRESHOLD:
            continue

        edge_color = color_for(min(start_lm.visibility, end_lm.visibility))
        cv2.line(frame, point_px(start_lm), point_px(end_lm), edge_color, 2)

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

def is_plank_detected(feat, pose):
    """
    Detects if the user is in a horizontal plank posture:
    - Body is roughly horizontal (torso_to_horizontal <= 30° or torso_angle >= 60°)
    - Prone support (hands/forearms and feet on the ground)
    - Body line extended (body_line_angle >= 140°)
    """
    # Torso must be near horizontal
    if feat["torso_to_horizontal"] > 30.0 and feat["torso_angle"] < 60.0:
        return False

    # Hands and feet must be grounded (lower down in frame, y > 0.25)
    if pose[LEFT_WRIST].y < 0.20 and pose[RIGHT_WRIST].y < 0.20:
        return False

    # Body line roughly straight
    if feat["avg_body_line_angle"] < 135.0:
        return False

    # Legs extended
    if feat["avg_knee_angle"] < 140.0:
        return False

    return True


def evaluate_plank_form(feat):
    """
    Validates Plank form against researched biomechanical thresholds.
    Returns (is_correct, feedback_message, failed_rules).
    """
    issues = []

    # 1. Full Body Line & Sag / Pike Direction Check
    body_line = feat["avg_body_line_angle"]
    sag_offset = feat.get("hip_sag_offset", 0.0)

    if body_line < BODY_LINE_MIN:
        if sag_offset > SAG_PIKE_OFFSET_MAX:
            issues.append(f"Engage core and lift sagging hips ({body_line:.1f}° < {BODY_LINE_MIN}°)")
        elif sag_offset < -SAG_PIKE_OFFSET_MAX:
            issues.append(f"Lower hips into straight line ({body_line:.1f}° < {BODY_LINE_MIN}°)")
        else:
            issues.append(f"Straighten body line ({body_line:.1f}° < {BODY_LINE_MIN}°)")

    # 2. Hip Angle (Shoulder - Hip - Knee)
    if feat["avg_hip_angle"] < HIP_ANGLE_MIN:
        issues.append(f"Align hips with spine ({feat['avg_hip_angle']:.1f}° < {HIP_ANGLE_MIN}°)")

    # 3. Knee extension (straight legs, pushing through heels)
    if feat["avg_knee_angle"] < KNEE_STRAIGHT_MIN:
        issues.append(f"Straighten legs and engage quads ({feat['avg_knee_angle']:.1f}° < {KNEE_STRAIGHT_MIN}°)")

    # 4. Arm / Elbow alignment (High Plank or Forearm Plank)
    avg_elbow = feat["avg_elbow_angle"]
    is_high_plank = avg_elbow >= HIGH_PLANK_ELBOW_MIN
    is_forearm_plank = FOREARM_ELBOW_MIN <= avg_elbow <= FOREARM_ELBOW_MAX

    if not (is_high_plank or is_forearm_plank):
        issues.append("Extend arms straight (high plank) or 90° (forearm)")

    # 5. Torso parallel to floor
    if feat["torso_to_horizontal"] > TORSO_HORIZONTAL_MAX:
        issues.append(f"Keep torso parallel to floor ({feat['torso_to_horizontal']:.1f}° > {TORSO_HORIZONTAL_MAX}°)")

    # 6. Shoulders and hips level (avoid twisting)
    if feat["shoulder_tilt_angle"] > SHOULDER_TILT_MAX or feat["hip_tilt_angle"] > HIP_TILT_MAX:
        issues.append("Keep shoulders and hips level (no twisting)")

    # 7. Hands stacked under shoulders (for high plank)
    if is_high_plank and feat.get("wrist_shoulder_x_offset", 0) > WRIST_SHOULDER_STACK_MAX:
        issues.append("Stack wrists directly under shoulders")

    if not issues:
        return True, "Good Form! Hold Plank!", []
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

# State / Status color mappings (BGR format)
STATUS_COLORS = {
    "CORRECT": (0, 255, 0),         # Green
    "ADJUST FORM": (0, 215, 255),    # Amber / Yellow
    "POSE NOT DETECTED": (0, 0, 255),# Red
}


def draw_hud(frame, status, hold_time, max_hold, feedback, feat_display, visibility, pose_detected):
    """
    Renders a modern translucent dark card HUD in the top-left corner:
    ┌──────────────────────────────────────────┐
    │ POSE: PLANK                              │
    │ STATUS: CORRECT                          │
    │ HOLD TIME: 05.4s  (MAX: 12.0s)           │
    │ FEEDBACK: Good Form! Hold Plank!         │
    │                                          │
    │ BODY LINE:     175.2°                    │
    │ AVG HIP:       172.8°                    │
    │ AVG KNEE:      176.4°                    │
    │ TORSO HORIZ:     4.2°                    │
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
        0.75,
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
    body_line_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    body_line_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    hip_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    hip_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    knee_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    knee_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    elbow_l_smoother = MovingAverage(SMOOTHING_WINDOW)
    elbow_r_smoother = MovingAverage(SMOOTHING_WINDOW)
    torso_horiz_smoother = MovingAverage(SMOOTHING_WINDOW)
    sag_offset_smoother = MovingAverage(SMOOTHING_WINDOW)
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status = "POSE NOT DETECTED"
        feedback = "Position yourself in camera view"
        smoothed_visibility = None
        pose_detected = False
        feat_display = {
            "BODY LINE": np.nan,
            "AVG HIP": np.nan,
            "AVG KNEE": np.nan,
            "TORSO HORIZ": np.nan,
        }

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            pose_detected = True
            draw_pose_skeleton(frame, pose)

            raw_visibility = min(pose[i].visibility for i in REQUIRED_LANDMARKS)
            smoothed_visibility = visibility_smoother.update(raw_visibility)

            if smoothed_visibility >= VISIBILITY_THRESHOLD:
                raw_feat = extract_features(pose)

                # Smooth key angles
                feat = raw_feat.copy()
                feat["left_body_line_angle"] = body_line_l_smoother.update(raw_feat["left_body_line_angle"])
                feat["right_body_line_angle"] = body_line_r_smoother.update(raw_feat["right_body_line_angle"])
                feat["avg_body_line_angle"] = (feat["left_body_line_angle"] + feat["right_body_line_angle"]) / 2.0

                feat["left_hip_angle"] = hip_l_smoother.update(raw_feat["left_hip_angle"])
                feat["right_hip_angle"] = hip_r_smoother.update(raw_feat["right_hip_angle"])
                feat["avg_hip_angle"] = (feat["left_hip_angle"] + feat["right_hip_angle"]) / 2.0

                feat["left_knee_angle"] = knee_l_smoother.update(raw_feat["left_knee_angle"])
                feat["right_knee_angle"] = knee_r_smoother.update(raw_feat["right_knee_angle"])
                feat["avg_knee_angle"] = (feat["left_knee_angle"] + feat["right_knee_angle"]) / 2.0

                feat["left_elbow_angle"] = elbow_l_smoother.update(raw_feat["left_elbow_angle"])
                feat["right_elbow_angle"] = elbow_r_smoother.update(raw_feat["right_elbow_angle"])
                feat["avg_elbow_angle"] = (feat["left_elbow_angle"] + feat["right_elbow_angle"]) / 2.0

                feat["torso_to_horizontal"] = torso_horiz_smoother.update(raw_feat["torso_to_horizontal"])
                feat["hip_sag_offset"] = sag_offset_smoother.update(raw_feat["hip_sag_offset"])
                feat["shoulder_tilt_angle"] = sh_tilt_smoother.update(raw_feat["shoulder_tilt_angle"])
                feat["hip_tilt_angle"] = hp_tilt_smoother.update(raw_feat["hip_tilt_angle"])

                feat_display = {
                    "BODY LINE": feat["avg_body_line_angle"],
                    "AVG HIP": feat["avg_hip_angle"],
                    "AVG KNEE": feat["avg_knee_angle"],
                    "TORSO HORIZ": feat["torso_to_horizontal"],
                }

                if is_plank_detected(feat, pose):
                    is_correct, fb_msg, _ = evaluate_plank_form(feat)
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
                            "hip_sag_offset": round(feat["hip_sag_offset"], 4),
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
                    feedback = "Align body in a straight plank line"
                    hold_timer.update(False)
            else:
                status = "POSE NOT DETECTED"
                feedback = "Low landmark visibility - adjust lighting/camera"
                hold_timer.update(False)
        else:
            smoothed_visibility = visibility_smoother.update(0.0)
            status = "POSE NOT DETECTED"
            feedback = "No person detected in frame"
            hold_timer.update(False)

        # Draw HUD overlay
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
    duration = time.time() - session_start
    print("\n----- Session Summary -----")
    print(f"Pose: {POSE_NAME}")
    print(f"Frames processed: {frame_idx}")
    print(f"Session duration: {duration:.1f} sec")
    print(f"Max continuous hold: {hold_timer.max_hold_time:.1f} sec")
    print(f"Correct samples logged: {hold_timer.total_correct_samples}")
    print(f"Dataset path: {CSV_PATH}")
    print("----------------------------\n")


if __name__ == "__main__":
    main()
