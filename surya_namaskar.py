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

# Live webcam (0). To use a recorded video file, set e.g. VIDEO_SOURCE = "surya_video.mp4"
VIDEO_SOURCE = 0

STEPS = [
    (1, "Pranamasana", "Prayer Pose"),
    (2, "Hasta Uttanasana", "Raised Arms / Backbend"),
    (3, "Padahastasana", "Standing Forward Bend"),
    (4, "Ashwa Sanchalanasana", "Equestrian Pose (Lead Leg 1)"),
    (5, "Utthita Chaturanga Dandasana", "Plank Pose"),
    (6, "Ashtanga Namaskara", "Eight-Limbed Salute"),
    (7, "Bhujangasana", "Cobra Pose"),
    (8, "Parvatasana", "Mountain / Downward Dog"),
    (9, "Ashwa Sanchalanasana", "Equestrian Pose (Opposite Leg)"),
    (10, "Padahastasana", "Standing Forward Bend"),
    (11, "Hasta Uttanasana", "Raised Arms / Backbend"),
    (12, "Pranamasana", "Prayer Pose"),
]

# Temporal smoothing & debouncing
# Yoga poses have deliberate, smooth transitions -> require higher debounce frames
SMOOTHING_WINDOW = 6
DEBOUNCE_FRAMES = 8

# Visibility thresholds
VISIBILITY_THRESHOLD = 0.5
DRAW_VISIBILITY_THRESHOLD = 0.2

# CSV logging configuration
# -----------------------------------------------------------------------------
# ML DATASET ARCHITECTURE NOTE:
# - INPUT FEATURES (for future ML models):
#     timestamp, torso_angle, avg_knee_angle, avg_hip_angle, avg_elbow_angle,
#     avg_shoulder_angle, shoulder_symmetry, hip_symmetry, lead_leg, visibility
# - TARGET / GROUND TRUTH LABEL:
#     step_index (1-12), step_name
# - FORM QUALITY LABELS:
#     form_status (GOOD, TRANSITION), feedback
# - DERIVED SESSION OUTPUT:
#     cycle_count
# -----------------------------------------------------------------------------
CSV_PATH = "surya_namaskar_dataset.csv"
FLUSH_EVERY = 30
CSV_COLUMNS = [
    "timestamp",
    "step_index",
    "step_name",
    "cycle_count",
    "detected_pose",
    "torso_angle",
    "avg_knee_angle",
    "avg_hip_angle",
    "avg_elbow_angle",
    "avg_shoulder_angle",
    "shoulder_symmetry",
    "hip_symmetry",
    "lead_leg",
    "visibility",
    "form_status",
    "feedback",
]


# =============================================================================
# 4. Geometry Helpers
# =============================================================================

def _xy(lm):
    """Convert landmark to (x, y) numpy array."""
    return np.array([lm.x, lm.y], dtype=np.float32)


def _midpoint(a, b):
    """Midpoint (x, y) of two landmarks."""
    return (_xy(a) + _xy(b)) / 2.0


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees [0, 180]."""
    a, b, c = _xy(a), _xy(b), _xy(c)
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0

    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


def horizontal_angle_of_vector(vec):
    """Angle (degrees) of a 2D vector relative to horizontal [0, 180]."""
    horizontal_vec = np.array([1.0, 0.0], dtype=np.float32)
    denom = np.linalg.norm(vec) * np.linalg.norm(horizontal_vec)
    if denom < 1e-8:
        return 0.0
    cos_angle = np.dot(vec, horizontal_vec) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vertical_angle_of_vector(vec):
    """Angle (degrees) of a 2D vector relative to vertical (0 = upright)."""
    vertical_vec = np.array([0.0, -1.0], dtype=np.float32)
    denom = np.linalg.norm(vec) * np.linalg.norm(vertical_vec)
    if denom < 1e-8:
        return 0.0
    cos_angle = np.dot(vec, vertical_vec) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_distance(a, b):
    """Euclidean distance between two landmarks."""
    return float(np.linalg.norm(_xy(a) - _xy(b)))


def calculate_torso_angle(shoulder_l, shoulder_r, hip_l, hip_r):
    """
    Angle of the torso relative to vertical (0 = perfectly upright).
    Uses shoulder/hip midpoints to remain robust against torso rotation.
    """
    shoulder_mid = _midpoint(shoulder_l, shoulder_r)
    hip_mid = _midpoint(hip_l, hip_r)
    return vertical_angle_of_vector(shoulder_mid - hip_mid)


# =============================================================================
# 5. Drawing Helpers (Pose Skeleton)
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
# 6. Temporal Smoothing Filter
# =============================================================================

class MovingAverage:
    """Sliding-window moving average filter to denoise per-frame measurements."""

    def __init__(self, window_size):
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)

    @property
    def ready(self):
        return len(self.values) > 0


# =============================================================================
# 7. Pose Classifiers & Biomechanical Rule Evaluators
# =============================================================================

def check_pranamasana(features):
    """Step 1 & 12: Pranamasana (Prayer Pose)"""
    torso_upright = features["torso_angle"] <= 18.0
    knees_straight = features["avg_knee_angle"] >= 155.0
    elbows_bent = 25.0 <= features["avg_elbow_angle"] <= 95.0
    hands_chest_level = abs(features["left_wrist_y"] - features["left_shoulder_y"]) < 0.30
    symmetric = features["shoulder_symmetry"] < 0.12 and features["hip_symmetry"] < 0.12

    score = 0
    if torso_upright: score += 1
    if knees_straight: score += 1
    if elbows_bent: score += 1
    if hands_chest_level: score += 1
    if symmetric: score += 1

    is_match = (score >= 4)
    feedback = "Good prayer alignment" if is_match else "Stand tall, palms together at chest"
    return is_match, feedback


def check_hasta_uttanasana(features):
    """Step 2 & 11: Hasta Uttanasana (Raised Arms / Backbend)"""
    arms_overhead = features["avg_shoulder_angle"] >= 135.0 and features["avg_elbow_angle"] >= 135.0
    wrists_above_shoulders = features["left_wrist_y"] < features["left_shoulder_y"] and features["right_wrist_y"] < features["right_shoulder_y"]
    knees_straight = features["avg_knee_angle"] >= 150.0
    backbend = features["torso_angle"] >= 6.0

    score = 0
    if arms_overhead: score += 1
    if wrists_above_shoulders: score += 1
    if knees_straight: score += 1
    if backbend: score += 1

    is_match = (score >= 3 and wrists_above_shoulders)
    feedback = "Good backbend & arm extension" if is_match else "Reach arms overhead and arch back gently"
    return is_match, feedback


def check_padahastasana(features):
    """Step 3 & 10: Padahastasana (Standing Forward Bend)"""
    torso_folded = features["torso_angle"] >= 50.0
    knees_mostly_straight = features["avg_knee_angle"] >= 130.0
    hands_reaching_down = features["left_wrist_y"] > features["hip_mid_y"]
    head_dropped = features["nose_y"] > features["shoulder_mid_y"] - 0.05

    score = 0
    if torso_folded: score += 1
    if knees_mostly_straight: score += 1
    if hands_reaching_down: score += 1
    if head_dropped: score += 1

    is_match = (score >= 3 and torso_folded)
    feedback = "Good forward fold" if is_match else "Fold forward from hips toward feet"
    return is_match, feedback


def check_ashwa_sanchalanasana(features, expected_lead_leg=None):
    """Step 4 & 9: Ashwa Sanchalanasana (Equestrian Pose / Low Lunge)"""
    left_knee = features["left_knee_angle"]
    right_knee = features["right_knee_angle"]
    knee_gap = abs(left_knee - right_knee)

    # Expected leg asymmetry: one knee bent (~75°-110°), other extended (~145°-180°)
    has_asymmetry = knee_gap >= 30.0

    if left_knee < right_knee:
        actual_lead = "LEFT"
        front_knee = left_knee
        back_knee = right_knee
    else:
        actual_lead = "RIGHT"
        front_knee = right_knee
        back_knee = left_knee

    front_knee_valid = 65.0 <= front_knee <= 120.0
    back_knee_valid = back_knee >= 135.0
    torso_low = features["shoulder_mid_y"] > 0.30

    is_pose_match = has_asymmetry and front_knee_valid and back_knee_valid and torso_low

    if not is_pose_match:
        return False, actual_lead, "Step one leg back into low lunge"

    # For Step 9, verify opposite leg was used
    if expected_lead_leg is not None and actual_lead != expected_lead_leg:
        return False, actual_lead, f"Step opposite leg forward ({expected_lead_leg} lead required)"

    return True, actual_lead, f"Good low lunge ({actual_lead} leg lead)"


def check_plank(features):
    """Step 5: Utthita Chaturanga Dandasana (Plank Pose)"""
    torso_horizontal = features["torso_angle"] >= 60.0
    hips_straight = features["avg_hip_angle"] >= 145.0
    knees_straight = features["avg_knee_angle"] >= 150.0
    arms_supporting = features["avg_elbow_angle"] >= 140.0
    symmetric = features["shoulder_symmetry"] < 0.12

    score = 0
    if torso_horizontal: score += 1
    if hips_straight: score += 1
    if knees_straight: score += 1
    if arms_supporting: score += 1
    if symmetric: score += 1

    is_match = (score >= 4 and torso_horizontal and arms_supporting)
    feedback = "Solid straight plank line" if is_match else "Hold straight plank - keep hips level"
    return is_match, feedback


def check_ashtanga_namaskara(features):
    """Step 6: Ashtanga Namaskara (Eight-Limbed Salute)"""
    knees_bent = 70.0 <= features["avg_knee_angle"] <= 150.0
    elbows_bent = 20.0 <= features["avg_elbow_angle"] <= 95.0
    chest_low = features["shoulder_mid_y"] >= features["hip_mid_y"] - 0.08
    torso_horizontal = features["torso_angle"] >= 60.0

    score = 0
    if knees_bent: score += 1
    if elbows_bent: score += 1
    if chest_low: score += 1
    if torso_horizontal: score += 1

    is_match = (score >= 3 and elbows_bent)
    feedback = "Good eight-limbed salute" if is_match else "Lower knees, chest and chin to floor"
    return is_match, feedback


def check_bhujangasana(features):
    """Step 7: Bhujangasana (Cobra Pose)"""
    chest_elevated = features["shoulder_mid_y"] < features["hip_mid_y"]
    knees_flat = features["avg_knee_angle"] >= 150.0
    arms_supporting = 75.0 <= features["avg_elbow_angle"] <= 175.0
    torso_elevation = 15.0 <= features["torso_angle"] <= 65.0

    score = 0
    if chest_elevated: score += 1
    if knees_flat: score += 1
    if arms_supporting: score += 1
    if torso_elevation: score += 1

    is_match = (score >= 3 and chest_elevated and knees_flat)
    feedback = "Good chest lift & arch" if is_match else "Slide forward, lift chest into cobra"
    return is_match, feedback


def check_parvatasana(features):
    """Step 8: Parvatasana (Mountain / Downward-Facing Dog)"""
    hips_highest = (features["hip_mid_y"] < features["shoulder_mid_y"] - 0.03 and
                    features["hip_mid_y"] < features["ankle_mid_y"] - 0.06)
    acute_hip_fold = 50.0 <= features["avg_hip_angle"] <= 120.0
    knees_straight = features["avg_knee_angle"] >= 135.0
    arms_straight = features["avg_elbow_angle"] >= 135.0

    score = 0
    if hips_highest: score += 2
    if acute_hip_fold: score += 1
    if knees_straight: score += 1
    if arms_straight: score += 1

    is_match = (score >= 4 and hips_highest)
    feedback = "Good inverted V shape" if is_match else "Press hips up and back into mountain pose"
    return is_match, feedback


# =============================================================================
# 8. Sequence State Machine & Surya Namaskar Flow Tracker
# =============================================================================

class SuryaNamaskarTracker:
    """
    Tracks the 12-step Surya Namaskar sequence with temporal debouncing
    and explicit left/right leg distinction for Steps 4 and 9.
    """

    def __init__(self, debounce_frames=DEBOUNCE_FRAMES):
        self.step_idx = 1
        self.cycle_count = 0
        self.debounce_frames = debounce_frames
        self.lead_leg_step4 = None
        self._candidate_step = None
        self._candidate_count = 0
        self.current_pose_name = "Pranamasana"
        self.form_status = "HOLD"
        self.feedback = "Begin in Pranamasana (Prayer Pose)"
        self.step_hold_frames = 0

    def get_expected_step_name(self):
        return STEPS[self.step_idx - 1][1]

    def get_expected_description(self):
        return STEPS[self.step_idx - 1][2]

    def evaluate_frame(self, features):
        """
        Evaluates the current frame against the expected sequence step.
        Advances the sequence upon sustained confirmation.
        """
        expected_step = self.step_idx
        match = False
        feedback = ""
        lead_leg = "NONE"

        if expected_step in (1, 12):
            match, feedback = check_pranamasana(features)
        elif expected_step in (2, 11):
            match, feedback = check_hasta_uttanasana(features)
        elif expected_step in (3, 10):
            match, feedback = check_padahastasana(features)
        elif expected_step == 4:
            match, lead_leg, feedback = check_ashwa_sanchalanasana(features, expected_lead_leg=None)
            if match:
                self.lead_leg_step4 = lead_leg
        elif expected_step == 5:
            match, feedback = check_plank(features)
        elif expected_step == 6:
            match, feedback = check_ashtanga_namaskara(features)
        elif expected_step == 7:
            match, feedback = check_bhujangasana(features)
        elif expected_step == 8:
            match, feedback = check_parvatasana(features)
        elif expected_step == 9:
            # Requires opposite leg of step 4
            expected_opposite = "RIGHT" if self.lead_leg_step4 == "LEFT" else "LEFT"
            match, lead_leg, feedback = check_ashwa_sanchalanasana(features, expected_lead_leg=expected_opposite)

        self.feedback = feedback

        if match:
            self.step_hold_frames += 1
            self.form_status = "GOOD"
            if self._candidate_step == expected_step:
                self._candidate_count += 1
            else:
                self._candidate_step = expected_step
                self._candidate_count = 1

            # Advance to next step after confirmation hold
            if self._candidate_count >= self.debounce_frames:
                self.advance_step()
        else:
            self.form_status = "TRANSITION"
            self._candidate_count = max(0, self._candidate_count - 1)

        return match, self.form_status, self.feedback

    def advance_step(self):
        """Advances to next step in the 12-step sequence."""
        print(f"[flow] Step {self.step_idx}/12: {self.get_expected_step_name()} CONFIRMED!")
        self._candidate_step = None
        self._candidate_count = 0
        self.step_hold_frames = 0

        if self.step_idx == 12:
            self.cycle_count += 1
            self.step_idx = 1
            self.lead_leg_step4 = None
            print(f"[cycle] Surya Namaskar Cycle #{self.cycle_count} COMPLETED!")
        else:
            self.step_idx += 1

        self.current_pose_name = self.get_expected_step_name()

    def reset_candidate(self):
        self._candidate_step = None
        self._candidate_count = 0


# =============================================================================
# 9. CSV Logging Helper
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
        index=False,
    )
    rows.clear()


# =============================================================================
# 10. HUD Rendering (Translucent Card Overlay)
# =============================================================================

def draw_hud(
    frame, step_idx, step_name, cycle_count,
    hold_frames, form_status, feedback,
    smoothed_knee, smoothed_elbow,
    smoothed_visibility, pose_detected
):
    """
    Renders a modern translucent dark card HUD in the top-left corner:
    ┌──────────────────────────────┐
    │ STEP: 4/12 (CYCLE: 1)        │
    │ POSE: Ashwa Sanchalanasana   │
    │                              │
    │ AVG KNEE: 88.4°              │
    │ AVG ELBOW: 162.1°            │
    │ FORM: GOOD (hold 8/8)        │
    │ VISIBILITY: 0.91             │
    └──────────────────────────────┘
    """
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (15, 15),
        (420, 255),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Step progress & Cycle count
    step_str = f"STEP: {step_idx}/12 (CYCLE: {cycle_count})"
    cv2.putText(
        frame,
        step_str,
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Current target pose
    cv2.putText(
        frame,
        f"POSE: {step_name}",
        (28, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Averaged angles
    knee_str = f"AVG KNEE: {smoothed_knee:.1f}°" if smoothed_knee is not None else "AVG KNEE: --"
    cv2.putText(
        frame,
        knee_str,
        (28, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    elbow_str = f"AVG ELBOW: {smoothed_elbow:.1f}°" if smoothed_elbow is not None else "AVG ELBOW: --"
    cv2.putText(
        frame,
        elbow_str,
        (28, 146),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    # Form status & Feedback
    status_color = (0, 255, 0) if form_status == "GOOD" else (0, 165, 255)
    cv2.putText(
        frame,
        f"FORM: {form_status} ({feedback})",
        (28, 176),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # Visibility score with color warning for low confidence
    if smoothed_visibility is not None:
        vis_str = f"VISIBILITY: {smoothed_visibility:.2f}"
        vis_color = (240, 240, 240) if smoothed_visibility >= VISIBILITY_THRESHOLD else (0, 165, 255)
    else:
        vis_str = "VISIBILITY: --"
        vis_color = (0, 165, 255)

    cv2.putText(
        frame,
        vis_str,
        (28, 208),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        vis_color,
        2,
        cv2.LINE_AA,
    )

    # Warning / Hint below the card
    if not pose_detected:
        cv2.putText(
            frame,
            "No person detected",
            (20, 282),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "[q]=quit",
        (20, 309),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


# =============================================================================
# 11. Main Execution Loop
# =============================================================================

def main():
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    knee_smoother = MovingAverage(SMOOTHING_WINDOW)
    elbow_smoother = MovingAverage(SMOOTHING_WINDOW)
    visibility_smoother = MovingAverage(SMOOTHING_WINDOW)
    tracker = SuryaNamaskarTracker(debounce_frames=DEBOUNCE_FRAMES)

    session_start = time.time()

    print("Surya Namaskar 12-step sequence tracker running. Press [q] to quit.")
    print(f"[flow] Starting at Step 1: {tracker.get_expected_step_name()}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB for MediaPipe Tasks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        smoothed_knee = None
        smoothed_elbow = None
        smoothed_visibility = None
        pose_detected = False

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            pose_detected = True

            # Draw skeleton connections & landmarks
            draw_pose_skeleton(frame, pose)

            raw_visibility = min(pose[i].visibility for i in REQUIRED_LANDMARKS)
            smoothed_visibility = visibility_smoother.update(raw_visibility)

            if smoothed_visibility >= VISIBILITY_THRESHOLD:
                # Extract bilateral geometric features
                left_knee = calculate_angle(pose[LEFT_HIP], pose[LEFT_KNEE], pose[LEFT_ANKLE])
                right_knee = calculate_angle(pose[RIGHT_HIP], pose[RIGHT_KNEE], pose[RIGHT_ANKLE])
                avg_knee = (left_knee + right_knee) / 2.0

                left_hip = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_KNEE])
                right_hip = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_KNEE])
                avg_hip = (left_hip + right_hip) / 2.0

                left_elbow = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST])
                right_elbow = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST])
                avg_elbow = (left_elbow + right_elbow) / 2.0

                left_shoulder = calculate_angle(pose[LEFT_ELBOW], pose[LEFT_SHOULDER], pose[LEFT_HIP])
                right_shoulder = calculate_angle(pose[RIGHT_ELBOW], pose[RIGHT_SHOULDER], pose[RIGHT_HIP])
                avg_shoulder = (left_shoulder + right_shoulder) / 2.0

                torso_angle = calculate_torso_angle(
                    pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER],
                    pose[LEFT_HIP], pose[RIGHT_HIP],
                )

                shoulder_sym = float(abs(pose[LEFT_SHOULDER].y - pose[RIGHT_SHOULDER].y))
                hip_sym = float(abs(pose[LEFT_HIP].y - pose[RIGHT_HIP].y))

                mid_shoulder = _midpoint(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
                mid_hip = _midpoint(pose[LEFT_HIP], pose[RIGHT_HIP])
                mid_ankle = _midpoint(pose[LEFT_ANKLE], pose[RIGHT_ANKLE])

                # Sliding-window moving average smoothing
                smoothed_knee = knee_smoother.update(avg_knee)
                smoothed_elbow = elbow_smoother.update(avg_elbow)

                features = {
                    "torso_angle": torso_angle,
                    "left_knee_angle": left_knee,
                    "right_knee_angle": right_knee,
                    "avg_knee_angle": smoothed_knee,
                    "left_hip_angle": left_hip,
                    "right_hip_angle": right_hip,
                    "avg_hip_angle": avg_hip,
                    "left_elbow_angle": left_elbow,
                    "right_elbow_angle": right_elbow,
                    "avg_elbow_angle": smoothed_elbow,
                    "left_shoulder_angle": left_shoulder,
                    "right_shoulder_angle": right_shoulder,
                    "avg_shoulder_angle": avg_shoulder,
                    "shoulder_symmetry": shoulder_sym,
                    "hip_symmetry": hip_sym,
                    "shoulder_mid_y": float(mid_shoulder[1]),
                    "hip_mid_y": float(mid_hip[1]),
                    "ankle_mid_y": float(mid_ankle[1]),
                    "left_wrist_y": float(pose[LEFT_WRIST].y),
                    "right_wrist_y": float(pose[RIGHT_WRIST].y),
                    "left_shoulder_y": float(pose[LEFT_SHOULDER].y),
                    "right_shoulder_y": float(pose[RIGHT_SHOULDER].y),
                    "nose_y": float(pose[NOSE].y),
                }

                # Evaluate current frame in sequence tracker
                tracker.evaluate_frame(features)

                row = {
                    "timestamp": timestamp_ms,
                    "step_index": tracker.step_idx,
                    "step_name": tracker.get_expected_step_name(),
                    "cycle_count": tracker.cycle_count,
                    "detected_pose": tracker.current_pose_name,
                    "torso_angle": round(torso_angle, 1),
                    "avg_knee_angle": round(smoothed_knee, 1),
                    "avg_hip_angle": round(avg_hip, 1),
                    "avg_elbow_angle": round(smoothed_elbow, 1),
                    "avg_shoulder_angle": round(avg_shoulder, 1),
                    "shoulder_symmetry": round(shoulder_sym, 4),
                    "hip_symmetry": round(hip_sym, 4),
                    "lead_leg": tracker.lead_leg_step4 or "NONE",
                    "visibility": round(smoothed_visibility, 3),
                    "form_status": tracker.form_status,
                    "feedback": tracker.feedback,
                }
            else:
                # Low visibility frame: clear pending candidate transitions
                tracker.reset_candidate()
                row = {
                    "timestamp": timestamp_ms,
                    "step_index": tracker.step_idx,
                    "step_name": tracker.get_expected_step_name(),
                    "cycle_count": tracker.cycle_count,
                    "detected_pose": "LOW CONFIDENCE",
                    "torso_angle": None,
                    "avg_knee_angle": None,
                    "avg_hip_angle": None,
                    "avg_elbow_angle": None,
                    "avg_shoulder_angle": None,
                    "shoulder_symmetry": None,
                    "hip_symmetry": None,
                    "lead_leg": tracker.lead_leg_step4 or "NONE",
                    "visibility": round(smoothed_visibility, 3),
                    "form_status": "LOW CONFIDENCE",
                    "feedback": "Adjust camera angle",
                }
        else:
            # No person detected in frame
            smoothed_visibility = visibility_smoother.update(0.0)
            tracker.reset_candidate()
            row = {
                "timestamp": timestamp_ms,
                "step_index": tracker.step_idx,
                "step_name": tracker.get_expected_step_name(),
                "cycle_count": tracker.cycle_count,
                "detected_pose": "NO PERSON",
                "torso_angle": None,
                "avg_knee_angle": None,
                "avg_hip_angle": None,
                "avg_elbow_angle": None,
                "avg_shoulder_angle": None,
                "shoulder_symmetry": None,
                "hip_symmetry": None,
                "lead_leg": tracker.lead_leg_step4 or "NONE",
                "visibility": round(smoothed_visibility, 3),
                "form_status": "NO PERSON",
                "feedback": "Step into camera frame",
            }

        buffer.append(row)
        if len(buffer) >= FLUSH_EVERY:
            flush_buffer(buffer, CSV_PATH)

        # Render translucent card HUD overlay
        draw_hud(
            frame,
            step_idx=tracker.step_idx,
            step_name=tracker.get_expected_step_name(),
            cycle_count=tracker.cycle_count,
            hold_frames=tracker._candidate_count,
            form_status=tracker.form_status,
            feedback=tracker.feedback,
            smoothed_knee=smoothed_knee,
            smoothed_elbow=smoothed_elbow,
            smoothed_visibility=smoothed_visibility,
            pose_detected=pose_detected,
        )

        cv2.imshow("Surya Namaskar Sequence Tracker", frame)
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
    print(f"Full 12-step cycles completed: {tracker.cycle_count}")
    print(f"Final step reached: Step {tracker.step_idx} ({tracker.get_expected_step_name()})")
    print(f"Saved to: {CSV_PATH}")
    print("----------------------------")


if __name__ == "__main__":
    main()
