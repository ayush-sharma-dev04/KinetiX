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

REQUIRED_LANDMARKS = [
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

# Live webcam (0). To use a recorded video file, set e.g. VIDEO_SOURCE = "pushup_video.mp4"
VIDEO_SOURCE = 0

# State-detection angle thresholds (in degrees)
# Average elbow angle drives state classification:
# Arms straight ~160-180 deg (UP), Deep descent <= 95 deg (BOTTOM)
ELBOW_UP_THRESHOLD = 160.0       # >= this => plank / arms extended (UP)
ELBOW_BOTTOM_THRESHOLD = 95.0    # <= this => deep descent / 90° elbow (BOTTOM)

# Temporal smoothing & debouncing
SMOOTHING_WINDOW = 5             # Frames for sliding-window moving-average filter
DEBOUNCE_FRAMES = 4              # Consecutive frames a state must persist to confirm transition

# Form check thresholds
ELBOW_ASYMMETRY_THRESHOLD = 18.0   # Degree gap between left & right elbow at depth
HIP_SAG_THRESHOLD = 150.0          # Below this indicates hip sag / weak core
HIP_PIKE_THRESHOLD = 185.0         # Above this indicates hips piking up
SHOULDER_TILT_THRESHOLD = 15.0     # Degrees of shoulder tilt indicating body twist
NECK_DEVIATION_THRESHOLD = 35.0    # Deviation from neutral neck indicating head craning

# Visibility thresholds
VISIBILITY_THRESHOLD = 0.5       # Smoothed visibility required to trust pose & drive state machine
DRAW_VISIBILITY_THRESHOLD = 0.2  # Landmarks/connections below this are not drawn (noise filter)

# CSV logging configuration
# -----------------------------------------------------------------------------
# ML DATASET ARCHITECTURE NOTE:
# - INPUT FEATURES (for future ML models):
#     timestamp, left_elbow_angle, right_elbow_angle, avg_elbow_angle,
#     smoothed_elbow_angle, left_hip_angle, right_hip_angle, avg_hip_angle,
#     smoothed_hip_angle, shoulder_tilt_angle, neck_angle, visibility
# - TARGET / GROUND TRUTH LABEL:
#     state (UP, DOWN, BOTTOM)
# - FORM QUALITY LABELS:
#     form_status (GOOD, FORM ISSUE), feedback
# - DERIVED SESSION OUTPUT:
#     rep_count
# -----------------------------------------------------------------------------
CSV_PATH = "pushup_dataset.csv"
FLUSH_EVERY = 30
CSV_COLUMNS = [
    "timestamp",
    "left_elbow_angle",
    "right_elbow_angle",
    "avg_elbow_angle",
    "smoothed_elbow_angle",
    "left_hip_angle",
    "right_hip_angle",
    "avg_hip_angle",
    "smoothed_hip_angle",
    "shoulder_tilt_angle",
    "neck_angle",
    "visibility",
    "state",
    "rep_count",
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
# 7. Direction-Aware State Classification & Debouncing
# =============================================================================

def classify_state(avg_elbow_angle, current_state):
    """
    Direction-aware state classifier based on smoothed average elbow angle.

    Enforces valid biological progression and prevents impossible direct jumps:
      - From UP: Can only transition to DOWN (UP -> DOWN -> BOTTOM)
      - From BOTTOM: Can only transition to DOWN (BOTTOM -> DOWN -> UP)
      - From DOWN: Can transition to BOTTOM (descending) or UP (ascending)
    """
    if current_state == "UP":
        if avg_elbow_angle >= ELBOW_UP_THRESHOLD:
            return "UP"
        else:
            return "DOWN"

    elif current_state == "BOTTOM":
        if avg_elbow_angle <= ELBOW_BOTTOM_THRESHOLD:
            return "BOTTOM"
        else:
            return "DOWN"

    elif current_state == "DOWN":
        if avg_elbow_angle <= ELBOW_BOTTOM_THRESHOLD:
            return "BOTTOM"
        elif avg_elbow_angle >= ELBOW_UP_THRESHOLD:
            return "UP"
        else:
            return "DOWN"

    return "UP"


class DebouncedState:
    """
    Filters out transient noise / single-frame spikes. A candidate state must
    persist for `debounce_frames` consecutive valid frames to be confirmed.
    """

    def __init__(self, debounce_frames, initial_state="UP"):
        self.debounce_frames = debounce_frames
        self.confirmed_state = initial_state
        self._candidate_state = None
        self._candidate_count = 0

    def update(self, raw_state):
        """Returns (confirmed_state, changed) tuple for the current frame."""
        if raw_state == self.confirmed_state:
            self._candidate_state = None
            self._candidate_count = 0
            return self.confirmed_state, False

        if raw_state == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = raw_state
            self._candidate_count = 1

        if self._candidate_count >= self.debounce_frames:
            self.confirmed_state = raw_state
            self._candidate_state = None
            self._candidate_count = 0
            return self.confirmed_state, True

        return self.confirmed_state, False

    def reset_candidate(self):
        """Clears pending candidate state when tracking is lost or invalid."""
        self._candidate_state = None
        self._candidate_count = 0


# =============================================================================
# 8. Rep Counting State Machine & Form Evaluator
# =============================================================================

class PushUpRepCounter:
    """
    Tracks confirmed state transitions and increments rep count only upon
    completion of a full valid cycle:
      UP -> DOWN -> BOTTOM -> DOWN -> UP

    A shallow push-up (UP -> DOWN -> UP) is registered as incomplete and not counted.
    Direct illegal transitions (e.g. UP -> BOTTOM or BOTTOM -> UP) are rejected.
    """

    ALLOWED_TRANSITIONS = {
        "UP": {"DOWN"},
        "DOWN": {"BOTTOM", "UP"},
        "BOTTOM": {"DOWN"},
    }

    def __init__(self):
        self.state = "UP"
        self.rep_count = 0
        self.shallow_count = 0
        self._reached_bottom = False

    def process_transition(self, new_state):
        if new_state == self.state:
            return

        valid_targets = self.ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in valid_targets:
            print(f"[warning] Blocked invalid direct transition: {self.state} -> {new_state}")
            return

        print(f"[state] {self.state} -> {new_state}")

        if new_state == "BOTTOM":
            self._reached_bottom = True

        if new_state == "UP" and self.state == "DOWN":
            if self._reached_bottom:
                self.rep_count += 1
                print(f"[rep] Rep #{self.rep_count} counted (reached BOTTOM)")
            else:
                self.shallow_count += 1
                print("[rep] Shallow push-up - NOT counted (never reached BOTTOM)")
            self._reached_bottom = False

        self.state = new_state


def evaluate_pushup_form(
    left_elbow, right_elbow, left_hip, right_hip, shoulder_tilt, neck_angle, current_state
):
    """
    Evaluates push-up biomechanics and form quality independently of rep counter.
    Returns (form_status, feedback_message) tuple.
    """
    feedback = "Good Form"
    is_good = True

    # 1. Check elbow asymmetry (uneven push depth)
    elbow_gap = abs(left_elbow - right_elbow)
    if elbow_gap > ELBOW_ASYMMETRY_THRESHOLD:
        is_good = False
        feedback = "Uneven push - balance both arms"

    # 2. Check plank straightness / hip sag or pike
    avg_hip = (left_hip + right_hip) / 2.0
    if avg_hip < HIP_SAG_THRESHOLD:
        is_good = False
        feedback = "Hips sagging - tighten core"
    elif avg_hip > HIP_PIKE_THRESHOLD:
        is_good = False
        feedback = "Hips piking - lower hips into line"

    # 3. Check shoulder tilt / body twist
    elif shoulder_tilt > SHOULDER_TILT_THRESHOLD:
        is_good = False
        feedback = "Torso twisting - keep shoulders level"

    # 4. Check neck alignment
    elif abs(neck_angle - 90.0) > NECK_DEVIATION_THRESHOLD:
        is_good = False
        feedback = "Keep neck neutral - don't crane head"

    status = "GOOD" if is_good else "FORM ISSUE"
    return status, feedback


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

# State color mappings (BGR format)
STATE_COLORS = {
    "UP": (0, 255, 0),       # Green
    "DOWN": (0, 215, 255),   # Amber / Yellow
    "BOTTOM": (255, 255, 0), # Cyan / Sky Blue
}


def draw_hud(
    frame, state, rep_count, smoothed_elbow, smoothed_hip,
    form_status, feedback, smoothed_visibility, pose_detected
):
    """
    Renders a modern translucent dark card HUD in the top-left corner:
    ┌──────────────────────────────┐
    │ STATE: DOWN                  │
    │ REPS:  3                     │
    │                              │
    │ AVG ELBOW: 88.4°             │
    │ PLANK HIP: 172.1°            │
    │ FORM: GOOD                   │
    │ VISIBILITY: 0.91             │
    └──────────────────────────────┘
    """
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (15, 15),
        (370, 240),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # State text & dynamic color
    state_color = STATE_COLORS.get(state, (0, 255, 0))
    cv2.putText(
        frame,
        f"STATE: {state}",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        state_color,
        2,
        cv2.LINE_AA,
    )

    # Rep counter
    cv2.putText(
        frame,
        f"REPS:  {rep_count}",
        (28, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Averaged angles
    elbow_str = f"AVG ELBOW: {smoothed_elbow:.1f}°" if smoothed_elbow is not None else "AVG ELBOW: --"
    cv2.putText(
        frame,
        elbow_str,
        (28, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    hip_str = f"PLANK HIP: {smoothed_hip:.1f}°" if smoothed_hip is not None else "PLANK HIP: --"
    cv2.putText(
        frame,
        hip_str,
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
        f"FORM: {form_status} - {feedback}",
        (28, 176),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
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
            (20, 268),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "[q]=quit",
        (20, 295),
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

    elbow_smoother = MovingAverage(SMOOTHING_WINDOW)
    hip_smoother = MovingAverage(SMOOTHING_WINDOW)
    visibility_smoother = MovingAverage(SMOOTHING_WINDOW)
    debounced_state = DebouncedState(DEBOUNCE_FRAMES, initial_state="UP")
    rep_counter = PushUpRepCounter()

    session_start = time.time()

    print("Automatic push-up state detection running. Press [q] to quit.")
    print(f"[state] Starting state: {rep_counter.state}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB for MediaPipe Tasks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        smoothed_elbow = None
        smoothed_hip = None
        smoothed_visibility = None
        form_status = "GOOD"
        feedback = "Good Form"
        pose_detected = False

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            pose_detected = True

            # Draw skeleton connections & landmarks
            draw_pose_skeleton(frame, pose)

            raw_visibility = min(pose[i].visibility for i in REQUIRED_LANDMARKS)
            smoothed_visibility = visibility_smoother.update(raw_visibility)

            if smoothed_visibility >= VISIBILITY_THRESHOLD:
                # 1. Joint angles
                left_elbow_angle = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST])
                right_elbow_angle = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST])

                left_hip_angle = calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_HIP], pose[LEFT_KNEE])
                right_hip_angle = calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], pose[RIGHT_KNEE])

                # 2. Symmetry / alignment features
                shoulder_tilt = horizontal_angle_of_vector(
                    _xy(pose[RIGHT_SHOULDER]) - _xy(pose[LEFT_SHOULDER])
                )
                mid_shoulder = _midpoint(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
                nose_xy = _xy(pose[NOSE])
                neck_angle = vertical_angle_of_vector(nose_xy - mid_shoulder)

                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0
                avg_hip_angle = (left_hip_angle + right_hip_angle) / 2.0

                # 3. Sliding-window moving average smoothing
                smoothed_elbow = elbow_smoother.update(avg_elbow_angle)
                smoothed_hip = hip_smoother.update(avg_hip_angle)

                # 4. Direction-aware state classification and debouncing
                raw_state = classify_state(smoothed_elbow, debounced_state.confirmed_state)
                confirmed_state, changed = debounced_state.update(raw_state)

                if changed:
                    rep_counter.process_transition(confirmed_state)

                # 5. Form evaluation
                form_status, feedback = evaluate_pushup_form(
                    left_elbow_angle, right_elbow_angle,
                    left_hip_angle, right_hip_angle,
                    shoulder_tilt, neck_angle,
                    rep_counter.state,
                )

                row = {
                    "timestamp": timestamp_ms,
                    "left_elbow_angle": round(left_elbow_angle, 1),
                    "right_elbow_angle": round(right_elbow_angle, 1),
                    "avg_elbow_angle": round(avg_elbow_angle, 1),
                    "smoothed_elbow_angle": round(smoothed_elbow, 1),
                    "left_hip_angle": round(left_hip_angle, 1),
                    "right_hip_angle": round(right_hip_angle, 1),
                    "avg_hip_angle": round(avg_hip_angle, 1),
                    "smoothed_hip_angle": round(smoothed_hip, 1),
                    "shoulder_tilt_angle": round(shoulder_tilt, 1),
                    "neck_angle": round(neck_angle, 1),
                    "visibility": round(smoothed_visibility, 3),
                    "state": rep_counter.state,
                    "rep_count": rep_counter.rep_count,
                    "form_status": form_status,
                    "feedback": feedback,
                }
            else:
                # Low visibility frame: clear pending candidate transitions
                debounced_state.reset_candidate()
                form_status = "LOW CONFIDENCE"
                feedback = "Adjust camera view"
                row = {
                    "timestamp": timestamp_ms,
                    "left_elbow_angle": None,
                    "right_elbow_angle": None,
                    "avg_elbow_angle": None,
                    "smoothed_elbow_angle": None,
                    "left_hip_angle": None,
                    "right_hip_angle": None,
                    "avg_hip_angle": None,
                    "smoothed_hip_angle": None,
                    "shoulder_tilt_angle": None,
                    "neck_angle": None,
                    "visibility": round(smoothed_visibility, 3),
                    "state": rep_counter.state,
                    "rep_count": rep_counter.rep_count,
                    "form_status": form_status,
                    "feedback": feedback,
                }
        else:
            # No person detected in frame
            smoothed_visibility = visibility_smoother.update(0.0)
            debounced_state.reset_candidate()
            form_status = "NO PERSON"
            feedback = "Step into frame"
            row = {
                "timestamp": timestamp_ms,
                "left_elbow_angle": None,
                "right_elbow_angle": None,
                "avg_elbow_angle": None,
                "smoothed_elbow_angle": None,
                "left_hip_angle": None,
                "right_hip_angle": None,
                "avg_hip_angle": None,
                "smoothed_hip_angle": None,
                "shoulder_tilt_angle": None,
                "neck_angle": None,
                "visibility": round(smoothed_visibility, 3),
                "state": rep_counter.state,
                "rep_count": rep_counter.rep_count,
                "form_status": form_status,
                "feedback": feedback,
            }

        buffer.append(row)
        if len(buffer) >= FLUSH_EVERY:
            flush_buffer(buffer, CSV_PATH)

        # Render translucent card HUD overlay
        draw_hud(
            frame,
            state=rep_counter.state,
            rep_count=rep_counter.rep_count,
            smoothed_elbow=smoothed_elbow,
            smoothed_hip=smoothed_hip,
            form_status=form_status,
            feedback=feedback,
            smoothed_visibility=smoothed_visibility,
            pose_detected=pose_detected,
        )

        cv2.imshow("Push-up Rep Tracker", frame)
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
    print(f"Full reps counted: {rep_counter.rep_count}")
    print(f"Shallow push-ups (not counted): {rep_counter.shallow_count}")
    print(f"Final state: {rep_counter.state}")
    print(f"Saved to: {CSV_PATH}")
    print("----------------------------")


if __name__ == "__main__":
    main()
