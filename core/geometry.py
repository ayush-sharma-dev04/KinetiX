"""
Geometric calculation utilities for fitness and yoga pose analysis.
"""
from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np

from core.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    REQUIRED_LANDMARKS,
)


def _xy(lm: Any) -> np.ndarray:
    """Convert landmark to (x, y) numpy array."""
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return np.array([lm.x, lm.y], dtype=np.float32)
    elif isinstance(lm, (list, tuple, np.ndarray)):
        return np.array([lm[0], lm[1]], dtype=np.float32)
    raise ValueError(f"Unsupported landmark type: {type(lm)}")


def calculate_distance(a: Any, b: Any) -> float:
    """Euclidean distance between two landmarks in normalized 2D coordinates."""
    va, vb = _xy(a), _xy(b)
    return float(np.linalg.norm(va - vb))


def calculate_angle(a: Any, b: Any, c: Any) -> float:
    """
    Calculate 2D interior angle at vertex `b` formed by rays b->a and b->c.
    Returns angle in degrees within [0.0, 180.0].
    """
    pa, pb, pc = _xy(a), _xy(b), _xy(c)

    ba = pa - pb
    bc = pc - pb

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0

    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


def calculate_torso_angle(shoulder_l: Any, shoulder_r: Any, hip_l: Any, hip_r: Any) -> float:
    """
    Angle of the torso relative to vertical (0° = perfectly upright).
    Uses shoulder midpoint and hip midpoint to remain robust against torso twisting.
    """
    shoulder_mid = (_xy(shoulder_l) + _xy(shoulder_r)) / 2.0
    hip_mid = (_xy(hip_l) + _xy(hip_r)) / 2.0

    torso_vec = shoulder_mid - hip_mid
    vertical_vec = np.array([0.0, -1.0], dtype=np.float32)  # Upward in image coordinate space

    denom = np.linalg.norm(torso_vec) * np.linalg.norm(vertical_vec)
    if denom < 1e-8:
        return 0.0

    cos_angle = np.dot(torso_vec, vertical_vec) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def calculate_segment_angle_to_horizontal(p1: Any, p2: Any) -> float:
    """
    Angle of segment p1->p2 relative to horizontal axis in degrees [0, 90].
    0° = horizontal, 90° = vertical.
    """
    v1, v2 = _xy(p1), _xy(p2)
    delta = np.abs(v2 - v1)
    angle_rad = np.arctan2(delta[1], delta[0])
    return float(np.degrees(angle_rad))


@dataclass
class YogaFeatures:
    """Structured container of extracted geometric features for yoga analysis."""
    # Knee angles (180° = straight leg)
    left_knee_angle: float
    right_knee_angle: float
    avg_knee_angle: float

    # Hip angles (shoulder-hip-knee, 180° = open alignment)
    left_hip_angle: float
    right_hip_angle: float
    avg_hip_angle: float

    # Elbow angles (shoulder-elbow-wrist, 180° = straight arm)
    left_elbow_angle: float
    right_elbow_angle: float
    avg_elbow_angle: float

    # Shoulder angles (hip-shoulder-elbow, 0° = arms down, 90° = horizontal, 180° = overhead)
    left_shoulder_angle: float
    right_shoulder_angle: float
    avg_shoulder_angle: float

    # Torso deviation from upright vertical (0° = vertical)
    torso_angle: float

    # Stance and proximity ratios
    stance_width_ratio: float      # Euclidean ankle distance / shoulder width
    horizontal_stance_ratio: float # Horizontal ankle distance / shoulder width
    feet_distance_ratio: float     # Euclidean ankle distance / hip width

    # Symmetry indicators
    shoulder_level_diff: float  # |shoulder_l.y - shoulder_r.y| / shoulder_dist
    hip_level_diff: float       # |hip_l.y - hip_r.y| / hip_dist

    # Vertical landmarks coordinates (in image coords: 0=top, 1=bottom)
    shoulder_mid_y: float
    hip_mid_y: float
    knee_mid_y: float
    ankle_mid_y: float

    left_wrist_y: float
    right_wrist_y: float
    left_shoulder_y: float
    right_shoulder_y: float
    left_ankle_y: float
    right_ankle_y: float
    left_knee_y: float
    right_knee_y: float

    # Hand positions
    hands_above_head: bool

    # Confidence visibility score
    visibility: float

    # Raw landmark reference
    landmarks: Any = None


def extract_yoga_features(pose_landmarks: List[Any]) -> YogaFeatures:
    """
    Extracts comprehensive bilateral geometric features from raw pose landmarks.
    """
    # Key landmark references
    sh_l = pose_landmarks[LEFT_SHOULDER]
    sh_r = pose_landmarks[RIGHT_SHOULDER]
    el_l = pose_landmarks[LEFT_ELBOW]
    el_r = pose_landmarks[RIGHT_ELBOW]
    wr_l = pose_landmarks[LEFT_WRIST]
    wr_r = pose_landmarks[RIGHT_WRIST]
    hp_l = pose_landmarks[LEFT_HIP]
    hp_r = pose_landmarks[RIGHT_HIP]
    kn_l = pose_landmarks[LEFT_KNEE]
    kn_r = pose_landmarks[RIGHT_KNEE]
    ak_l = pose_landmarks[LEFT_ANKLE]
    ak_r = pose_landmarks[RIGHT_ANKLE]

    # Bilateral joint angles
    l_knee = calculate_angle(hp_l, kn_l, ak_l)
    r_knee = calculate_angle(hp_r, kn_r, ak_r)
    avg_knee = (l_knee + r_knee) / 2.0

    l_hip = calculate_angle(sh_l, hp_l, kn_l)
    r_hip = calculate_angle(sh_r, hp_r, kn_r)
    avg_hip = (l_hip + r_hip) / 2.0

    l_elbow = calculate_angle(sh_l, el_l, wr_l)
    r_elbow = calculate_angle(sh_r, el_r, wr_r)
    avg_elbow = (l_elbow + r_elbow) / 2.0

    l_shoulder = calculate_angle(hp_l, sh_l, el_l)
    r_shoulder = calculate_angle(hp_r, sh_r, el_r)
    avg_shoulder = (l_shoulder + r_shoulder) / 2.0

    torso_ang = calculate_torso_angle(sh_l, sh_r, hp_l, hp_r)

    # Distances & ratios
    sh_dist = max(1e-4, calculate_distance(sh_l, sh_r))
    hp_dist = max(1e-4, calculate_distance(hp_l, hp_r))
    ak_dist = calculate_distance(ak_l, ak_r)
    ak_h_dist = abs(ak_l.x - ak_r.x)

    stance_ratio = ak_dist / sh_dist
    h_stance_ratio = ak_h_dist / sh_dist
    feet_ratio = ak_dist / hp_dist

    # Symmetry differences (normalized by width)
    sh_y_l, sh_y_r = sh_l.y, sh_r.y
    hp_y_l, hp_y_r = hp_l.y, hp_r.y

    sh_diff = abs(sh_y_l - sh_y_r) / sh_dist
    hp_diff = abs(hp_y_l - hp_y_r) / hp_dist

    # Vertical midpoints
    sh_mid_y = (sh_y_l + sh_y_r) / 2.0
    hp_mid_y = (hp_y_l + hp_y_r) / 2.0
    kn_mid_y = (kn_l.y + kn_r.y) / 2.0
    ak_mid_y = (ak_l.y + ak_r.y) / 2.0

    hands_above = (wr_l.y < min(sh_y_l, sh_y_r)) and (wr_r.y < min(sh_y_l, sh_y_r))

    # Overall visibility
    vis_scores = [getattr(pose_landmarks[i], "visibility", 1.0) for i in REQUIRED_LANDMARKS]
    min_vis = min(vis_scores) if vis_scores else 1.0

    return YogaFeatures(
        left_knee_angle=l_knee,
        right_knee_angle=r_knee,
        avg_knee_angle=avg_knee,
        left_hip_angle=l_hip,
        right_hip_angle=r_hip,
        avg_hip_angle=avg_hip,
        left_elbow_angle=l_elbow,
        right_elbow_angle=r_elbow,
        avg_elbow_angle=avg_elbow,
        left_shoulder_angle=l_shoulder,
        right_shoulder_angle=r_shoulder,
        avg_shoulder_angle=avg_shoulder,
        torso_angle=torso_ang,
        stance_width_ratio=stance_ratio,
        horizontal_stance_ratio=h_stance_ratio,
        feet_distance_ratio=feet_ratio,
        shoulder_level_diff=sh_diff,
        hip_level_diff=hp_diff,
        shoulder_mid_y=sh_mid_y,
        hip_mid_y=hp_mid_y,
        knee_mid_y=kn_mid_y,
        ankle_mid_y=ak_mid_y,
        left_wrist_y=wr_l.y,
        right_wrist_y=wr_r.y,
        left_shoulder_y=sh_y_l,
        right_shoulder_y=sh_y_r,
        left_ankle_y=ak_l.y,
        right_ankle_y=ak_r.y,
        left_knee_y=kn_l.y,
        right_knee_y=kn_r.y,
        hands_above_head=hands_above,
        visibility=min_vis,
        landmarks=pose_landmarks,
    )
