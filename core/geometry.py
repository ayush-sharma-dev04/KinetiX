"""
Geometric calculation utilities for fitness and yoga pose analysis.
Supports both 2D normalized screen-relative geometry and 3D/world-landmark metrics.
"""
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple
import numpy as np

from core.landmarks import (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    REQUIRED_LANDMARKS,
)


def _xy(lm: Any) -> np.ndarray:
    """Convert landmark to (x, y) 2D numpy array."""
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return np.array([lm.x, lm.y], dtype=np.float32)
    elif isinstance(lm, (list, tuple, np.ndarray)):
        return np.array([lm[0], lm[1]], dtype=np.float32)
    raise ValueError(f"Unsupported landmark type: {type(lm)}")


def _xyz(lm: Any) -> np.ndarray:
    """Convert landmark to (x, y, z) 3D numpy array."""
    if hasattr(lm, "x") and hasattr(lm, "y"):
        z = getattr(lm, "z", 0.0)
        return np.array([lm.x, lm.y, z], dtype=np.float32)
    elif isinstance(lm, (list, tuple, np.ndarray)):
        z = lm[2] if len(lm) > 2 else 0.0
        return np.array([lm[0], lm[1], z], dtype=np.float32)
    raise ValueError(f"Unsupported landmark type: {type(lm)}")


def calculate_distance(a: Any, b: Any) -> float:
    """Euclidean distance between two landmarks in normalized 2D coordinates."""
    va, vb = _xy(a), _xy(b)
    return float(np.linalg.norm(va - vb))


def calculate_distance_3d(a: Any, b: Any) -> float:
    """Euclidean distance between two landmarks in 3D metric coordinates."""
    va, vb = _xyz(a), _xyz(b)
    return float(np.linalg.norm(va - vb))


def calculate_angle_2d(a: Any, b: Any, c: Any) -> float:
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


def calculate_angle_3d(a: Any, b: Any, c: Any) -> float:
    """
    Calculate true 3D interior angle at joint vertex `b` formed by rays b->a and b->c.
    Returns angle in degrees within [0.0, 180.0].
    """
    pa, pb, pc = _xyz(a), _xyz(b), _xyz(c)

    ba = pa - pb
    bc = pc - pb

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0

    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    return float(np.degrees(angle_rad))


def calculate_angle(a: Any, b: Any, c: Any, prefer_3d: bool = True) -> float:
    """
    Calculate interior angle at vertex `b`.
    Uses 3D geometry if z-coordinates are non-trivial or prefer_3d is True,
    otherwise computes 2D angle.
    """
    pa, pb, pc = _xyz(a), _xyz(b), _xyz(c)
    if prefer_3d and (abs(pa[2]) > 1e-6 or abs(pb[2]) > 1e-6 or abs(pc[2]) > 1e-6):
        return calculate_angle_3d(a, b, c)
    return calculate_angle_2d(a, b, c)


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


def calculate_nose_torso_offset(
    nose: Any, shoulder_l: Any, shoulder_r: Any, hip_l: Any, hip_r: Any
) -> float:
    """
    Calculates perpendicular horizontal deviation of the nose relative to the
    spinal axis (line segment from hip midpoint to shoulder midpoint).
    Normalized by torso length.
    Returns offset ratio >= 0.0 (0.0 = perfect alignment).
    """
    n = _xy(nose)
    sh_mid = (_xy(shoulder_l) + _xy(shoulder_r)) / 2.0
    hp_mid = (_xy(hip_l) + _xy(hip_r)) / 2.0

    spine_vec = sh_mid - hp_mid
    torso_len = np.linalg.norm(spine_vec)
    if torso_len < 1e-7:
        return 0.0

    u_spine = spine_vec / torso_len
    # Perpendicular normal in 2D
    u_perp = np.array([-u_spine[1], u_spine[0]], dtype=np.float32)

    # Vector from hip midpoint to nose
    nose_vec = n - hp_mid
    perp_dist = abs(np.dot(nose_vec, u_perp))

    return float(perp_dist / torso_len)


def calculate_forward_collapse(
    shoulder_l: Any, shoulder_r: Any, hip_l: Any, hip_r: Any
) -> float:
    """
    Measures 3D forward collapse / flexion angle out of the coronal plane.
    Particularly useful in Triangle pose to distinguish true lateral tilt from
    torso rounding/collapsing forward toward the camera/ground.
    Returns angle in degrees [0.0, 90.0].
    """
    sh_l, sh_r = _xyz(shoulder_l), _xyz(shoulder_r)
    hp_l, hp_r = _xyz(hip_l), _xyz(hip_r)

    sh_mid = (sh_l + sh_r) / 2.0
    hp_mid = (hp_l + hp_r) / 2.0

    torso_vec = sh_mid - hp_mid
    hip_axis = hp_r - hp_l

    norm_torso = np.linalg.norm(torso_vec)
    norm_hip = np.linalg.norm(hip_axis)

    if norm_torso < 1e-7 or norm_hip < 1e-7:
        return 0.0

    # Normal vector perpendicular to hips and vertical (sagittal normal)
    coronal_normal = np.cross(hip_axis, np.array([0.0, -1.0, 0.0], dtype=np.float32))
    norm_normal = np.linalg.norm(coronal_normal)
    if norm_normal < 1e-7:
        return 0.0

    coronal_normal = coronal_normal / norm_normal
    sin_collapse = abs(np.dot(torso_vec, coronal_normal)) / norm_torso
    sin_collapse = np.clip(sin_collapse, 0.0, 1.0)

    return float(np.degrees(np.arcsin(sin_collapse)))


def calculate_foot_orientation(heel: Any, foot_index: Any) -> float:
    """
    Calculates angle of the foot vector (heel -> foot_index) in degrees relative to
    the horizontal x-axis [-180.0, 180.0].
    """
    h, f = _xy(heel), _xy(foot_index)
    vec = f - h
    if np.linalg.norm(vec) < 1e-7:
        return 0.0
    angle_rad = np.arctan2(vec[1], vec[0])
    return float(np.degrees(angle_rad))


def calculate_knee_over_ankle_offset(
    knee: Any, ankle: Any, foot_index: Any, lead_direction: float = 1.0
) -> float:
    """
    Horizontal displacement of knee relative to ankle in the direction of the foot.
    Positive value indicates the knee has traveled forward past the ankle.
    """
    kn = _xy(knee)
    ak = _xy(ankle)
    fi = _xy(foot_index)

    foot_vec = fi - ak
    foot_len = np.linalg.norm(foot_vec)
    if foot_len < 1e-7:
        dir_sign = 1.0 if lead_direction >= 0 else -1.0
        return float((kn[0] - ak[0]) * dir_sign)

    dir_x = np.sign(foot_vec[0]) if abs(foot_vec[0]) > 1e-4 else (1.0 if lead_direction >= 0 else -1.0)
    dx = (kn[0] - ak[0]) * dir_x
    return float(dx)


@dataclass
class YogaFeatures:
    """Structured container of extracted geometric features for yoga analysis."""
    # Knee angles (hip-knee-ankle, 180° = straight leg)
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

    # Biomechanical Ankle angles (knee-ankle-foot index / heel)
    left_ankle_angle: float = 90.0
    right_ankle_angle: float = 90.0
    avg_ankle_angle: float = 90.0

    # Head / Nose alignment features
    nose_torso_offset: float = 0.0
    nose_y: float = 0.0
    nose_x: float = 0.0
    head_above_shoulders: bool = True

    # Foot geometry & orientation
    left_foot_angle: float = 0.0
    right_foot_angle: float = 0.0
    feet_parallel_diff: float = 0.0
    heel_distance_ratio: float = 0.0
    foot_index_distance_ratio: float = 0.0
    left_knee_over_ankle_offset: float = 0.0
    right_knee_over_ankle_offset: float = 0.0

    # 3D / World landmark features & alignments
    torso_forward_collapse_angle: float = 0.0
    shoulder_wrist_horizontal_diff_left: float = 0.0
    shoulder_wrist_horizontal_diff_right: float = 0.0
    wrist_vertical_diff: float = 0.0
    hand_foot_distance_ratio: float = 0.0
    lifted_knee_lateral_disp_left: float = 0.0
    lifted_knee_lateral_disp_right: float = 0.0
    has_world_landmarks: bool = False
    world_landmarks: Any = None


def extract_yoga_features(
    pose_landmarks: List[Any],
    world_landmarks: Optional[List[Any]] = None,
) -> YogaFeatures:
    """
    Extracts comprehensive bilateral geometric features from raw pose landmarks.
    Prefers MediaPipe world landmarks (x,y,z in meters) for 3D joint/segment angles
    and orientation when available, and normalized 2D landmarks for screen-relative
    positions, bounds, and vertical/horizontal relationships.
    """
    # Use 3D world landmarks for angles if available, otherwise fallback to normalized
    use_world = world_landmarks is not None and len(world_landmarks) >= 33
    src_3d = world_landmarks if use_world else pose_landmarks

    # 2D Screen Landmark References
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

    # Optional foot landmarks (if present in list)
    hl_l = pose_landmarks[LEFT_HEEL] if len(pose_landmarks) > LEFT_HEEL else ak_l
    hl_r = pose_landmarks[RIGHT_HEEL] if len(pose_landmarks) > RIGHT_HEEL else ak_r
    fi_l = pose_landmarks[LEFT_FOOT_INDEX] if len(pose_landmarks) > LEFT_FOOT_INDEX else ak_l
    fi_r = pose_landmarks[RIGHT_FOOT_INDEX] if len(pose_landmarks) > RIGHT_FOOT_INDEX else ak_r
    ns = pose_landmarks[NOSE] if len(pose_landmarks) > NOSE else sh_l

    # 3D Joint References (for true 3D joint angle calculation)
    w_sh_l = src_3d[LEFT_SHOULDER]
    w_sh_r = src_3d[RIGHT_SHOULDER]
    w_el_l = src_3d[LEFT_ELBOW]
    w_el_r = src_3d[RIGHT_ELBOW]
    w_wr_l = src_3d[LEFT_WRIST]
    w_wr_r = src_3d[RIGHT_WRIST]
    w_hp_l = src_3d[LEFT_HIP]
    w_hp_r = src_3d[RIGHT_HIP]
    w_kn_l = src_3d[LEFT_KNEE]
    w_kn_r = src_3d[RIGHT_KNEE]
    w_ak_l = src_3d[LEFT_ANKLE]
    w_ak_r = src_3d[RIGHT_ANKLE]
    w_hl_l = src_3d[LEFT_HEEL] if len(src_3d) > LEFT_HEEL else w_ak_l
    w_hl_r = src_3d[RIGHT_HEEL] if len(src_3d) > RIGHT_HEEL else w_ak_r
    w_fi_l = src_3d[LEFT_FOOT_INDEX] if len(src_3d) > LEFT_FOOT_INDEX else w_ak_l
    w_fi_r = src_3d[RIGHT_FOOT_INDEX] if len(src_3d) > RIGHT_FOOT_INDEX else w_ak_r

    # Bilateral 3D Joint Angles
    l_knee = calculate_angle(w_hp_l, w_kn_l, w_ak_l, prefer_3d=True)
    r_knee = calculate_angle(w_hp_r, w_kn_r, w_ak_r, prefer_3d=True)
    avg_knee = (l_knee + r_knee) / 2.0

    l_hip = calculate_angle(w_sh_l, w_hp_l, w_kn_l, prefer_3d=True)
    r_hip = calculate_angle(w_sh_r, w_hp_r, w_kn_r, prefer_3d=True)
    avg_hip = (l_hip + r_hip) / 2.0

    l_elbow = calculate_angle(w_sh_l, w_el_l, w_wr_l, prefer_3d=True)
    r_elbow = calculate_angle(w_sh_r, w_el_r, w_wr_r, prefer_3d=True)
    avg_elbow = (l_elbow + r_elbow) / 2.0

    l_shoulder = calculate_angle(w_hp_l, w_sh_l, w_el_l, prefer_3d=True)
    r_shoulder = calculate_angle(w_hp_r, w_sh_r, w_el_r, prefer_3d=True)
    avg_shoulder = (l_shoulder + r_shoulder) / 2.0

    l_ankle = calculate_angle(w_kn_l, w_ak_l, w_fi_l, prefer_3d=True)
    r_ankle = calculate_angle(w_kn_r, w_ak_r, w_fi_r, prefer_3d=True)
    avg_ankle = (l_ankle + r_ankle) / 2.0

    # 2D Screen Torso Angle & Alignments
    torso_ang = calculate_torso_angle(sh_l, sh_r, hp_l, hp_r)
    nose_offset = calculate_nose_torso_offset(ns, sh_l, sh_r, hp_l, hp_r)

    # 3D Torso Forward Collapse
    collapse_ang = calculate_forward_collapse(w_sh_l, w_sh_r, w_hp_l, w_hp_r)

    # Distances & Stance Ratios
    sh_dist = max(1e-4, calculate_distance(sh_l, sh_r))
    hp_dist = max(1e-4, calculate_distance(hp_l, hp_r))
    ak_dist = calculate_distance(ak_l, ak_r)
    ak_h_dist = abs(ak_l.x - ak_r.x)

    stance_ratio = ak_dist / sh_dist
    h_stance_ratio = ak_h_dist / sh_dist
    feet_ratio = ak_dist / hp_dist

    # Heel & Foot Index ratios
    hl_dist = calculate_distance(hl_l, hl_r)
    fi_dist = calculate_distance(fi_l, fi_r)
    heel_ratio = hl_dist / hp_dist
    fi_ratio = fi_dist / hp_dist

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

    # Hand positions
    hands_above = (wr_l.y < min(sh_y_l, sh_y_r)) and (wr_r.y < min(sh_y_l, sh_y_r))
    head_above = ns.y < sh_mid_y
    wrist_v_diff = abs(wr_l.y - wr_r.y)

    sh_wr_diff_l = abs(wr_l.y - sh_y_l) / sh_dist
    sh_wr_diff_r = abs(wr_r.y - sh_y_r) / sh_dist

    # Hand-to-Foot separation distance (normalized by torso length)
    wr_mid_2d = (_xy(wr_l) + _xy(wr_r)) / 2.0
    ak_mid_2d = (_xy(ak_l) + _xy(ak_r)) / 2.0
    torso_len_2d = max(1e-4, float(np.linalg.norm((_xy(sh_l) + _xy(sh_r)) / 2.0 - (_xy(hp_l) + _xy(hp_r)) / 2.0)))
    hand_foot_ratio = float(np.linalg.norm(wr_mid_2d - ak_mid_2d)) / torso_len_2d

    # Foot orientations
    l_foot_ang = calculate_foot_orientation(hl_l, fi_l)
    r_foot_ang = calculate_foot_orientation(hl_r, fi_r)
    feet_par_diff = abs(l_foot_ang - r_foot_ang)
    if feet_par_diff > 180.0:
        feet_par_diff = 360.0 - feet_par_diff

    # Knee over ankle horizontal tracking
    lead_dir_l = np.sign(fi_l.x - ak_l.x) if abs(fi_l.x - ak_l.x) > 1e-4 else -1.0
    lead_dir_r = np.sign(fi_r.x - ak_r.x) if abs(fi_r.x - ak_r.x) > 1e-4 else 1.0
    l_knee_offset = calculate_knee_over_ankle_offset(kn_l, ak_l, fi_l, lead_dir_l)
    r_knee_offset = calculate_knee_over_ankle_offset(kn_r, ak_r, fi_r, lead_dir_r)

    # Lateral displacement of knees from hip midline
    hp_mid_x = (hp_l.x + hp_r.x) / 2.0
    l_knee_disp = abs(kn_l.x - hp_mid_x) / torso_len_2d
    r_knee_disp = abs(kn_r.x - hp_mid_x) / torso_len_2d

    # Overall visibility
    vis_scores = [getattr(pose_landmarks[i], "visibility", 1.0) for i in REQUIRED_LANDMARKS if i < len(pose_landmarks)]
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
        left_ankle_angle=l_ankle,
        right_ankle_angle=r_ankle,
        avg_ankle_angle=avg_ankle,
        nose_torso_offset=nose_offset,
        nose_y=ns.y,
        nose_x=ns.x,
        head_above_shoulders=head_above,
        left_foot_angle=l_foot_ang,
        right_foot_angle=r_foot_ang,
        feet_parallel_diff=feet_par_diff,
        heel_distance_ratio=heel_ratio,
        foot_index_distance_ratio=fi_ratio,
        left_knee_over_ankle_offset=l_knee_offset,
        right_knee_over_ankle_offset=r_knee_offset,
        torso_forward_collapse_angle=collapse_ang,
        shoulder_wrist_horizontal_diff_left=sh_wr_diff_l,
        shoulder_wrist_horizontal_diff_right=sh_wr_diff_r,
        wrist_vertical_diff=wrist_v_diff,
        hand_foot_distance_ratio=hand_foot_ratio,
        lifted_knee_lateral_disp_left=l_knee_disp,
        lifted_knee_lateral_disp_right=r_knee_disp,
        has_world_landmarks=use_world,
        world_landmarks=world_landmarks,
    )

