"""
MediaPipe Skeleton and Pose Visualization Drawing Utilities.
"""
from typing import List, Optional, Tuple, Set, Any
import cv2
import numpy as np

from core.landmarks import POSE_CONNECTIONS

# Default visibility thresholds
VISIBILITY_THRESHOLD = 0.5
DRAW_VISIBILITY_THRESHOLD = 0.2


def draw_pose_skeleton(
    frame: np.ndarray,
    pose_landmarks: List[Any],
    highlight_joints: Optional[Set[int]] = None,
    min_visibility: float = VISIBILITY_THRESHOLD,
    draw_visibility: float = DRAW_VISIBILITY_THRESHOLD,
) -> None:
    """
    Draws 33 pose landmarks and skeleton connections on the frame,
    color-coded by landmark visibility confidence.
    
    Green  = Confident (visibility >= min_visibility)
    Orange = Borderline (draw_visibility <= visibility < min_visibility)
    Red/Cyan = Highlighted joints (e.g. form error points)
    """
    h, w = frame.shape[:2]
    highlight_joints = highlight_joints or set()

    def point_px(lm):
        return int(lm.x * w), int(lm.y * h)

    def color_for(visibility):
        if visibility >= min_visibility:
            return (0, 230, 118)   # Bright green (BGR)
        return (0, 165, 255)       # Orange (BGR)

    # 1. Draw connection lines
    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = pose_landmarks[start_idx]
        end_lm = pose_landmarks[end_idx]

        v_start = getattr(start_lm, "visibility", 1.0)
        v_end = getattr(end_lm, "visibility", 1.0)

        if v_start < draw_visibility or v_end < draw_visibility:
            continue

        edge_color = color_for(min(v_start, v_end))
        p1 = point_px(start_lm)
        p2 = point_px(end_lm)

        cv2.line(frame, p1, p2, edge_color, 2, cv2.LINE_AA)

    # 2. Draw landmark dots
    for idx, lm in enumerate(pose_landmarks):
        v = getattr(lm, "visibility", 1.0)
        if v < draw_visibility:
            continue

        pt = point_px(lm)
        if idx in highlight_joints:
            # Highlight joint with warning circle
            cv2.circle(frame, pt, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 9, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, color_for(v), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (20, 20, 20), 1, cv2.LINE_AA)
