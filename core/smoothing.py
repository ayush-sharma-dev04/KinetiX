"""
Temporal smoothing filters and debounce state management.
"""
from collections import deque
from typing import Optional, Any
from core.geometry import YogaFeatures


class MovingAverage:
    """Sliding-window moving average filter to denoise per-frame scalar measurements."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        self.values.append(value)
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values.clear()

    @property
    def ready(self) -> bool:
        return len(self.values) > 0


class YogaFeatureSmoother:
    """
    Applies temporal moving average filters across all continuous geometric features.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.filters = {
            "left_knee_angle": MovingAverage(window_size),
            "right_knee_angle": MovingAverage(window_size),
            "avg_knee_angle": MovingAverage(window_size),
            "left_hip_angle": MovingAverage(window_size),
            "right_hip_angle": MovingAverage(window_size),
            "avg_hip_angle": MovingAverage(window_size),
            "left_elbow_angle": MovingAverage(window_size),
            "right_elbow_angle": MovingAverage(window_size),
            "avg_elbow_angle": MovingAverage(window_size),
            "left_shoulder_angle": MovingAverage(window_size),
            "right_shoulder_angle": MovingAverage(window_size),
            "avg_shoulder_angle": MovingAverage(window_size),
            "torso_angle": MovingAverage(window_size),
            "stance_width_ratio": MovingAverage(window_size),
            "horizontal_stance_ratio": MovingAverage(window_size),
            "feet_distance_ratio": MovingAverage(window_size),
            "shoulder_level_diff": MovingAverage(window_size),
            "hip_level_diff": MovingAverage(window_size),
            "visibility": MovingAverage(window_size),
        }

    def update(self, raw: YogaFeatures) -> YogaFeatures:
        """Returns smoothed copy of YogaFeatures."""
        smoothed = YogaFeatures(
            left_knee_angle=self.filters["left_knee_angle"].update(raw.left_knee_angle),
            right_knee_angle=self.filters["right_knee_angle"].update(raw.right_knee_angle),
            avg_knee_angle=self.filters["avg_knee_angle"].update(raw.avg_knee_angle),
            left_hip_angle=self.filters["left_hip_angle"].update(raw.left_hip_angle),
            right_hip_angle=self.filters["right_hip_angle"].update(raw.right_hip_angle),
            avg_hip_angle=self.filters["avg_hip_angle"].update(raw.avg_hip_angle),
            left_elbow_angle=self.filters["left_elbow_angle"].update(raw.left_elbow_angle),
            right_elbow_angle=self.filters["right_elbow_angle"].update(raw.right_elbow_angle),
            avg_elbow_angle=self.filters["avg_elbow_angle"].update(raw.avg_elbow_angle),
            left_shoulder_angle=self.filters["left_shoulder_angle"].update(raw.left_shoulder_angle),
            right_shoulder_angle=self.filters["right_shoulder_angle"].update(raw.right_shoulder_angle),
            avg_shoulder_angle=self.filters["avg_shoulder_angle"].update(raw.avg_shoulder_angle),
            torso_angle=self.filters["torso_angle"].update(raw.torso_angle),
            stance_width_ratio=self.filters["stance_width_ratio"].update(raw.stance_width_ratio),
            horizontal_stance_ratio=self.filters["horizontal_stance_ratio"].update(raw.horizontal_stance_ratio),
            feet_distance_ratio=self.filters["feet_distance_ratio"].update(raw.feet_distance_ratio),
            shoulder_level_diff=self.filters["shoulder_level_diff"].update(raw.shoulder_level_diff),
            hip_level_diff=self.filters["hip_level_diff"].update(raw.hip_level_diff),
            shoulder_mid_y=raw.shoulder_mid_y,
            hip_mid_y=raw.hip_mid_y,
            knee_mid_y=raw.knee_mid_y,
            ankle_mid_y=raw.ankle_mid_y,
            left_wrist_y=raw.left_wrist_y,
            right_wrist_y=raw.right_wrist_y,
            left_shoulder_y=raw.left_shoulder_y,
            right_shoulder_y=raw.right_shoulder_y,
            left_ankle_y=raw.left_ankle_y,
            right_ankle_y=raw.right_ankle_y,
            left_knee_y=raw.left_knee_y,
            right_knee_y=raw.right_knee_y,
            hands_above_head=raw.hands_above_head,
            visibility=self.filters["visibility"].update(raw.visibility),
            landmarks=raw.landmarks,
        )
        return smoothed

    def reset(self):
        for f in self.filters.values():
            f.clear()
