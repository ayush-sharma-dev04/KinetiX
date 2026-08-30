"""
Dataset logging utility for Yoga pose analysis and ML training data collection.
"""
import os
from typing import List, Dict, Any, Optional
import pandas as pd

from core.geometry import YogaFeatures
from yoga.base_pose import BaseYogaPose, FormEvaluation


YOGA_CSV_COLUMNS = [
    "timestamp",
    "pose_label",
    "sanskrit_name",
    "form_status",
    "feedback_reasons",
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
    "left_ankle_angle",
    "right_ankle_angle",
    "avg_ankle_angle",
    "torso_angle",
    "nose_torso_offset",
    "stance_width_ratio",
    "feet_distance_ratio",
    "shoulder_level_diff",
    "hip_level_diff",
    "visibility",
    "hold_time",
]


class YogaDatasetLogger:
    """
    Buffers and flushes yoga feature records to CSV for supervised ML training.
    """

    def __init__(self, csv_path: str = "yoga_dataset.csv", flush_every: int = 30):
        self.csv_path = csv_path
        self.flush_every = flush_every
        self.buffer: List[Dict[str, Any]] = []

    def log_frame(
        self,
        timestamp_ms: int,
        confirmed_pose: Optional[BaseYogaPose],
        form_eval: FormEvaluation,
        features: Optional[YogaFeatures],
        hold_time: float,
    ) -> None:
        """Appends a single frame record to the buffer."""
        pose_label = confirmed_pose.pose_id if confirmed_pose else "NONE"
        sanskrit_name = confirmed_pose.sanskrit_name if confirmed_pose else "NONE"
        form_status = form_eval.status if form_eval else "LOST"
        reasons_str = "; ".join(form_eval.reasons) if (form_eval and form_eval.reasons) else ""

        if features is not None:
            row = {
                "timestamp": timestamp_ms,
                "pose_label": pose_label,
                "sanskrit_name": sanskrit_name,
                "form_status": form_status,
                "feedback_reasons": reasons_str,
                "left_knee_angle": round(features.left_knee_angle, 1),
                "right_knee_angle": round(features.right_knee_angle, 1),
                "avg_knee_angle": round(features.avg_knee_angle, 1),
                "left_hip_angle": round(features.left_hip_angle, 1),
                "right_hip_angle": round(features.right_hip_angle, 1),
                "avg_hip_angle": round(features.avg_hip_angle, 1),
                "left_elbow_angle": round(features.left_elbow_angle, 1),
                "right_elbow_angle": round(features.right_elbow_angle, 1),
                "avg_elbow_angle": round(features.avg_elbow_angle, 1),
                "left_shoulder_angle": round(features.left_shoulder_angle, 1),
                "right_shoulder_angle": round(features.right_shoulder_angle, 1),
                "avg_shoulder_angle": round(features.avg_shoulder_angle, 1),
                "left_ankle_angle": round(features.left_ankle_angle, 1),
                "right_ankle_angle": round(features.right_ankle_angle, 1),
                "avg_ankle_angle": round(features.avg_ankle_angle, 1),
                "torso_angle": round(features.torso_angle, 1),
                "nose_torso_offset": round(features.nose_torso_offset, 3),
                "stance_width_ratio": round(features.stance_width_ratio, 2),
                "feet_distance_ratio": round(features.feet_distance_ratio, 2),
                "shoulder_level_diff": round(features.shoulder_level_diff, 3),
                "hip_level_diff": round(features.hip_level_diff, 3),
                "visibility": round(features.visibility, 3),
                "hold_time": round(hold_time, 2),
            }
        else:
            row = {
                "timestamp": timestamp_ms,
                "pose_label": pose_label,
                "sanskrit_name": sanskrit_name,
                "form_status": form_status,
                "feedback_reasons": reasons_str,
                "left_knee_angle": None,
                "right_knee_angle": None,
                "avg_knee_angle": None,
                "left_hip_angle": None,
                "right_hip_angle": None,
                "avg_hip_angle": None,
                "left_elbow_angle": None,
                "right_elbow_angle": None,
                "avg_elbow_angle": None,
                "left_shoulder_angle": None,
                "right_shoulder_angle": None,
                "avg_shoulder_angle": None,
                "left_ankle_angle": None,
                "right_ankle_angle": None,
                "avg_ankle_angle": None,
                "torso_angle": None,
                "nose_torso_offset": None,
                "stance_width_ratio": None,
                "feet_distance_ratio": None,
                "shoulder_level_diff": None,
                "hip_level_diff": None,
                "visibility": 0.0,
                "hold_time": round(hold_time, 2),
            }

        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Flushes the buffered frame rows to CSV dataset."""
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer, columns=YOGA_CSV_COLUMNS)
        write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0

        df.to_csv(
            self.csv_path,
            mode="a",
            header=write_header,
            index=False,
        )
        self.buffer.clear()
