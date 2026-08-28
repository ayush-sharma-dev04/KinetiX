"""
MediaPipe Pose Landmarker initialization wrapper.
"""
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def create_landmarker(model_path: str = "pose_landmarker_lite.task") -> vision.PoseLandmarker:
    """Initialize MediaPipe Pose Landmarker for video mode."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MediaPipe pose landmarker model not found at: {model_path}. "
            f"Please ensure pose_landmarker_lite.task is in the working directory."
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)
