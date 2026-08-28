"""
Mountain Pose (Tadasana) detection and form evaluation.
"""
from typing import Tuple, List, Set
from core.geometry import YogaFeatures
from core.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus


class TadasanaPose(BaseYogaPose):
    """
    Mountain Pose (Tadasana):
    Upright standing posture with feet grounded close together, legs straight,
    torso vertical, and shoulders/hips aligned symmetrically.
    """

    # Configurable candidate thresholds
    CANDIDATE_TORSO_MAX = 18.0
    CANDIDATE_KNEE_MIN = 155.0
    CANDIDATE_HIP_MIN = 150.0
    CANDIDATE_STANCE_MAX = 1.35
    CANDIDATE_FEET_MAX = 1.75

    # Configurable form quality thresholds
    FORM_TORSO_MAX = 10.0
    FORM_KNEE_MIN = 165.0
    FORM_HIP_MIN = 165.0
    FORM_FEET_RATIO_MAX = 1.25
    FORM_SHOULDER_DIFF_MAX = 0.12
    FORM_HIP_DIFF_MAX = 0.12

    @property
    def pose_id(self) -> str:
        return "tadasana"

    @property
    def name(self) -> str:
        return "Mountain Pose"

    @property
    def sanskrit_name(self) -> str:
        return "Tadasana"

    def is_candidate(self, f: YogaFeatures) -> Tuple[bool, float]:
        # Upright torso check
        if f.torso_angle > self.CANDIDATE_TORSO_MAX:
            return False, 0.0

        # Both legs straight
        if f.left_knee_angle < self.CANDIDATE_KNEE_MIN or f.right_knee_angle < self.CANDIDATE_KNEE_MIN:
            return False, 0.0

        # Both hips extended
        if f.avg_hip_angle < self.CANDIDATE_HIP_MIN:
            return False, 0.0

        # Stance is narrow (not wide warrior or split)
        if f.stance_width_ratio > self.CANDIDATE_STANCE_MAX or f.feet_distance_ratio > self.CANDIDATE_FEET_MAX:
            return False, 0.0

        # Leg symmetry (ensure not in Tree pose)
        knee_diff = abs(f.left_knee_angle - f.right_knee_angle)
        if knee_diff > 30.0:
            return False, 0.0

        # Vertical alignment of feet
        ankle_y_diff = abs(f.left_ankle_y - f.right_ankle_y)
        if ankle_y_diff > 0.08:
            return False, 0.0

        # Arms should not be in T-pose (Warrior II)
        if f.left_shoulder_angle > 70.0 and f.right_shoulder_angle > 70.0 and f.stance_width_ratio > 1.2:
            return False, 0.0

        confidence = 0.95 - (f.torso_angle / 50.0)
        return True, float(max(0.5, min(1.0, confidence)))

    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        reasons: List[str] = []
        error_joints: Set[int] = set()

        # 1. Torso vertical alignment
        if f.torso_angle > self.FORM_TORSO_MAX:
            reasons.append("TORSO LEANING - STAND TALL AND UPRIGHT")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        # 2. Knee extension
        if f.left_knee_angle < self.FORM_KNEE_MIN:
            reasons.append("LEFT KNEE BENT - STRAIGHTEN LEFT LEG")
            error_joints.add(LEFT_KNEE)
        if f.right_knee_angle < self.FORM_KNEE_MIN:
            reasons.append("RIGHT KNEE BENT - STRAIGHTEN RIGHT LEG")
            error_joints.add(RIGHT_KNEE)

        # 3. Hip extension
        if f.avg_hip_angle < self.FORM_HIP_MIN:
            reasons.append("HIPS SLIGHTLY FLEXED - ENGAGE GLUTES AND CORE")
            error_joints.update([LEFT_HIP, RIGHT_HIP])

        # 4. Feet proximity
        if f.feet_distance_ratio > self.FORM_FEET_RATIO_MAX:
            reasons.append("FEET TOO WIDE - BRING FEET CLOSER TOGETHER")
            error_joints.update([LEFT_ANKLE, RIGHT_ANKLE])

        # 5. Shoulder symmetry
        if f.shoulder_level_diff > self.FORM_SHOULDER_DIFF_MAX:
            reasons.append("UNEVEN SHOULDERS - RELAX AND LEVEL SHOULDERS")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER])

        # 6. Hip symmetry
        if f.hip_level_diff > self.FORM_HIP_DIFF_MAX:
            reasons.append("UNEVEN HIPS - DISTRIBUTE WEIGHT EVENLY")
            error_joints.update([LEFT_HIP, RIGHT_HIP])

        status = FormStatus.CORRECT if len(reasons) == 0 else FormStatus.ADJUST

        metrics = {
            "avg_knee": round(f.avg_knee_angle, 1),
            "avg_hip": round(f.avg_hip_angle, 1),
            "torso_angle": round(f.torso_angle, 1),
            "feet_ratio": round(f.feet_distance_ratio, 2),
            "shoulder_diff": round(f.shoulder_level_diff, 2),
        }

        return FormEvaluation(
            status=status,
            reasons=reasons,
            metrics=metrics,
            error_joints=error_joints,
        )
