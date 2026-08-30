"""
Mountain Pose (Tadasana) detection and form evaluation.
"""
from typing import Tuple, List, Set
from core.geometry import YogaFeatures
from core.landmarks import (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus


class TadasanaPose(BaseYogaPose):
    """
    Mountain Pose (Tadasana):
    Upright standing posture with feet grounded close together, legs straight,
    torso vertical, shoulders/hips aligned symmetrically, and head centered directly
    over the shoulders and pelvis.
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
    FORM_SHOULDER_DIFF_MAX = 0.10
    FORM_HIP_DIFF_MAX = 0.10
    FORM_HEAD_OFFSET_MAX = 0.12
    FORM_ANKLE_MIN = 60.0
    FORM_ANKLE_MAX = 125.0

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
        # 1. Upright torso check
        if f.torso_angle > self.CANDIDATE_TORSO_MAX:
            return False, 0.0

        # 2. Both legs straight
        if f.left_knee_angle < self.CANDIDATE_KNEE_MIN or f.right_knee_angle < self.CANDIDATE_KNEE_MIN:
            return False, 0.0

        # 3. Both hips extended
        if f.avg_hip_angle < self.CANDIDATE_HIP_MIN:
            return False, 0.0

        # 4. Stance is narrow (not wide warrior or split)
        if f.stance_width_ratio > self.CANDIDATE_STANCE_MAX or f.feet_distance_ratio > self.CANDIDATE_FEET_MAX:
            return False, 0.0

        # 5. Leg symmetry (ensure not in Tree pose)
        knee_diff = abs(f.left_knee_angle - f.right_knee_angle)
        if knee_diff > 25.0:
            return False, 0.0

        # 6. Vertical alignment of feet on floor
        ankle_y_diff = abs(f.left_ankle_y - f.right_ankle_y)
        if ankle_y_diff > 0.08:
            return False, 0.0

        # 7. Head above shoulders
        if not f.head_above_shoulders and f.nose_y > f.shoulder_mid_y:
            return False, 0.0

        # 8. Arms should not be in wide T-pose (Warrior II)
        if f.left_shoulder_angle > 70.0 and f.right_shoulder_angle > 70.0 and f.stance_width_ratio > 1.2:
            return False, 0.0

        confidence = 0.95 - (f.torso_angle / 50.0)
        return True, float(max(0.5, min(1.0, confidence)))

    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        reasons: List[str] = []
        error_joints: Set[int] = set()

        # 1. Torso vertical alignment (shoulder & hip midpoints)
        if f.torso_angle > self.FORM_TORSO_MAX:
            reasons.append("TORSO LEANING - STAND TALL AND UPRIGHT")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        # 2. Knee extension (hip-knee-ankle 3D)
        if f.left_knee_angle < self.FORM_KNEE_MIN:
            reasons.append(f"LEFT KNEE BENT ({f.left_knee_angle:.0f}°) - STRAIGHTEN LEFT LEG")
            error_joints.add(LEFT_KNEE)
        if f.right_knee_angle < self.FORM_KNEE_MIN:
            reasons.append(f"RIGHT KNEE BENT ({f.right_knee_angle:.0f}°) - STRAIGHTEN RIGHT LEG")
            error_joints.add(RIGHT_KNEE)

        # 3. Hip extension (shoulder-hip-knee 3D)
        if f.left_hip_angle < self.FORM_HIP_MIN or f.right_hip_angle < self.FORM_HIP_MIN:
            reasons.append("HIPS SLIGHTLY FLEXED - ENGAGE GLUTES AND CORE")
            error_joints.update([LEFT_HIP, RIGHT_HIP])

        # 4. Head -> shoulders -> hips vertical alignment (nose relative to spine axis)
        if f.nose_torso_offset > self.FORM_HEAD_OFFSET_MAX:
            reasons.append("HEAD MISALIGNED - KEEP HEAD CENTERED OVER SHOULDERS")
            error_joints.add(NOSE)

        # 5. Shoulder symmetry
        if f.shoulder_level_diff > self.FORM_SHOULDER_DIFF_MAX:
            reasons.append("UNEVEN SHOULDERS - RELAX AND LEVEL SHOULDERS")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER])

        # 6. Hip symmetry
        if f.hip_level_diff > self.FORM_HIP_DIFF_MAX:
            reasons.append("UNEVEN HIPS - DISTRIBUTE WEIGHT EVENLY")
            error_joints.update([LEFT_HIP, RIGHT_HIP])

        # 7. Ankle / Foot alignment and stance width
        if f.feet_distance_ratio > self.FORM_FEET_RATIO_MAX:
            reasons.append("FEET TOO WIDE - BRING FEET CLOSER TOGETHER")
            error_joints.update([LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX])

        # 8. Ankle angles (knee-ankle-foot index 3D)
        if f.left_ankle_angle < self.FORM_ANKLE_MIN or f.left_ankle_angle > self.FORM_ANKLE_MAX:
            reasons.append("LEFT ANKLE UNSTABLE - GROUND FOOT EVENLY")
            error_joints.update([LEFT_ANKLE, LEFT_FOOT_INDEX])
        if f.right_ankle_angle < self.FORM_ANKLE_MIN or f.right_ankle_angle > self.FORM_ANKLE_MAX:
            reasons.append("RIGHT ANKLE UNSTABLE - GROUND FOOT EVENLY")
            error_joints.update([RIGHT_ANKLE, RIGHT_FOOT_INDEX])

        status = FormStatus.CORRECT if len(reasons) == 0 else FormStatus.ADJUST

        metrics = {
            "left_knee": round(f.left_knee_angle, 1),
            "right_knee": round(f.right_knee_angle, 1),
            "left_hip": round(f.left_hip_angle, 1),
            "right_hip": round(f.right_hip_angle, 1),
            "torso_angle": round(f.torso_angle, 1),
            "feet_ratio": round(f.feet_distance_ratio, 2),
            "shoulder_diff": round(f.shoulder_level_diff, 2),
            "hip_diff": round(f.hip_level_diff, 2),
            "head_offset": round(f.nose_torso_offset, 2),
            "left_ankle": round(f.left_ankle_angle, 1),
            "right_ankle": round(f.right_ankle_angle, 1),
        }

        return FormEvaluation(
            status=status,
            reasons=reasons,
            metrics=metrics,
            error_joints=error_joints,
        )

