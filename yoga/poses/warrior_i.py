"""
Warrior I Pose (Virabhadrasana I) detection and form evaluation.
"""
from typing import Tuple, List, Set
from core.geometry import YogaFeatures
from core.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
)
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus


class WarriorIPose(BaseYogaPose):
    """
    Warrior I (Virabhadrasana I):
    Lunge posture with front knee bent (~90°), rear leg extended straight,
    torso upright, and both arms extended straight overhead towards the sky.
    Supports both Left-Forward and Right-Forward orientations.
    """

    # Candidate thresholds
    CANDIDATE_STANCE_MIN = 1.25
    CANDIDATE_TORSO_MAX = 28.0
    CANDIDATE_FRONT_KNEE_MAX = 138.0
    CANDIDATE_FRONT_KNEE_MIN = 70.0
    CANDIDATE_REAR_KNEE_MIN = 142.0
    CANDIDATE_SHOULDER_MIN = 125.0

    # Form quality thresholds
    FORM_FRONT_KNEE_MIN = 82.0
    FORM_FRONT_KNEE_MAX = 115.0
    FORM_REAR_KNEE_MIN = 155.0
    FORM_ARM_OVERHEAD_MIN = 145.0
    FORM_ELBOW_MIN = 145.0
    FORM_TORSO_MAX = 15.0
    FORM_STANCE_MIN = 1.40

    @property
    def pose_id(self) -> str:
        return "warrior_i"

    @property
    def name(self) -> str:
        return "Warrior I"

    @property
    def sanskrit_name(self) -> str:
        return "Virabhadrasana I"

    def _determine_lead_leg(self, f: YogaFeatures) -> str:
        """Returns 'LEFT' if left knee is more bent (forward), else 'RIGHT'."""
        if f.left_knee_angle < f.right_knee_angle:
            return "LEFT"
        return "RIGHT"

    def is_candidate(self, f: YogaFeatures) -> Tuple[bool, float]:
        # Stance must be wide
        if f.stance_width_ratio < self.CANDIDATE_STANCE_MIN:
            return False, 0.0

        # Torso generally upright
        if f.torso_angle > self.CANDIDATE_TORSO_MAX:
            return False, 0.0

        # Asymmetric knees: one bent, one straight
        lead_leg = self._determine_lead_leg(f)
        front_knee = f.left_knee_angle if lead_leg == "LEFT" else f.right_knee_angle
        rear_knee = f.right_knee_angle if lead_leg == "LEFT" else f.left_knee_angle

        if not (self.CANDIDATE_FRONT_KNEE_MIN <= front_knee <= self.CANDIDATE_FRONT_KNEE_MAX):
            return False, 0.0

        if rear_knee < self.CANDIDATE_REAR_KNEE_MIN:
            return False, 0.0

        # Arms must be raised overhead (not horizontal)
        if not f.hands_above_head:
            return False, 0.0

        if f.avg_shoulder_angle < self.CANDIDATE_SHOULDER_MIN:
            return False, 0.0

        confidence = 0.90
        return True, float(confidence)

    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        reasons: List[str] = []
        error_joints: Set[int] = set()

        lead_leg = self._determine_lead_leg(f)
        if lead_leg == "LEFT":
            front_knee = f.left_knee_angle
            rear_knee = f.right_knee_angle
            front_knee_idx = LEFT_KNEE
            rear_knee_idx = RIGHT_KNEE
        else:
            front_knee = f.right_knee_angle
            rear_knee = f.left_knee_angle
            front_knee_idx = RIGHT_KNEE
            rear_knee_idx = LEFT_KNEE

        # 1. Front knee angle
        if front_knee > self.FORM_FRONT_KNEE_MAX:
            reasons.append(f"FRONT KNEE TOO STRAIGHT ({front_knee:.0f}°) - SINK DEEPER TO 90°")
            error_joints.add(front_knee_idx)
        elif front_knee < self.FORM_FRONT_KNEE_MIN:
            reasons.append(f"FRONT KNEE OVERSHOOTING ({front_knee:.0f}°) - AVOID OVER-BENDING")
            error_joints.add(front_knee_idx)

        # 2. Rear leg straight
        if rear_knee < self.FORM_REAR_KNEE_MIN:
            reasons.append(f"BACK KNEE BENT ({rear_knee:.0f}°) - PRESS HEEL DOWN & EXTEND")
            error_joints.add(rear_knee_idx)

        # 3. Overhead arm extension
        if f.left_shoulder_angle < self.FORM_ARM_OVERHEAD_MIN:
            reasons.append("LEFT ARM NOT FULLY OVERHEAD - REACH UPWARDS")
            error_joints.update([LEFT_SHOULDER, LEFT_WRIST])

        if f.right_shoulder_angle < self.FORM_ARM_OVERHEAD_MIN:
            reasons.append("RIGHT ARM NOT FULLY OVERHEAD - REACH UPWARDS")
            error_joints.update([RIGHT_SHOULDER, RIGHT_WRIST])

        # 4. Straight elbows
        if f.left_elbow_angle < self.FORM_ELBOW_MIN or f.right_elbow_angle < self.FORM_ELBOW_MIN:
            reasons.append("ELBOWS BENT - EXTEND ARMS FULLY OVERHEAD")
            error_joints.update([LEFT_ELBOW, RIGHT_ELBOW])

        # 5. Torso upright & lifted
        if f.torso_angle > self.FORM_TORSO_MAX:
            reasons.append(f"TORSO LEANING ({f.torso_angle:.0f}°) - LIFT CHEST & ENGAGE CORE")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        # 6. Stance width
        if f.stance_width_ratio < self.FORM_STANCE_MIN:
            reasons.append("STANCE TOO NARROW - STEP FEET FURTHER APART")
            error_joints.update([LEFT_ANKLE, RIGHT_ANKLE])

        status = FormStatus.CORRECT if len(reasons) == 0 else FormStatus.ADJUST

        metrics = {
            "orientation": f"{lead_leg}_FORWARD",
            "front_knee": round(front_knee, 1),
            "rear_knee": round(rear_knee, 1),
            "torso_angle": round(f.torso_angle, 1),
            "left_arm": round(f.left_shoulder_angle, 1),
            "right_arm": round(f.right_shoulder_angle, 1),
            "stance_ratio": round(f.stance_width_ratio, 2),
        }

        return FormEvaluation(
            status=status,
            reasons=reasons,
            metrics=metrics,
            error_joints=error_joints,
        )
