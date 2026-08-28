"""
Tree Pose (Vrksasana) detection and form evaluation.
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


class TreePose(BaseYogaPose):
    """
    Tree Pose (Vrksasana):
    Single-leg standing balance pose with one leg straight (support) and the
    opposite leg bent with the foot placed against the inner calf or thigh.
    Torso is tall and upright.
    Supports Left-Leg-Standing and Right-Leg-Standing.
    """

    # Candidate thresholds
    CANDIDATE_TORSO_MAX = 18.0
    CANDIDATE_SUPPORT_KNEE_MIN = 150.0
    CANDIDATE_LIFTED_KNEE_MAX = 135.0
    CANDIDATE_KNEE_DIFF_MIN = 28.0
    CANDIDATE_H_STANCE_MAX = 1.35  # Tree pose has narrow horizontal base of support

    # Form quality thresholds
    FORM_SUPPORT_KNEE_MIN = 165.0
    FORM_LIFTED_KNEE_MAX = 115.0
    FORM_TORSO_MAX = 10.0
    FORM_HIP_DIFF_MAX = 0.15

    @property
    def pose_id(self) -> str:
        return "tree_pose"

    @property
    def name(self) -> str:
        return "Tree Pose"

    @property
    def sanskrit_name(self) -> str:
        return "Vrksasana"

    def _determine_support_leg(self, f: YogaFeatures) -> str:
        """Returns 'LEFT' if left leg is support (straight), else 'RIGHT'."""
        if f.left_knee_angle > f.right_knee_angle:
            return "LEFT"
        return "RIGHT"

    def is_candidate(self, f: YogaFeatures) -> Tuple[bool, float]:
        # Upright torso
        if f.torso_angle > self.CANDIDATE_TORSO_MAX:
            return False, 0.0

        support_leg = self._determine_support_leg(f)
        supp_knee = f.left_knee_angle if support_leg == "LEFT" else f.right_knee_angle
        lift_knee = f.right_knee_angle if support_leg == "LEFT" else f.left_knee_angle

        # Significant asymmetry between straight support leg and bent lifted leg
        knee_diff = supp_knee - lift_knee
        if knee_diff < self.CANDIDATE_KNEE_DIFF_MIN:
            return False, 0.0

        if supp_knee < self.CANDIDATE_SUPPORT_KNEE_MIN:
            return False, 0.0

        if lift_knee > self.CANDIDATE_LIFTED_KNEE_MAX:
            return False, 0.0

        # Narrow horizontal base of support (not wide warrior)
        if f.horizontal_stance_ratio > self.CANDIDATE_H_STANCE_MAX:
            return False, 0.0

        # Lifted ankle must be elevated off the ground relative to support ankle
        supp_ankle_y = f.left_ankle_y if support_leg == "LEFT" else f.right_ankle_y
        lift_ankle_y = f.right_ankle_y if support_leg == "LEFT" else f.left_ankle_y

        if lift_ankle_y > supp_ankle_y - 0.04:  # In image coords, higher on screen = smaller Y
            return False, 0.0

        confidence = 0.92
        return True, float(confidence)

    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        reasons: List[str] = []
        error_joints: Set[int] = set()

        support_leg = self._determine_support_leg(f)
        if support_leg == "LEFT":
            supp_knee = f.left_knee_angle
            lift_knee = f.right_knee_angle
            supp_knee_idx = LEFT_KNEE
            lift_knee_idx = RIGHT_KNEE
            supp_ankle_y = f.left_ankle_y
            lift_ankle_y = f.right_ankle_y
            lift_ankle_idx = RIGHT_ANKLE
        else:
            supp_knee = f.right_knee_angle
            lift_knee = f.left_knee_angle
            supp_knee_idx = RIGHT_KNEE
            lift_knee_idx = LEFT_KNEE
            supp_ankle_y = f.right_ankle_y
            lift_ankle_y = f.left_ankle_y
            lift_ankle_idx = LEFT_ANKLE

        # 1. Standing leg locked & straight
        if supp_knee < self.FORM_SUPPORT_KNEE_MIN:
            reasons.append(f"SUPPORT KNEE BENT ({supp_knee:.0f}°) - LOCK STANDING LEG")
            error_joints.add(supp_knee_idx)

        # 2. Lifted leg adequately bent and open
        if lift_knee > self.FORM_LIFTED_KNEE_MAX:
            reasons.append(f"LIFTED LEG NOT BENT ENOUGH ({lift_knee:.0f}°) - BEND & OPEN HIP")
            error_joints.add(lift_knee_idx)

        # 3. Lifted foot elevation
        if lift_ankle_y > supp_ankle_y - 0.08:
            reasons.append("LIFT FOOT HIGHER - PLACE ON INNER CALF OR THIGH")
            error_joints.add(lift_ankle_idx)

        # 4. Torso upright
        if f.torso_angle > self.FORM_TORSO_MAX:
            reasons.append(f"TORSO LEANING ({f.torso_angle:.0f}°) - ALIGN HEAD OVER PELVIS")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        # 5. Pelvis level
        if f.hip_level_diff > self.FORM_HIP_DIFF_MAX:
            reasons.append("HIPS UNBALANCED - LEVEL YOUR PELVIS")
            error_joints.update([LEFT_HIP, RIGHT_HIP])

        status = FormStatus.CORRECT if len(reasons) == 0 else FormStatus.ADJUST

        metrics = {
            "support_leg": f"{support_leg}_LEG_STANDING",
            "support_knee": round(supp_knee, 1),
            "lifted_knee": round(lift_knee, 1),
            "torso_angle": round(f.torso_angle, 1),
            "hip_diff": round(f.hip_level_diff, 2),
        }

        return FormEvaluation(
            status=status,
            reasons=reasons,
            metrics=metrics,
            error_joints=error_joints,
        )
