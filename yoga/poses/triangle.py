"""
Triangle Pose (Trikonasana) detection and form evaluation.
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


class TrianglePose(BaseYogaPose):
    """
    Triangle Pose (Trikonasana):
    Wide standing posture with both legs straight, torso laterally tilted to one side,
    one hand reaching down towards the front shin/ankle, and the opposite arm
    extended vertically upward towards the ceiling forming a straight vertical line.
    Supports both Left-Side and Right-Side tilt orientations.
    """

    # Candidate thresholds
    CANDIDATE_STANCE_MIN = 1.30
    CANDIDATE_TORSO_MIN = 22.0
    CANDIDATE_TORSO_MAX = 78.0
    CANDIDATE_KNEE_MIN = 145.0
    CANDIDATE_WRIST_DIFF_MIN = 0.20  # Significant vertical distance between two wrists

    # Form quality thresholds
    FORM_KNEE_MIN = 155.0
    FORM_TORSO_MIN = 30.0
    FORM_TORSO_MAX = 68.0
    FORM_ELBOW_MIN = 145.0
    FORM_STANCE_MIN = 1.50

    @property
    def pose_id(self) -> str:
        return "triangle_pose"

    @property
    def name(self) -> str:
        return "Triangle Pose"

    @property
    def sanskrit_name(self) -> str:
        return "Trikonasana"

    def _determine_side(self, f: YogaFeatures) -> str:
        """
        Determines tilt direction.
        If left shoulder is lower on screen (larger Y), tilting to LEFT, else RIGHT.
        """
        if f.left_shoulder_y > f.right_shoulder_y:
            return "LEFT"
        return "RIGHT"

    def is_candidate(self, f: YogaFeatures) -> Tuple[bool, float]:
        # Stance must be wide
        if f.stance_width_ratio < self.CANDIDATE_STANCE_MIN:
            return False, 0.0

        # Torso laterally tilted (not upright and not inverted dog)
        if not (self.CANDIDATE_TORSO_MIN <= f.torso_angle <= self.CANDIDATE_TORSO_MAX):
            return False, 0.0

        # Both legs straight (unlike Warrior where one leg is deeply bent)
        if f.left_knee_angle < self.CANDIDATE_KNEE_MIN or f.right_knee_angle < self.CANDIDATE_KNEE_MIN:
            return False, 0.0

        # Vertical separation of hands (one up, one down)
        wrist_y_diff = abs(f.left_wrist_y - f.right_wrist_y)
        if wrist_y_diff < self.CANDIDATE_WRIST_DIFF_MIN:
            return False, 0.0

        # Hips must not be the highest point of the body (excludes Downward Dog)
        if f.hip_mid_y < min(f.left_shoulder_y, f.right_shoulder_y):
            return False, 0.0

        confidence = 0.90
        return True, float(confidence)

    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        reasons: List[str] = []
        error_joints: Set[int] = set()

        side = self._determine_side(f)
        if side == "LEFT":
            top_wrist_y = f.right_wrist_y
            top_shoulder_y = f.right_shoulder_y
            top_shoulder_idx = RIGHT_SHOULDER
            top_wrist_idx = RIGHT_WRIST
        else:
            top_wrist_y = f.left_wrist_y
            top_shoulder_y = f.left_shoulder_y
            top_shoulder_idx = LEFT_SHOULDER
            top_wrist_idx = LEFT_WRIST

        # 1. Both legs straight
        if f.left_knee_angle < self.FORM_KNEE_MIN:
            reasons.append(f"LEFT KNEE BENT ({f.left_knee_angle:.0f}°) - STRAIGHTEN BOTH LEGS")
            error_joints.add(LEFT_KNEE)
        if f.right_knee_angle < self.FORM_KNEE_MIN:
            reasons.append(f"RIGHT KNEE BENT ({f.right_knee_angle:.0f}°) - STRAIGHTEN BOTH LEGS")
            error_joints.add(RIGHT_KNEE)

        # 2. Torso lateral tilt depth
        if f.torso_angle < self.FORM_TORSO_MIN:
            reasons.append(f"TORSO TILT TOO SHALLOW ({f.torso_angle:.0f}°) - HINGE DEEPER AT HIP")
            error_joints.update([LEFT_HIP, RIGHT_HIP])
        elif f.torso_angle > self.FORM_TORSO_MAX:
            reasons.append(f"TORSO COLLAPSED ({f.torso_angle:.0f}°) - LENGTHEN BOTH SIDES OF WAIST")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])

        # 3. Top arm reaching vertically upward
        if top_wrist_y > top_shoulder_y - 0.05:  # Top wrist should be above top shoulder
            reasons.append("EXTEND TOP ARM DIRECTLY TOWARDS THE SKY")
            error_joints.update([top_shoulder_idx, top_wrist_idx])

        # 4. Straight arms
        if f.left_elbow_angle < self.FORM_ELBOW_MIN or f.right_elbow_angle < self.FORM_ELBOW_MIN:
            reasons.append("ELBOWS BENT - EXTEND BOTH ARMS STRAIGHT")
            error_joints.update([LEFT_ELBOW, RIGHT_ELBOW])

        # 5. Stance width
        if f.stance_width_ratio < self.FORM_STANCE_MIN:
            reasons.append("STANCE TOO NARROW - STEP FEET WIDER APART")
            error_joints.update([LEFT_ANKLE, RIGHT_ANKLE])

        status = FormStatus.CORRECT if len(reasons) == 0 else FormStatus.ADJUST

        metrics = {
            "side": f"{side}_SIDE_TILT",
            "left_knee": round(f.left_knee_angle, 1),
            "right_knee": round(f.right_knee_angle, 1),
            "torso_angle": round(f.torso_angle, 1),
            "stance_ratio": round(f.stance_width_ratio, 2),
        }

        return FormEvaluation(
            status=status,
            reasons=reasons,
            metrics=metrics,
            error_joints=error_joints,
        )
