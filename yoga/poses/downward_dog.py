"""
Downward-Facing Dog (Adho Mukha Svanasana) detection and form evaluation.
"""
from typing import Tuple, List, Set
from core.geometry import YogaFeatures
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
)
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus


class DownwardDogPose(BaseYogaPose):
    """
    Downward-Facing Dog (Adho Mukha Svanasana):
    Inverted-V configuration where the hips are elevated as the highest point
    of the body, hands and feet are grounded, arms and legs are extended,
    and the spine forms an elongated slope from wrists through shoulders to hips.
    """

    # Candidate thresholds
    CANDIDATE_HIP_Y_OFFSET = 0.03  # Hips must be higher (smaller Y) than shoulders
    CANDIDATE_HIP_ANGLE_MIN = 45.0
    CANDIDATE_HIP_ANGLE_MAX = 120.0
    CANDIDATE_TORSO_ANGLE_MIN = 35.0
    CANDIDATE_KNEE_MIN = 135.0

    # Form quality thresholds
    FORM_HIP_ANGLE_MIN = 60.0
    FORM_HIP_ANGLE_MAX = 105.0
    FORM_KNEE_MIN = 145.0
    FORM_ELBOW_MIN = 150.0
    FORM_SHOULDER_OPEN_MIN = 140.0
    FORM_ANKLE_MIN = 50.0
    FORM_ANKLE_MAX = 130.0

    @property
    def pose_id(self) -> str:
        return "downward_dog"

    @property
    def name(self) -> str:
        return "Downward-Facing Dog"

    @property
    def sanskrit_name(self) -> str:
        return "Adho Mukha Svanasana"

    def is_candidate(self, f: YogaFeatures) -> Tuple[bool, float]:
        # 1. Hips must be elevated above shoulders and knees (smaller Y coordinate)
        if f.hip_mid_y > f.shoulder_mid_y - self.CANDIDATE_HIP_Y_OFFSET:
            return False, 0.0

        if f.hip_mid_y > f.knee_mid_y:
            return False, 0.0

        # 2. Inverted-V angle at hips
        if not (self.CANDIDATE_HIP_ANGLE_MIN <= f.avg_hip_angle <= self.CANDIDATE_HIP_ANGLE_MAX):
            return False, 0.0

        # 3. Torso sloping downward toward floor
        if f.torso_angle < self.CANDIDATE_TORSO_ANGLE_MIN:
            return False, 0.0

        # 4. Legs reasonably extended (not tucked in child's pose)
        if f.avg_knee_angle < self.CANDIDATE_KNEE_MIN:
            return False, 0.0

        # 5. Wrists/hands are near ground (larger Y than hips)
        if min(f.left_wrist_y, f.right_wrist_y) < f.hip_mid_y:
            return False, 0.0

        confidence = 0.95
        return True, float(confidence)

    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        reasons: List[str] = []
        error_joints: Set[int] = set()

        # 1. Hip flexion / Inverted-V peak (shoulder-hip-knee 3D)
        if f.avg_hip_angle > self.FORM_HIP_ANGLE_MAX:
            reasons.append(f"HIPS NOT HIGH ENOUGH ({f.avg_hip_angle:.0f}°) - LIFT TAILBONE UP & BACK")
            error_joints.update([LEFT_HIP, RIGHT_HIP])
        elif f.avg_hip_angle < self.FORM_HIP_ANGLE_MIN:
            reasons.append(f"BODY TOO COMPACT ({f.avg_hip_angle:.0f}°) - STEP HANDS & FEET FURTHER APART")
            error_joints.update([LEFT_HIP, RIGHT_HIP])

        # 2. Knees straight / leg extension (hip-knee-ankle 3D)
        if f.left_knee_angle < self.FORM_KNEE_MIN:
            reasons.append(f"LEFT KNEE BENT ({f.left_knee_angle:.0f}°) - EXTEND LEG & PRESS HEEL DOWN")
            error_joints.add(LEFT_KNEE)
        if f.right_knee_angle < self.FORM_KNEE_MIN:
            reasons.append(f"RIGHT KNEE BENT ({f.right_knee_angle:.0f}°) - EXTEND LEG & PRESS HEEL DOWN")
            error_joints.add(RIGHT_KNEE)

        # 3. Arms and elbows straight (shoulder-elbow-wrist 3D)
        if f.left_elbow_angle < self.FORM_ELBOW_MIN or f.right_elbow_angle < self.FORM_ELBOW_MIN:
            reasons.append("ELBOWS BENT - PUSH FLOOR AWAY FIRMLY THROUGH PALMS")
            error_joints.update([LEFT_ELBOW, RIGHT_ELBOW])

        # 4. Shoulder extension / spinal lengthening (hip-shoulder-elbow 3D)
        if f.avg_shoulder_angle < self.FORM_SHOULDER_OPEN_MIN:
            reasons.append("OPEN SHOULDERS - PRESS CHEST GENTLY TOWARDS THIGHS")
            error_joints.update([LEFT_SHOULDER, RIGHT_SHOULDER])

        # 5. Head / Neck alignment relative to arms and shoulders (relaxed neck, nose behind shoulders)
        if f.nose_y < f.shoulder_mid_y - 0.06:
            reasons.append("RELAX NECK - LET HEAD HANG NATURALLY BETWEEN ARMS")
            error_joints.add(NOSE)

        # 6. Ankle angles and heel grounding
        if (f.left_ankle_angle < self.FORM_ANKLE_MIN or f.left_ankle_angle > self.FORM_ANKLE_MAX or
                f.right_ankle_angle < self.FORM_ANKLE_MIN or f.right_ankle_angle > self.FORM_ANKLE_MAX):
            reasons.append("PRESS HEELS DOWN TOWARDS FLOOR - GROUND FEET")
            error_joints.update([LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL])

        status = FormStatus.CORRECT if len(reasons) == 0 else FormStatus.ADJUST

        metrics = {
            "avg_hip_angle": round(f.avg_hip_angle, 1),
            "left_knee": round(f.left_knee_angle, 1),
            "right_knee": round(f.right_knee_angle, 1),
            "left_elbow": round(f.left_elbow_angle, 1),
            "right_elbow": round(f.right_elbow_angle, 1),
            "avg_shoulder_angle": round(f.avg_shoulder_angle, 1),
            "torso_angle": round(f.torso_angle, 1),
            "left_ankle": round(f.left_ankle_angle, 1),
            "right_ankle": round(f.right_ankle_angle, 1),
            "hand_foot_ratio": round(f.hand_foot_distance_ratio, 2),
        }

        return FormEvaluation(
            status=status,
            reasons=reasons,
            metrics=metrics,
            error_joints=error_joints,
        )

