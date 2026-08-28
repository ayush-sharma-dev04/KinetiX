"""
Yoga Pose Classifier and Multi-Frame Confirmation Engine.
"""
from typing import List, Optional, Tuple
from core.geometry import YogaFeatures
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus
from yoga.poses import get_default_pose_registry


class YogaPoseClassifier:
    """
    Evaluates yoga pose candidates with temporal debouncing.
    Ensures a posture is held steadily across consecutive frames before confirming it.
    
    Lifecycle states:
      - NOT_DETECTED : No matching posture candidate in current frame.
      - CANDIDATE    : Posture detected, accumulating confirmation frames.
      - CONFIRMED    : Posture maintained for required duration, actively checking form.
      - LOST         : Previously confirmed posture lost or interrupted.
    """

    STATE_NOT_DETECTED = "NOT_DETECTED"
    STATE_CANDIDATE = "CANDIDATE"
    STATE_CONFIRMED = "CONFIRMED"
    STATE_LOST = "LOST"

    def __init__(
        self,
        poses: Optional[List[BaseYogaPose]] = None,
        confirm_frames: int = 8,
        lost_frames: int = 25,
    ):
        self.poses = poses if poses is not None else get_default_pose_registry()
        self.confirm_frames = confirm_frames
        self.lost_frames = lost_frames

        # State tracking
        self.candidate_pose: Optional[BaseYogaPose] = None
        self.candidate_count: int = 0

        self.confirmed_pose: Optional[BaseYogaPose] = None
        self.lost_count: int = 0

        self.lifecycle_state: str = self.STATE_NOT_DETECTED

    def update(
        self, f: Optional[YogaFeatures]
    ) -> Tuple[Optional[BaseYogaPose], str, FormEvaluation]:
        """
        Processes a feature frame, updates candidate/confirmed state, and evaluates form.
        
        Returns:
            Tuple[Optional[BaseYogaPose], str, FormEvaluation]:
                - active confirmed pose (or None)
                - lifecycle state string
                - form evaluation result
        """
        if f is None or f.visibility < 0.35:
            # Low visibility or missing person
            return self._handle_no_detection()

        # 1. Evaluate candidate matches across all registered poses
        raw_match: Optional[BaseYogaPose] = None
        best_confidence: float = 0.0

        for pose in self.poses:
            is_match, conf = pose.is_candidate(f)
            if is_match and conf > best_confidence:
                raw_match = pose
                best_confidence = conf

        # 2. Update debounced candidate tracking
        if raw_match is not None:
            if self.candidate_pose == raw_match:
                self.candidate_count += 1
            else:
                self.candidate_pose = raw_match
                self.candidate_count = 1
        else:
            self.candidate_pose = None
            self.candidate_count = 0

        # 3. Transition to CONFIRMED if candidate threshold met
        if self.candidate_pose is not None and self.candidate_count >= self.confirm_frames:
            self.confirmed_pose = self.candidate_pose
            self.lost_count = 0
            self.lifecycle_state = self.STATE_CONFIRMED

        # 4. Handle pose loss debounce if confirmed pose is absent
        if self.confirmed_pose is not None:
            if raw_match == self.confirmed_pose:
                self.lost_count = 0
                self.lifecycle_state = self.STATE_CONFIRMED
            else:
                self.lost_count += 1
                if self.lost_count >= self.lost_frames:
                    # Confirmed pose is fully lost
                    self.confirmed_pose = None
                    self.lost_count = 0
                    self.lifecycle_state = self.STATE_LOST
                else:
                    # Still in grace period, but marked as LOST / unstable
                    self.lifecycle_state = self.STATE_CONFIRMED  # Keep confirmed during brief flutter
        elif self.candidate_pose is not None:
            self.lifecycle_state = self.STATE_CANDIDATE
        else:
            self.lifecycle_state = self.STATE_NOT_DETECTED

        # 5. Form Evaluation
        if self.confirmed_pose is not None:
            form_eval = self.confirmed_pose.evaluate_form(f)
        elif self.candidate_pose is not None:
            # Preliminary preview for candidate
            prelim = self.candidate_pose.evaluate_form(f)
            form_eval = FormEvaluation(
                status="CANDIDATE",
                reasons=[f"Hold steady ({self.candidate_count}/{self.confirm_frames})"],
                metrics=prelim.metrics,
                error_joints=prelim.error_joints,
            )
        else:
            form_eval = FormEvaluation(
                status=FormStatus.LOST,
                reasons=["No pose detected"],
            )

        return self.confirmed_pose, self.lifecycle_state, form_eval

    def _handle_no_detection(self) -> Tuple[Optional[BaseYogaPose], str, FormEvaluation]:
        self.candidate_pose = None
        self.candidate_count = 0

        if self.confirmed_pose is not None:
            self.lost_count += 1
            if self.lost_count >= self.lost_frames:
                self.confirmed_pose = None
                self.lost_count = 0
                self.lifecycle_state = self.STATE_NOT_DETECTED

        form_eval = FormEvaluation(
            status=FormStatus.LOST,
            reasons=["Tracking lost / Person not visible"],
        )
        return self.confirmed_pose, self.lifecycle_state, form_eval

    def reset(self):
        """Reset all tracking states."""
        self.candidate_pose = None
        self.candidate_count = 0
        self.confirmed_pose = None
        self.lost_count = 0
        self.lifecycle_state = self.STATE_NOT_DETECTED
