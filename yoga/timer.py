"""
Robust Hold Timer for Yoga Pose Execution.
"""
from typing import Optional
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus


class YogaHoldTimer:
    """
    Manages hold duration for confirmed yoga poses.
    
    Rules:
      1. Starts timing only when a pose is CONFIRMED and form is CORRECT.
      2. Ticks forward while form remains CORRECT.
      3. PAUSES when form becomes ADJUST (does not reset prematurely).
      4. RESETS only when the pose is lost completely or switched to another pose.
    """

    def __init__(self):
        self.active_pose_id: Optional[str] = None
        self.hold_seconds: float = 0.0
        self.is_holding: bool = False
        self.is_paused: bool = False

    def update(
        self,
        dt: float,
        confirmed_pose: Optional[BaseYogaPose],
        form_eval: FormEvaluation,
    ) -> float:
        """
        Updates hold timer state based on current frame duration `dt` and form status.
        
        Returns:
            float: Current accumulated hold duration in seconds.
        """
        if confirmed_pose is None:
            # Pose lost completely
            self.hold_seconds = 0.0
            self.active_pose_id = None
            self.is_holding = False
            self.is_paused = False
            return 0.0

        # Handle switching to a different confirmed pose
        if self.active_pose_id != confirmed_pose.pose_id:
            self.active_pose_id = confirmed_pose.pose_id
            self.hold_seconds = 0.0

        # Timing progression
        if form_eval.status == FormStatus.CORRECT:
            self.hold_seconds += max(0.0, dt)
            self.is_holding = True
            self.is_paused = False
        elif form_eval.status == FormStatus.ADJUST:
            # Form needs correction - pause timer
            self.is_holding = False
            self.is_paused = (self.hold_seconds > 0.0)
        else:
            self.is_holding = False
            self.is_paused = (self.hold_seconds > 0.0)

        return self.hold_seconds

    def reset(self):
        """Manually reset the timer."""
        self.hold_seconds = 0.0
        self.active_pose_id = None
        self.is_holding = False
        self.is_paused = False

    @property
    def timer_status_str(self) -> str:
        if self.is_holding:
            return "RUNNING"
        elif self.is_paused:
            return "PAUSED"
        elif self.active_pose_id is not None:
            return "READY"
        return "IDLE"
