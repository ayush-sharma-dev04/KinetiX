"""
Translucent Card HUD Visualizer for Yoga Pose and Form Feedback.
"""
from typing import Optional, List, Tuple
import cv2
import numpy as np

from core.geometry import YogaFeatures
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus


class YogaHUD:
    """
    Renders a modern top-left translucent card overlay displaying:
      - Pose Name (English & Sanskrit)
      - Form Status (CORRECT, ADJUST, CANDIDATE, POSE LOST)
      - Real-time Hold Timer with status [RUNNING / PAUSED / IDLE]
      - Actionable Form Feedback Reasons
      - Live Joint & Torso Angles
      - Visibility Confidence
    """

    # Color definitions in BGR format
    COLOR_CORRECT = (0, 230, 118)      # Bright vibrant green
    COLOR_ADJUST = (0, 195, 255)       # Amber / Yellow-orange
    COLOR_CANDIDATE = (255, 215, 0)    # Cyan / Sky blue
    COLOR_LOST = (60, 60, 255)         # Red
    COLOR_TEXT_PRIMARY = (255, 255, 255) # White
    COLOR_TEXT_SECONDARY = (200, 205, 210) # Light grey
    COLOR_BG = (18, 18, 22)            # Dark slate

    def __init__(self, top_left: Tuple[int, int] = (15, 15)):
        self.top_left = top_left

    def draw(
        self,
        frame: np.ndarray,
        confirmed_pose: Optional[BaseYogaPose],
        lifecycle_state: str,
        form_eval: FormEvaluation,
        hold_seconds: float,
        timer_status_str: str,
        features: Optional[YogaFeatures],
    ) -> None:
        """
        Renders HUD card overlay on frame.
        """
        h, w = frame.shape[:2]
        x0, y0 = self.top_left

        # Calculate card dimensions
        reasons_to_show = form_eval.reasons[:2] if form_eval else []
        extra_h = len(reasons_to_show) * 22 if (form_eval and form_eval.status == FormStatus.ADJUST) else 0
        card_w = 380
        card_h = 245 + extra_h

        # Create translucent background box
        x1 = min(w - 10, x0 + card_w)
        y1 = min(h - 10, y0 + card_h)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            self.COLOR_BG,
            -1
        )
        # 80% opacity overlay
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Draw subtle border
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 65, 75), 1, cv2.LINE_AA)

        # 1. Pose Title
        if confirmed_pose is not None:
            pose_title = f"{confirmed_pose.name.upper()}"
            sub_title = f"({confirmed_pose.sanskrit_name})"
        elif lifecycle_state == "CANDIDATE":
            pose_title = "DETECTING..."
            sub_title = "HOLD POSITION"
        else:
            pose_title = "SEARCHING POSE"
            sub_title = "Align body in frame"

        cv2.putText(
            frame,
            f"POSE: {pose_title}",
            (x0 + 14, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.COLOR_TEXT_PRIMARY,
            2,
            cv2.LINE_AA,
        )

        # 2. Form Status & Color
        if form_eval.status == FormStatus.CORRECT:
            status_text = "FORM: CORRECT"
            status_color = self.COLOR_CORRECT
        elif form_eval.status == FormStatus.ADJUST:
            status_text = "FORM: ADJUST"
            status_color = self.COLOR_ADJUST
        elif form_eval.status == "CANDIDATE":
            status_text = "STATUS: CANDIDATE"
            status_color = self.COLOR_CANDIDATE
        else:
            status_text = "STATUS: POSE LOST"
            status_color = self.COLOR_LOST

        cv2.putText(
            frame,
            status_text,
            (x0 + 14, y0 + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # 3. Hold Timer
        timer_text = f"HOLD: {hold_seconds:.1f} s"
        timer_tag = f"[{timer_status_str}]"
        timer_tag_color = (
            self.COLOR_CORRECT if timer_status_str == "RUNNING"
            else self.COLOR_ADJUST if timer_status_str == "PAUSED"
            else self.COLOR_TEXT_SECONDARY
        )

        cv2.putText(
            frame,
            timer_text,
            (x0 + 14, y0 + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            self.COLOR_TEXT_PRIMARY,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            timer_tag,
            (x0 + 175, y0 + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            timer_tag_color,
            2,
            cv2.LINE_AA,
        )

        # 4. Actionable Feedback Reasons (if ADJUST)
        current_y = y0 + 112
        if form_eval and form_eval.status == FormStatus.ADJUST and reasons_to_show:
            for r in reasons_to_show:
                # Truncate long reasons if necessary
                display_r = r if len(r) <= 36 else r[:33] + "..."
                cv2.putText(
                    frame,
                    f"> {display_r}",
                    (x0 + 14, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    self.COLOR_ADJUST,
                    1,
                    cv2.LINE_AA,
                )
                current_y += 20
        elif form_eval and form_eval.status == "CANDIDATE" and form_eval.reasons:
            cv2.putText(
                frame,
                f"> {form_eval.reasons[0]}",
                (x0 + 14, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                self.COLOR_CANDIDATE,
                1,
                cv2.LINE_AA,
            )
            current_y += 20

        # Subtle separator line
        cv2.line(
            frame,
            (x0 + 12, current_y),
            (x1 - 12, current_y),
            (50, 55, 65),
            1,
            cv2.LINE_AA,
        )
        current_y += 20

        # 5. Numerical Metrics
        if features is not None:
            knee_str = f"AVG KNEE: {features.avg_knee_angle:.1f}°"
            hip_str = f"AVG HIP:  {features.avg_hip_angle:.1f}°"
            torso_str = f"TORSO:    {features.torso_angle:.1f}°"
            vis_val = features.visibility
            vis_str = f"VISIBILITY: {vis_val:.2f}"
            vis_color = self.COLOR_TEXT_SECONDARY if vis_val >= 0.5 else self.COLOR_ADJUST
        else:
            knee_str = "AVG KNEE: --"
            hip_str = "AVG HIP:  --"
            torso_str = "TORSO:    --"
            vis_str = "VISIBILITY: 0.00"
            vis_color = self.COLOR_LOST

        cv2.putText(
            frame, knee_str, (x0 + 14, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, hip_str, (x0 + 195, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
        )
        current_y += 24

        cv2.putText(
            frame, torso_str, (x0 + 14, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, vis_str, (x0 + 195, current_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, vis_color, 1, cv2.LINE_AA
        )
        current_y += 24

        # 6. Keyboard shortcuts footer
        cv2.putText(
            frame,
            "[q]=quit   [r]=reset timer",
            (x0 + 14, current_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (140, 145, 155),
            1,
            cv2.LINE_AA,
        )
