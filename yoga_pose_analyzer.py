"""
AI Yoga Pose Analyzer - Live Webcam & Video Analysis Pipeline.

Supports automatic detection, debounced pose confirmation, multi-rule form checking,
hold timer, translucent HUD visualization, and structured CSV dataset logging for 6 key poses:
  1. Mountain Pose (Tadasana)
  2. Downward-Facing Dog (Adho Mukha Svanasana)
  3. Warrior I (Virabhadrasana I)
  4. Warrior II (Virabhadrasana II)
  5. Tree Pose (Vrksasana)
  6. Triangle Pose (Trikonasana)
"""
import argparse
import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core.landmarker import create_landmarker
from core.drawing import draw_pose_skeleton
from core.geometry import extract_yoga_features, YogaFeatures
from core.smoothing import YogaFeatureSmoother
from yoga.base_pose import BaseYogaPose, FormEvaluation, FormStatus
from yoga.classifier import YogaPoseClassifier
from yoga.timer import YogaHoldTimer
from yoga.hud import YogaHUD
from yoga.logger import YogaDatasetLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Fitness CV - Automatic Yoga Pose Recognition & Form Coach"
    )
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Input video source: webcam index (e.g. 0) or path to video file (e.g. video.mp4)",
    )
    parser.add_argument(
        "--model", "-m",
        default="pose_landmarker_lite.task",
        help="Path to MediaPipe pose landmarker task file",
    )
    parser.add_argument(
        "--csv", "-c",
        default="yoga_dataset.csv",
        help="Output CSV file path for logging dataset records",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="Optional path to export rendered video output (.mp4)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without GUI display",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Temporal moving average window size for feature smoothing (default: 5)",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=8,
        help="Consecutive frames required to confirm a candidate pose (default: 8)",
    )
    parser.add_argument(
        "--lost-frames",
        type=int,
        default=25,
        help="Consecutive lost frames before hold timer resets (default: 25)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input source
    source = int(args.source) if args.source.isdigit() else args.source

    print("==================================================")
    print("      AI FITNESS BACKEND - YOGA POSE ANALYZER      ")
    print("==================================================")
    print(f"Source: {source}")
    print(f"Model: {args.model}")
    print(f"CSV Dataset: {args.csv}")
    print(f"Temporal Smoothing Window: {args.smooth_window}")
    print(f"Pose Confirmation Frames: {args.confirm_frames}")
    print(f"Pose Loss Reset Frames: {args.lost_frames}")
    print("--------------------------------------------------")

    # Initialize MediaPipe Landmarker
    try:
        landmarker = create_landmarker(args.model)
    except Exception as e:
        print(f"[Error] Failed to initialize MediaPipe Landmarker: {e}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Optional video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (frame_w, frame_h))
        print(f"Saving output video to: {args.save_video}")

    # Core pipeline components
    smoother = YogaFeatureSmoother(window_size=args.smooth_window)
    classifier = YogaPoseClassifier(
        confirm_frames=args.confirm_frames,
        lost_frames=args.lost_frames,
    )
    timer = YogaHoldTimer()
    hud = YogaHUD()
    logger = YogaDatasetLogger(csv_path=args.csv, flush_every=30)

    frame_idx = 0
    prev_time = time.time()
    session_start = time.time()
    pose_hold_summary = {}

    print("Yoga Pose Analyzer running. Press [q] in window to quit, [r] to reset timer.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            dt = 1.0 / fps if isinstance(source, str) else (current_time - prev_time)
            prev_time = current_time

            # Format frame for MediaPipe Tasks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000.0 / fps)

            # Pose landmark extraction
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            smoothed_features: Optional[YogaFeatures] = None
            confirmed_pose: Optional[BaseYogaPose] = None
            lifecycle_state: str = YogaPoseClassifier.STATE_NOT_DETECTED
            form_eval: FormEvaluation = FormEvaluation(status=FormStatus.LOST, reasons=["No person detected"])

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                raw_landmarks = result.pose_landmarks[0]
                raw_world_landmarks = (
                    result.pose_world_landmarks[0]
                    if (hasattr(result, "pose_world_landmarks") and result.pose_world_landmarks and len(result.pose_world_landmarks) > 0)
                    else None
                )

                # 1. Extract raw bilateral geometry & angles (using 3D world coords when available)
                raw_features = extract_yoga_features(raw_landmarks, world_landmarks=raw_world_landmarks)

                # 2. Apply temporal smoothing
                smoothed_features = smoother.update(raw_features)

                # 3. Classify candidate and confirm debounced pose
                confirmed_pose, lifecycle_state, form_eval = classifier.update(smoothed_features)

                # 4. Update hold timer
                hold_seconds = timer.update(dt, confirmed_pose, form_eval)

                # 5. Draw skeleton on frame (with joint error highlighting)
                draw_pose_skeleton(
                    frame,
                    raw_landmarks,
                    highlight_joints=form_eval.error_joints,
                    min_visibility=0.5,
                )
            else:
                smoother.reset()
                confirmed_pose, lifecycle_state, form_eval = classifier.update(None)
                hold_seconds = timer.update(dt, None, form_eval)

            # Update session summary metrics
            if confirmed_pose is not None:
                pose_name = confirmed_pose.name
                pose_hold_summary[pose_name] = max(
                    pose_hold_summary.get(pose_name, 0.0), hold_seconds
                )

            # 6. Render translucent HUD Card Overlay
            hud.draw(
                frame=frame,
                confirmed_pose=confirmed_pose,
                lifecycle_state=lifecycle_state,
                form_eval=form_eval,
                hold_seconds=hold_seconds,
                timer_status_str=timer.timer_status_str,
                features=smoothed_features,
            )

            # 7. Log structured numerical data for ML training
            logger.log_frame(
                timestamp_ms=timestamp_ms,
                confirmed_pose=confirmed_pose,
                form_eval=form_eval,
                features=smoothed_features,
                hold_time=hold_seconds,
            )

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("AI Yoga Pose Analyzer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    timer.reset()
                    print("[Timer] Manually reset.")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.flush()
        landmarker.close()

    # Session Summary Report
    session_duration = time.time() - session_start
    print("\n==================================================")
    print("              YOGA SESSION SUMMARY                ")
    print("==================================================")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total session duration: {session_duration:.1f} s")
    print("Max Hold Duration by Pose:")
    if pose_hold_summary:
        for pose_name, max_hold in pose_hold_summary.items():
            print(f"  - {pose_name}: {max_hold:.1f} s")
    else:
        print("  (No confirmed yoga poses recorded)")
    print(f"Dataset CSV saved to: {args.csv}")
    print("==================================================")


if __name__ == "__main__":
    main()
