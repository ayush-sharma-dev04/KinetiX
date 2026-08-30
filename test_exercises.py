"""
Comprehensive Unit Test Suite for Fitness Exercise Analyzers:
  - pushup.py
  - situp.py
  - crunch.py
  - lunge.py
  - surya_namaskar.py
"""
import unittest
import numpy as np

# 1. Push-up imports
from pushup import (
    PushUpRepCounter,
    classify_state as classify_pushup_state,
    evaluate_pushup_form,
    DebouncedState as PushUpDebouncedState,
    MovingAverage as PushUpMovingAverage,
    calculate_angle as pushup_calculate_angle,
)

# 2. Sit-up imports
from situp import (
    SitUpRepCounter,
    classify_state as classify_situp_state,
    evaluate_situp_form,
    calculate_torso_angle_to_horizontal as situp_torso_angle,
    DebouncedState as SitUpDebouncedState,
)

# 3. Crunch imports
from crunch import (
    CrunchRepCounter,
    classify_state as classify_crunch_state,
    evaluate_crunch_form,
    DebouncedState as CrunchDebouncedState,
)

# 4. Lunge imports
from lunge import (
    LungeRepCounter,
    classify_state as classify_lunge_state,
    evaluate_lunge_form,
    determine_front_leg,
    DebouncedState as LungeDebouncedState,
)

# 5. Surya Namaskar imports
from surya_namaskar import (
    SuryaNamaskarTracker,
    check_pranamasana,
    check_hasta_uttanasana,
    check_padahastasana,
    check_ashwa_sanchalanasana,
    check_plank,
    check_ashtanga_namaskara,
    check_bhujangasana,
    check_parvatasana,
)


class MockLandmark:
    """Mock landmark matching MediaPipe NormalizedLandmark interface."""
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.95):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class TestPushUp(unittest.TestCase):
    def test_full_rep_lifecycle(self):
        counter = PushUpRepCounter()
        self.assertEqual(counter.state, "UP")
        self.assertEqual(counter.rep_count, 0)

        # UP -> DOWN -> BOTTOM -> DOWN -> UP
        counter.process_transition("DOWN")
        self.assertEqual(counter.state, "DOWN")
        self.assertEqual(counter.rep_count, 0)

        counter.process_transition("BOTTOM")
        self.assertEqual(counter.state, "BOTTOM")
        self.assertTrue(counter._reached_bottom)

        counter.process_transition("DOWN")
        self.assertEqual(counter.state, "DOWN")

        counter.process_transition("UP")
        self.assertEqual(counter.state, "UP")
        self.assertEqual(counter.rep_count, 1)
        self.assertEqual(counter.shallow_count, 0)

    def test_shallow_pushup_not_counted(self):
        counter = PushUpRepCounter()
        # UP -> DOWN -> UP (never reached BOTTOM)
        counter.process_transition("DOWN")
        counter.process_transition("UP")
        self.assertEqual(counter.rep_count, 0)
        self.assertEqual(counter.shallow_count, 1)

    def test_invalid_direct_transition_blocked(self):
        counter = PushUpRepCounter()
        # Cannot jump directly UP -> BOTTOM
        counter.process_transition("BOTTOM")
        self.assertEqual(counter.state, "UP")
        self.assertEqual(counter.rep_count, 0)

    def test_pushup_classification(self):
        # From UP: angle < 160 drops to DOWN
        self.assertEqual(classify_pushup_state(150.0, "UP"), "DOWN")
        self.assertEqual(classify_pushup_state(170.0, "UP"), "UP")

        # From DOWN: <= 95 reaches BOTTOM, >= 160 reaches UP
        self.assertEqual(classify_pushup_state(85.0, "DOWN"), "BOTTOM")
        self.assertEqual(classify_pushup_state(165.0, "DOWN"), "UP")
        self.assertEqual(classify_pushup_state(120.0, "DOWN"), "DOWN")

        # From BOTTOM: > 95 goes to DOWN
        self.assertEqual(classify_pushup_state(110.0, "BOTTOM"), "DOWN")
        self.assertEqual(classify_pushup_state(90.0, "BOTTOM"), "BOTTOM")

    def test_pushup_form_evaluation(self):
        # Good form
        status, feedback = evaluate_pushup_form(
            left_elbow=90.0, right_elbow=90.0,
            left_hip=175.0, right_hip=175.0,
            shoulder_tilt=2.0, neck_angle=90.0,
            current_state="BOTTOM"
        )
        self.assertEqual(status, "GOOD")

        # Asymmetric elbow push
        status, feedback = evaluate_pushup_form(
            left_elbow=90.0, right_elbow=115.0,
            left_hip=175.0, right_hip=175.0,
            shoulder_tilt=2.0, neck_angle=90.0,
            current_state="BOTTOM"
        )
        self.assertEqual(status, "FORM ISSUE")
        self.assertIn("Uneven", feedback)

        # Sagging hips
        status, feedback = evaluate_pushup_form(
            left_elbow=90.0, right_elbow=90.0,
            left_hip=140.0, right_hip=140.0,
            shoulder_tilt=2.0, neck_angle=90.0,
            current_state="DOWN"
        )
        self.assertEqual(status, "FORM ISSUE")
        self.assertIn("sagging", feedback)

    def test_debounced_state(self):
        debouncer = PushUpDebouncedState(debounce_frames=3, initial_state="UP")
        # Frame 1: candidate DOWN
        state, changed = debouncer.update("DOWN")
        self.assertEqual(state, "UP")
        self.assertFalse(changed)

        # Frame 2: candidate DOWN
        state, changed = debouncer.update("DOWN")
        self.assertEqual(state, "UP")
        self.assertFalse(changed)

        # Frame 3: confirmed DOWN
        state, changed = debouncer.update("DOWN")
        self.assertEqual(state, "DOWN")
        self.assertTrue(changed)


class TestSitUp(unittest.TestCase):
    def test_full_situp_cycle(self):
        counter = SitUpRepCounter()
        self.assertEqual(counter.state, "DOWN")

        # DOWN -> ASCENDING -> UP -> DESCENDING -> DOWN
        counter.process_transition("ASCENDING")
        self.assertEqual(counter.state, "ASCENDING")

        counter.process_transition("UP")
        self.assertEqual(counter.state, "UP")
        self.assertTrue(counter._reached_up)

        counter.process_transition("DESCENDING")
        self.assertEqual(counter.state, "DESCENDING")

        counter.process_transition("DOWN")
        self.assertEqual(counter.state, "DOWN")
        self.assertEqual(counter.rep_count, 1)
        self.assertEqual(counter.shallow_count, 0)

    def test_incomplete_situp_not_counted(self):
        counter = SitUpRepCounter()
        # DOWN -> ASCENDING -> DOWN (never reached UP)
        counter.process_transition("ASCENDING")
        counter.process_transition("DOWN")
        self.assertEqual(counter.rep_count, 0)
        self.assertEqual(counter.shallow_count, 1)

    def test_situp_classification(self):
        # From DOWN: torso > 20 -> ASCENDING
        self.assertEqual(classify_situp_state(10.0, 120.0, "DOWN"), "DOWN")
        self.assertEqual(classify_situp_state(35.0, 100.0, "DOWN"), "ASCENDING")

        # From ASCENDING: torso >= 60 -> UP
        self.assertEqual(classify_situp_state(65.0, 55.0, "ASCENDING"), "UP")
        self.assertEqual(classify_situp_state(15.0, 120.0, "ASCENDING"), "DOWN")

        # From UP: torso < 55 -> DESCENDING
        self.assertEqual(classify_situp_state(50.0, 65.0, "UP"), "DESCENDING")

    def test_situp_form_evaluation(self):
        # Good form
        status, feedback = evaluate_situp_form(
            torso_angle=65.0, hip_tilt=2.0, shoulder_tilt=3.0, neck_angle=80.0, current_state="UP"
        )
        self.assertEqual(status, "GOOD")

        # Twisting torso
        status, feedback = evaluate_situp_form(
            torso_angle=40.0, hip_tilt=20.0, shoulder_tilt=18.0, neck_angle=80.0, current_state="ASCENDING"
        )
        self.assertEqual(status, "FORM ISSUE")
        self.assertIn("twisting", feedback)

        # Pulling neck
        status, feedback = evaluate_situp_form(
            torso_angle=40.0, hip_tilt=2.0, shoulder_tilt=3.0, neck_angle=30.0, current_state="ASCENDING"
        )
        self.assertEqual(status, "FORM ISSUE")
        self.assertIn("pull neck", feedback)


class TestCrunch(unittest.TestCase):
    def test_full_crunch_cycle(self):
        counter = CrunchRepCounter()
        self.assertEqual(counter.state, "DOWN")

        # DOWN -> CRUNCHING -> UP -> RELEASING -> DOWN
        counter.process_transition("CRUNCHING")
        counter.process_transition("UP")
        self.assertTrue(counter._reached_up)
        counter.process_transition("RELEASING")
        counter.process_transition("DOWN")

        self.assertEqual(counter.rep_count, 1)
        self.assertEqual(counter.shallow_count, 0)

    def test_shallow_crunch_not_counted(self):
        counter = CrunchRepCounter()
        counter.process_transition("CRUNCHING")
        counter.process_transition("DOWN")
        self.assertEqual(counter.rep_count, 0)
        self.assertEqual(counter.shallow_count, 1)

    def test_crunch_classification(self):
        self.assertEqual(classify_crunch_state(5.0, "DOWN"), "DOWN")
        self.assertEqual(classify_crunch_state(14.0, "DOWN"), "CRUNCHING")
        self.assertEqual(classify_crunch_state(25.0, "CRUNCHING"), "UP")
        self.assertEqual(classify_crunch_state(18.0, "UP"), "RELEASING")
        self.assertEqual(classify_crunch_state(5.0, "RELEASING"), "DOWN")

    def test_crunch_form_overlift(self):
        # Lifting too high (sit-up instead of crunch)
        status, feedback = evaluate_crunch_form(
            torso_angle=55.0, hip_tilt=2.0, shoulder_tilt=2.0, neck_angle=80.0, current_state="UP"
        )
        self.assertEqual(status, "FORM ISSUE")
        self.assertIn("too high", feedback)


class TestLunge(unittest.TestCase):
    def test_lunge_rep_lifecycle_alternating(self):
        counter = LungeRepCounter()
        self.assertEqual(counter.state, "UP")

        # Rep 1: Left leg lead
        counter.set_lead_leg("LEFT")
        counter.process_transition("DOWN", lead_leg="LEFT")
        counter.process_transition("BOTTOM", lead_leg="LEFT")
        counter.process_transition("DOWN", lead_leg="LEFT")
        counter.process_transition("UP")

        self.assertEqual(counter.rep_count, 1)
        self.assertEqual(counter.left_reps, 1)
        self.assertEqual(counter.right_reps, 0)

        # Rep 2: Right leg lead
        counter.set_lead_leg("RIGHT")
        counter.process_transition("DOWN", lead_leg="RIGHT")
        counter.process_transition("BOTTOM", lead_leg="RIGHT")
        counter.process_transition("DOWN", lead_leg="RIGHT")
        counter.process_transition("UP")

        self.assertEqual(counter.rep_count, 2)
        self.assertEqual(counter.left_reps, 1)
        self.assertEqual(counter.right_reps, 1)

    def test_lunge_classification(self):
        # Straight legs -> UP
        self.assertEqual(classify_lunge_state(170.0, 170.0, "UP"), "UP")
        # Front knee bends -> DOWN
        self.assertEqual(classify_lunge_state(130.0, 140.0, "UP"), "DOWN")
        # Depth reached (front <= 100, back <= 115) -> BOTTOM
        self.assertEqual(classify_lunge_state(90.0, 100.0, "DOWN"), "BOTTOM")
        # Ascending from BOTTOM -> DOWN
        self.assertEqual(classify_lunge_state(120.0, 130.0, "BOTTOM"), "DOWN")
        # Standing back UP -> UP
        self.assertEqual(classify_lunge_state(165.0, 165.0, "DOWN"), "UP")

    def test_lunge_lead_leg_detection(self):
        # Create mock pose where Left ankle is forward (larger x relative to hips in facing direction)
        pose = [MockLandmark(0.5, 0.5) for _ in range(33)]
        pose[0] = MockLandmark(0.7, 0.2)   # Nose facing right (x=0.7 > hip x=0.5)
        pose[11] = MockLandmark(0.48, 0.3) # Left shoulder
        pose[12] = MockLandmark(0.52, 0.3) # Right shoulder
        pose[23] = MockLandmark(0.48, 0.5) # Left hip
        pose[24] = MockLandmark(0.52, 0.5) # Right hip
        pose[27] = MockLandmark(0.75, 0.9) # Left ankle forward
        pose[28] = MockLandmark(0.30, 0.9) # Right ankle behind

        lead = determine_front_leg(pose)
        self.assertEqual(lead, "LEFT")

        # Now swap: Right ankle forward
        pose[27] = MockLandmark(0.30, 0.9) # Left ankle behind
        pose[28] = MockLandmark(0.75, 0.9) # Right ankle forward
        lead = determine_front_leg(pose)
        self.assertEqual(lead, "RIGHT")


class TestSuryaNamaskar(unittest.TestCase):
    def test_12_step_sequence_full_cycle(self):
        tracker = SuryaNamaskarTracker(debounce_frames=2)
        self.assertEqual(tracker.step_idx, 1)
        self.assertEqual(tracker.cycle_count, 0)
        self.assertEqual(tracker.get_expected_step_name(), "Pranamasana")

        # Step 1: Pranamasana
        pranamasana_feat = {
            "torso_angle": 5.0, "avg_knee_angle": 175.0, "avg_elbow_angle": 50.0,
            "left_wrist_y": 0.35, "left_shoulder_y": 0.30,
            "shoulder_symmetry": 0.02, "hip_symmetry": 0.02,
        }
        tracker.evaluate_frame(pranamasana_feat)
        tracker.evaluate_frame(pranamasana_feat) # 2nd frame confirms step 1 -> step 2
        self.assertEqual(tracker.step_idx, 2)
        self.assertEqual(tracker.get_expected_step_name(), "Hasta Uttanasana")

        # Step 2: Hasta Uttanasana
        hasta_feat = {
            "torso_angle": 15.0, "avg_knee_angle": 175.0, "avg_elbow_angle": 170.0,
            "avg_shoulder_angle": 170.0, "left_wrist_y": 0.10, "right_wrist_y": 0.10,
            "left_shoulder_y": 0.30, "right_shoulder_y": 0.30,
        }
        tracker.evaluate_frame(hasta_feat)
        tracker.evaluate_frame(hasta_feat)
        self.assertEqual(tracker.step_idx, 3)
        self.assertEqual(tracker.get_expected_step_name(), "Padahastasana")

        # Step 3: Padahastasana
        pada_feat = {
            "torso_angle": 75.0, "avg_knee_angle": 160.0,
            "left_wrist_y": 0.85, "hip_mid_y": 0.50,
            "nose_y": 0.60, "shoulder_mid_y": 0.55,
        }
        tracker.evaluate_frame(pada_feat)
        tracker.evaluate_frame(pada_feat)
        self.assertEqual(tracker.step_idx, 4)
        self.assertEqual(tracker.get_expected_step_name(), "Ashwa Sanchalanasana")

        # Step 4: Ashwa Sanchalanasana (Left lead)
        ashwa_left_feat = {
            "left_knee_angle": 85.0, "right_knee_angle": 170.0,
            "shoulder_mid_y": 0.45,
        }
        tracker.evaluate_frame(ashwa_left_feat)
        tracker.evaluate_frame(ashwa_left_feat)
        self.assertEqual(tracker.step_idx, 5)
        self.assertEqual(tracker.lead_leg_step4, "LEFT")
        self.assertEqual(tracker.get_expected_step_name(), "Utthita Chaturanga Dandasana")

        # Step 5: Plank
        plank_feat = {
            "torso_angle": 80.0, "avg_hip_angle": 170.0, "avg_knee_angle": 170.0,
            "avg_elbow_angle": 170.0, "shoulder_symmetry": 0.02,
        }
        tracker.evaluate_frame(plank_feat)
        tracker.evaluate_frame(plank_feat)
        self.assertEqual(tracker.step_idx, 6)
        self.assertEqual(tracker.get_expected_step_name(), "Ashtanga Namaskara")

        # Step 6: Ashtanga Namaskara
        ashtanga_feat = {
            "avg_knee_angle": 110.0, "avg_elbow_angle": 45.0,
            "shoulder_mid_y": 0.65, "hip_mid_y": 0.58, "torso_angle": 75.0,
        }
        tracker.evaluate_frame(ashtanga_feat)
        tracker.evaluate_frame(ashtanga_feat)
        self.assertEqual(tracker.step_idx, 7)
        self.assertEqual(tracker.get_expected_step_name(), "Bhujangasana")

        # Step 7: Bhujangasana
        cobra_feat = {
            "shoulder_mid_y": 0.40, "hip_mid_y": 0.65, "avg_knee_angle": 175.0,
            "avg_elbow_angle": 140.0, "torso_angle": 35.0,
        }
        tracker.evaluate_frame(cobra_feat)
        tracker.evaluate_frame(cobra_feat)
        self.assertEqual(tracker.step_idx, 8)
        self.assertEqual(tracker.get_expected_step_name(), "Parvatasana")

        # Step 8: Parvatasana
        parvat_feat = {
            "hip_mid_y": 0.30, "shoulder_mid_y": 0.55, "ankle_mid_y": 0.85,
            "avg_hip_angle": 75.0, "avg_knee_angle": 165.0, "avg_elbow_angle": 170.0,
        }
        tracker.evaluate_frame(parvat_feat)
        tracker.evaluate_frame(parvat_feat)
        self.assertEqual(tracker.step_idx, 9)
        self.assertEqual(tracker.get_expected_step_name(), "Ashwa Sanchalanasana")

        # Step 9: Ashwa Sanchalanasana with OPPOSITE leg (RIGHT lead)
        # First test: using wrong leg (LEFT) fails for Step 9
        match, _, feedback = tracker.evaluate_frame(ashwa_left_feat)
        self.assertFalse(match)
        self.assertIn("opposite", feedback.lower())

        # Correct leg (RIGHT lead) succeeds for Step 9
        ashwa_right_feat = {
            "left_knee_angle": 170.0, "right_knee_angle": 85.0,
            "shoulder_mid_y": 0.45,
        }
        tracker.evaluate_frame(ashwa_right_feat)
        tracker.evaluate_frame(ashwa_right_feat)
        self.assertEqual(tracker.step_idx, 10)
        self.assertEqual(tracker.get_expected_step_name(), "Padahastasana")

        # Step 10: Padahastasana
        tracker.evaluate_frame(pada_feat)
        tracker.evaluate_frame(pada_feat)
        self.assertEqual(tracker.step_idx, 11)
        self.assertEqual(tracker.get_expected_step_name(), "Hasta Uttanasana")

        # Step 11: Hasta Uttanasana
        tracker.evaluate_frame(hasta_feat)
        tracker.evaluate_frame(hasta_feat)
        self.assertEqual(tracker.step_idx, 12)
        self.assertEqual(tracker.get_expected_step_name(), "Pranamasana")

        # Step 12: Pranamasana (Cycle completion)
        tracker.evaluate_frame(pranamasana_feat)
        tracker.evaluate_frame(pranamasana_feat)
        self.assertEqual(tracker.cycle_count, 1)
        self.assertEqual(tracker.step_idx, 1) # Resets to step 1 for next cycle


if __name__ == "__main__":
    unittest.main()
