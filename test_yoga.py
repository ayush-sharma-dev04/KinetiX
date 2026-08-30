"""
Comprehensive Test Suite for AI Yoga Pose Analyzer & Biomechanical Rules.
"""
import os
import unittest
import numpy as np

from core.landmarks import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)
from core.geometry import (
    calculate_angle, calculate_angle_2d, calculate_angle_3d,
    calculate_torso_angle, calculate_distance, calculate_distance_3d,
    calculate_nose_torso_offset, calculate_forward_collapse,
    extract_yoga_features, YogaFeatures,
)
from core.smoothing import MovingAverage, YogaFeatureSmoother
from core.drawing import draw_pose_skeleton
from yoga.base_pose import FormStatus, FormEvaluation
from yoga.poses.tadasana import TadasanaPose
from yoga.poses.warrior_ii import WarriorIIPose
from yoga.poses.warrior_i import WarriorIPose
from yoga.poses.tree_pose import TreePose
from yoga.poses.triangle import TrianglePose
from yoga.poses.downward_dog import DownwardDogPose
from yoga.classifier import YogaPoseClassifier
from yoga.timer import YogaHoldTimer
from yoga.hud import YogaHUD
from yoga.logger import YogaDatasetLogger


class MockLandmark:
    """Mock landmark matching MediaPipe NormalizedLandmark / Landmark interface."""
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.95):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def create_mock_pose_landmarks(
    sh_l=(0.4, 0.3), sh_r=(0.6, 0.3),
    el_l=(0.35, 0.45), el_r=(0.65, 0.45),
    wr_l=(0.35, 0.6), wr_r=(0.65, 0.6),
    hp_l=(0.45, 0.5), hp_r=(0.55, 0.5),
    kn_l=(0.45, 0.7), kn_r=(0.55, 0.7),
    ak_l=(0.45, 0.9), ak_r=(0.55, 0.9),
    hl_l=None, hl_r=None,
    fi_l=None, fi_r=None,
    nose=None,
    z: float = 0.0,
    visibility: float = 0.95,
):
    """Generates 33 mock landmarks for testing with full body points."""
    default_nose_x = (sh_l[0] + sh_r[0]) / 2.0
    default_nose_y = min(sh_l[1], sh_r[1]) - 0.15
    nose_pt = nose if nose is not None else (default_nose_x, default_nose_y)

    hl_l_pt = hl_l if hl_l is not None else (ak_l[0], ak_l[1] + 0.02)
    hl_r_pt = hl_r if hl_r is not None else (ak_r[0], ak_r[1] + 0.02)
    fi_l_pt = fi_l if fi_l is not None else (ak_l[0] - 0.04, ak_l[1] + 0.02)
    fi_r_pt = fi_r if fi_r is not None else (ak_r[0] + 0.04, ak_r[1] + 0.02)

    landmarks = [MockLandmark(default_nose_x, default_nose_y, z=z, visibility=visibility) for _ in range(33)]
    landmarks[NOSE] = MockLandmark(nose_pt[0], nose_pt[1], z=z, visibility=visibility)
    landmarks[LEFT_SHOULDER] = MockLandmark(sh_l[0], sh_l[1], z=z, visibility=visibility)
    landmarks[RIGHT_SHOULDER] = MockLandmark(sh_r[0], sh_r[1], z=z, visibility=visibility)
    landmarks[LEFT_ELBOW] = MockLandmark(el_l[0], el_l[1], z=z, visibility=visibility)
    landmarks[RIGHT_ELBOW] = MockLandmark(el_r[0], el_r[1], z=z, visibility=visibility)
    landmarks[LEFT_WRIST] = MockLandmark(wr_l[0], wr_l[1], z=z, visibility=visibility)
    landmarks[RIGHT_WRIST] = MockLandmark(wr_r[0], wr_r[1], z=z, visibility=visibility)
    landmarks[LEFT_HIP] = MockLandmark(hp_l[0], hp_l[1], z=z, visibility=visibility)
    landmarks[RIGHT_HIP] = MockLandmark(hp_r[0], hp_r[1], z=z, visibility=visibility)
    landmarks[LEFT_KNEE] = MockLandmark(kn_l[0], kn_l[1], z=z, visibility=visibility)
    landmarks[RIGHT_KNEE] = MockLandmark(kn_r[0], kn_r[1], z=z, visibility=visibility)
    landmarks[LEFT_ANKLE] = MockLandmark(ak_l[0], ak_l[1], z=z, visibility=visibility)
    landmarks[RIGHT_ANKLE] = MockLandmark(ak_r[0], ak_r[1], z=z, visibility=visibility)
    landmarks[LEFT_HEEL] = MockLandmark(hl_l_pt[0], hl_l_pt[1], z=z, visibility=visibility)
    landmarks[RIGHT_HEEL] = MockLandmark(hl_r_pt[0], hl_r_pt[1], z=z, visibility=visibility)
    landmarks[LEFT_FOOT_INDEX] = MockLandmark(fi_l_pt[0], fi_l_pt[1], z=z, visibility=visibility)
    landmarks[RIGHT_FOOT_INDEX] = MockLandmark(fi_r_pt[0], fi_r_pt[1], z=z, visibility=visibility)
    return landmarks


class TestGeometryAndSmoothing(unittest.TestCase):
    def test_calculate_angle_right_angle(self):
        a = (0.0, 1.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=3)

    def test_calculate_angle_straight_line(self):
        a = (0.0, -1.0)
        b = (0.0, 0.0)
        c = (0.0, 1.0)
        angle = calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=3)

    def test_calculate_angle_3d(self):
        # Right angle in 3D (X and Z axes)
        a = (1.0, 0.0, 0.0)
        b = (0.0, 0.0, 0.0)
        c = (0.0, 0.0, 1.0)
        angle = calculate_angle_3d(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=3)

    def test_calculate_torso_angle_vertical(self):
        sh_l = (0.4, 0.3)
        sh_r = (0.6, 0.3)
        hp_l = (0.4, 0.7)
        hp_r = (0.6, 0.7)
        torso = calculate_torso_angle(sh_l, sh_r, hp_l, hp_r)
        self.assertAlmostEqual(torso, 0.0, delta=1.0)

    def test_calculate_nose_torso_offset(self):
        sh_l = (0.4, 0.3)
        sh_r = (0.6, 0.3)
        hp_l = (0.4, 0.7)
        hp_r = (0.6, 0.7)
        # Perfectly aligned nose at x=0.5
        nose_center = (0.5, 0.15)
        offset_center = calculate_nose_torso_offset(nose_center, sh_l, sh_r, hp_l, hp_r)
        self.assertAlmostEqual(offset_center, 0.0, delta=0.01)

        # Off-center nose at x=0.7
        nose_skew = (0.7, 0.15)
        offset_skew = calculate_nose_torso_offset(nose_skew, sh_l, sh_r, hp_l, hp_r)
        self.assertGreater(offset_skew, 0.20)

    def test_moving_average_filter(self):
        ma = MovingAverage(window_size=3)
        self.assertEqual(ma.update(10.0), 10.0)
        self.assertEqual(ma.update(20.0), 15.0)
        self.assertEqual(ma.update(30.0), 20.0)
        self.assertEqual(ma.update(40.0), 30.0)

    def test_world_landmarks_extraction(self):
        norm_lm = create_mock_pose_landmarks()
        world_lm = create_mock_pose_landmarks(z=0.1)
        features = extract_yoga_features(norm_lm, world_landmarks=world_lm)
        self.assertTrue(features.has_world_landmarks)
        self.assertIsNotNone(features.world_landmarks)


class TestYogaPoses(unittest.TestCase):
    def test_tadasana_correct_and_adjust(self):
        pose = TadasanaPose()
        # Clean upright posture, feet close, straight legs, aligned head
        lm = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            el_l=(0.43, 0.45), el_r=(0.57, 0.45),
            wr_l=(0.43, 0.6), wr_r=(0.57, 0.6),
            hp_l=(0.46, 0.5), hp_r=(0.54, 0.5),
            kn_l=(0.46, 0.7), kn_r=(0.54, 0.7),
            ak_l=(0.46, 0.9), ak_r=(0.54, 0.9),
            nose=(0.50, 0.15),
        )
        f = extract_yoga_features(lm)
        is_cand, conf = pose.is_candidate(f)
        self.assertTrue(is_cand)
        self.assertGreaterEqual(conf, 0.5)

        evaluation = pose.evaluate_form(f)
        self.assertEqual(evaluation.status, FormStatus.CORRECT)
        self.assertEqual(len(evaluation.reasons), 0)

        # Bent knees form defect
        lm_bent = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            hp_l=(0.46, 0.5), hp_r=(0.54, 0.5),
            kn_l=(0.40, 0.7), kn_r=(0.54, 0.7),
            ak_l=(0.46, 0.9), ak_r=(0.54, 0.9),
        )
        f_bent = extract_yoga_features(lm_bent)
        eval_bent = pose.evaluate_form(f_bent)
        self.assertEqual(eval_bent.status, FormStatus.ADJUST)
        self.assertTrue(any("KNEE" in r for r in eval_bent.reasons))

        # Head misalignment form defect
        lm_head = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            hp_l=(0.46, 0.5), hp_r=(0.54, 0.5),
            kn_l=(0.46, 0.7), kn_r=(0.54, 0.7),
            ak_l=(0.46, 0.9), ak_r=(0.54, 0.9),
            nose=(0.65, 0.15), # skewed head
        )
        f_head = extract_yoga_features(lm_head)
        eval_head = pose.evaluate_form(f_head)
        self.assertEqual(eval_head.status, FormStatus.ADJUST)
        self.assertTrue(any("HEAD" in r for r in eval_head.reasons))

    def test_warrior_ii_correct_and_adjust(self):
        pose = WarriorIIPose()
        # Warrior II (Left knee bent 90°, Right knee straight 180°, arms horizontal)
        lm = create_mock_pose_landmarks(
            sh_l=(0.45, 0.35), sh_r=(0.55, 0.35),
            el_l=(0.25, 0.35), el_r=(0.75, 0.35),
            wr_l=(0.10, 0.35), wr_r=(0.90, 0.35),
            hp_l=(0.45, 0.65), hp_r=(0.55, 0.50),
            kn_l=(0.30, 0.65), kn_r=(0.75, 0.70),
            ak_l=(0.30, 0.90), ak_r=(0.90, 0.90),
            fi_l=(0.25, 0.92), fi_r=(0.95, 0.92),
            nose=(0.50, 0.20),
        )
        f = extract_yoga_features(lm)
        is_cand, conf = pose.is_candidate(f)
        self.assertTrue(is_cand)
        self.assertGreaterEqual(conf, 0.8)

        evaluation = pose.evaluate_form(f)
        self.assertEqual(evaluation.status, FormStatus.CORRECT)
        self.assertEqual(evaluation.metrics["orientation"], "LEFT_FORWARD")

        # Adjust case: Dropped arms
        lm_dropped = create_mock_pose_landmarks(
            sh_l=(0.45, 0.35), sh_r=(0.55, 0.35),
            el_l=(0.35, 0.45), el_r=(0.65, 0.45),
            wr_l=(0.30, 0.60), wr_r=(0.70, 0.60),
            hp_l=(0.45, 0.65), hp_r=(0.55, 0.50),
            kn_l=(0.30, 0.65), kn_r=(0.75, 0.70),
            ak_l=(0.30, 0.90), ak_r=(0.90, 0.90),
        )
        f_dropped = extract_yoga_features(lm_dropped)
        eval_dropped = pose.evaluate_form(f_dropped)
        self.assertEqual(eval_dropped.status, FormStatus.ADJUST)
        self.assertTrue(any("ARM" in r for r in eval_dropped.reasons))

    def test_warrior_i_correct_and_overshoot(self):
        pose = WarriorIPose()
        # Warrior I: Wide stance, left knee bent ~90°, right knee straight, arms overhead
        lm = create_mock_pose_landmarks(
            sh_l=(0.45, 0.35), sh_r=(0.55, 0.35),
            el_l=(0.45, 0.20), el_r=(0.55, 0.20),
            wr_l=(0.45, 0.05), wr_r=(0.55, 0.05),
            hp_l=(0.45, 0.65), hp_r=(0.55, 0.55),
            kn_l=(0.30, 0.65), kn_r=(0.75, 0.72),
            ak_l=(0.30, 0.90), ak_r=(0.85, 0.90),
            fi_l=(0.22, 0.92), fi_r=(0.88, 0.92),
            nose=(0.50, 0.20),
        )
        f = extract_yoga_features(lm)
        is_cand, conf = pose.is_candidate(f)
        self.assertTrue(is_cand)
        evaluation = pose.evaluate_form(f)
        self.assertEqual(evaluation.status, FormStatus.CORRECT)
        self.assertEqual(evaluation.metrics["orientation"], "LEFT_FORWARD")

        # Front knee overshooting ankle
        lm_overshoot = create_mock_pose_landmarks(
            sh_l=(0.45, 0.35), sh_r=(0.55, 0.35),
            el_l=(0.45, 0.20), el_r=(0.55, 0.20),
            wr_l=(0.45, 0.05), wr_r=(0.55, 0.05),
            hp_l=(0.45, 0.65), hp_r=(0.55, 0.55),
            kn_l=(0.18, 0.65), kn_r=(0.75, 0.72), # knee far forward
            ak_l=(0.30, 0.90), ak_r=(0.85, 0.90),
            fi_l=(0.22, 0.92), fi_r=(0.88, 0.92),
        )
        f_overshoot = extract_yoga_features(lm_overshoot)
        eval_overshoot = pose.evaluate_form(f_overshoot)
        self.assertEqual(eval_overshoot.status, FormStatus.ADJUST)
        self.assertTrue(any("OVERSHOOTING" in r for r in eval_overshoot.reasons))

    def test_tree_pose_correct_and_knee_pressure(self):
        pose = TreePose()
        # Tree Pose: Standing on right leg, left foot against right upper inner calf/thigh
        lm = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            el_l=(0.42, 0.45), el_r=(0.58, 0.45),
            wr_l=(0.50, 0.40), wr_r=(0.50, 0.40),
            hp_l=(0.45, 0.5), hp_r=(0.55, 0.5),
            kn_l=(0.30, 0.65), kn_r=(0.55, 0.70),
            ak_l=(0.50, 0.58), ak_r=(0.55, 0.90), # foot placed at thigh above knee
            nose=(0.50, 0.15),
        )
        f = extract_yoga_features(lm)
        is_cand, conf = pose.is_candidate(f)
        self.assertTrue(is_cand)
        evaluation = pose.evaluate_form(f)
        self.assertEqual(evaluation.status, FormStatus.CORRECT)
        self.assertEqual(evaluation.metrics["support_leg"], "RIGHT_LEG_STANDING")

        # Foot placed directly on knee joint
        lm_on_knee = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            hp_l=(0.45, 0.5), hp_r=(0.55, 0.5),
            kn_l=(0.30, 0.70), kn_r=(0.55, 0.70),
            ak_l=(0.54, 0.70), ak_r=(0.55, 0.90), # foot directly on support knee (y=0.70)
        )
        f_on_knee = extract_yoga_features(lm_on_knee)
        eval_on_knee = pose.evaluate_form(f_on_knee)
        self.assertEqual(eval_on_knee.status, FormStatus.ADJUST)
        self.assertTrue(any("KNEE JOINT" in r for r in eval_on_knee.reasons))

    def test_triangle_pose_correct_and_forward_collapse(self):
        pose = TrianglePose()
        # Triangle pose: Wide stance, both legs straight, torso laterally tilted to left
        lm = create_mock_pose_landmarks(
            sh_l=(0.28, 0.45), sh_r=(0.42, 0.28),
            el_l=(0.22, 0.62), el_r=(0.48, 0.16),
            wr_l=(0.18, 0.80), wr_r=(0.55, 0.05),
            hp_l=(0.44, 0.55), hp_r=(0.56, 0.55),
            kn_l=(0.30, 0.72), kn_r=(0.70, 0.72),
            ak_l=(0.20, 0.90), ak_r=(0.80, 0.90),
            nose=(0.32, 0.35),
        )
        f = extract_yoga_features(lm)
        is_cand, conf = pose.is_candidate(f)
        self.assertTrue(is_cand)
        evaluation = pose.evaluate_form(f)
        self.assertEqual(evaluation.status, FormStatus.CORRECT)
        self.assertEqual(evaluation.metrics["side"], "LEFT_SIDE_TILT")

        # Forward chest collapse (3D world landmark forward deviation)
        world_collapse = create_mock_pose_landmarks(
            sh_l=(0.28, 0.45), sh_r=(0.42, 0.28),
            hp_l=(0.44, 0.55), hp_r=(0.56, 0.55),
            kn_l=(0.30, 0.72), kn_r=(0.70, 0.72),
            ak_l=(0.20, 0.90), ak_r=(0.80, 0.90),
        )
        # Introduce significant z-offset at shoulders relative to hips
        world_collapse[LEFT_SHOULDER].z = 0.35
        world_collapse[RIGHT_SHOULDER].z = 0.35
        world_collapse[LEFT_HIP].z = 0.0
        world_collapse[RIGHT_HIP].z = 0.0

        f_collapse = extract_yoga_features(lm, world_landmarks=world_collapse)
        eval_collapse = pose.evaluate_form(f_collapse)
        self.assertEqual(eval_collapse.status, FormStatus.ADJUST)
        self.assertTrue(any("FORWARD" in r for r in eval_collapse.reasons))

    def test_downward_dog_correct_and_craned_neck(self):
        pose = DownwardDogPose()
        # Downward Dog: Inverted V, hips elevated (y=0.25), hands on ground (y=0.75), feet on ground (y=0.75)
        lm = create_mock_pose_landmarks(
            sh_l=(0.32, 0.55), sh_r=(0.32, 0.55),
            el_l=(0.25, 0.65), el_r=(0.25, 0.65),
            wr_l=(0.20, 0.75), wr_r=(0.20, 0.75),
            hp_l=(0.50, 0.25), hp_r=(0.50, 0.25),
            kn_l=(0.65, 0.50), kn_r=(0.65, 0.50),
            ak_l=(0.80, 0.75), ak_r=(0.80, 0.75),
            fi_l=(0.76, 0.77), fi_r=(0.76, 0.77),
            nose=(0.35, 0.60), # head relaxed between arms
        )
        f = extract_yoga_features(lm)
        is_cand, conf = pose.is_candidate(f)
        self.assertTrue(is_cand)
        evaluation = pose.evaluate_form(f)
        self.assertEqual(evaluation.status, FormStatus.CORRECT)

        # Craned neck / raised head
        lm_neck = create_mock_pose_landmarks(
            sh_l=(0.32, 0.55), sh_r=(0.32, 0.55),
            el_l=(0.25, 0.65), el_r=(0.25, 0.65),
            wr_l=(0.20, 0.75), wr_r=(0.20, 0.75),
            hp_l=(0.50, 0.25), hp_r=(0.50, 0.25),
            kn_l=(0.65, 0.50), kn_r=(0.65, 0.50),
            ak_l=(0.80, 0.75), ak_r=(0.80, 0.75),
            fi_l=(0.76, 0.77), fi_r=(0.76, 0.77),
            nose=(0.25, 0.45), # head craned up above shoulders
        )
    def test_all_pose_form_defects(self):
        # 1. Tadasana: Torso leaning, uneven shoulders, uneven hips, wide feet
        tadasana = TadasanaPose()
        lm_tad_defects = create_mock_pose_landmarks(
            sh_l=(0.45, 0.28), sh_r=(0.55, 0.32), # uneven shoulders
            hp_l=(0.46, 0.48), hp_r=(0.54, 0.52), # uneven hips
            kn_l=(0.46, 0.70), kn_r=(0.54, 0.70),
            ak_l=(0.40, 0.90), ak_r=(0.60, 0.90), # feet too wide
            nose=(0.50, 0.15),
        )
        f_tad = extract_yoga_features(lm_tad_defects)
        eval_tad = tadasana.evaluate_form(f_tad)
        self.assertEqual(eval_tad.status, FormStatus.ADJUST)
        self.assertTrue(any("SHOULDERS" in r for r in eval_tad.reasons))
        self.assertTrue(any("HIPS" in r for r in eval_tad.reasons))
        self.assertTrue(any("FEET" in r for r in eval_tad.reasons))

        # 2. Downward Dog: Hips too low, elbows bent, closed shoulders
        dog = DownwardDogPose()
        lm_dog_defects = create_mock_pose_landmarks(
            sh_l=(0.32, 0.50), sh_r=(0.32, 0.50),
            el_l=(0.28, 0.65), el_r=(0.28, 0.65), # bent elbows
            wr_l=(0.20, 0.75), wr_r=(0.20, 0.75),
            hp_l=(0.50, 0.40), hp_r=(0.50, 0.40), # hips low (flat V)
            kn_l=(0.65, 0.58), kn_r=(0.65, 0.58),
            ak_l=(0.80, 0.75), ak_r=(0.80, 0.75),
            fi_l=(0.76, 0.77), fi_r=(0.76, 0.77),
            nose=(0.35, 0.60),
        )
        f_dog = extract_yoga_features(lm_dog_defects)
        eval_dog = dog.evaluate_form(f_dog)
        self.assertEqual(eval_dog.status, FormStatus.ADJUST)
        self.assertTrue(any("HIPS NOT HIGH ENOUGH" in r or "ELBOWS" in r or "SHOULDERS" in r for r in eval_dog.reasons))

        # 3. Warrior I: Back knee bent, arms not overhead
        w1 = WarriorIPose()
        lm_w1_defects = create_mock_pose_landmarks(
            sh_l=(0.45, 0.35), sh_r=(0.55, 0.35),
            el_l=(0.35, 0.40), el_r=(0.65, 0.40), # arms low / not overhead
            wr_l=(0.35, 0.45), wr_r=(0.65, 0.45),
            hp_l=(0.45, 0.65), hp_r=(0.55, 0.55),
            kn_l=(0.30, 0.65), kn_r=(0.70, 0.80), # bent back knee
            ak_l=(0.30, 0.90), ak_r=(0.85, 0.90),
            fi_l=(0.22, 0.92), fi_r=(0.88, 0.92),
            nose=(0.50, 0.20),
        )
        f_w1 = extract_yoga_features(lm_w1_defects)
        eval_w1 = w1.evaluate_form(f_w1)
        self.assertEqual(eval_w1.status, FormStatus.ADJUST)
        self.assertTrue(any("BACK KNEE BENT" in r or "OVERHEAD" in r for r in eval_w1.reasons))

        # 4. Warrior II: Dropped / uneven arms, bent rear knee
        w2 = WarriorIIPose()
        lm_w2_defects = create_mock_pose_landmarks(
            sh_l=(0.45, 0.35), sh_r=(0.55, 0.35),
            el_l=(0.25, 0.50), el_r=(0.75, 0.35), # left arm dropped low
            wr_l=(0.10, 0.60), wr_r=(0.90, 0.35),
            hp_l=(0.45, 0.65), hp_r=(0.55, 0.50),
            kn_l=(0.30, 0.65), kn_r=(0.75, 0.80), # rear knee bent
            ak_l=(0.30, 0.90), ak_r=(0.90, 0.90),
            fi_l=(0.25, 0.92), fi_r=(0.95, 0.92),
            nose=(0.50, 0.20),
        )
        f_w2 = extract_yoga_features(lm_w2_defects)
        eval_w2 = w2.evaluate_form(f_w2)
        self.assertEqual(eval_w2.status, FormStatus.ADJUST)
        self.assertTrue(any("LEVEL" in r or "BACK KNEE" in r for r in eval_w2.reasons))

        # 5. Tree Pose: Bent support leg, unlevel hips
        tree = TreePose()
        lm_tree_defects = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            el_l=(0.42, 0.45), el_r=(0.58, 0.45),
            wr_l=(0.50, 0.40), wr_r=(0.50, 0.40),
            hp_l=(0.45, 0.46), hp_r=(0.55, 0.54), # unlevel hips
            kn_l=(0.30, 0.65), kn_r=(0.55, 0.78), # support knee bent
            ak_l=(0.50, 0.58), ak_r=(0.55, 0.90),
            nose=(0.50, 0.15),
        )
        f_tree = extract_yoga_features(lm_tree_defects)
        eval_tree = tree.evaluate_form(f_tree)
        self.assertEqual(eval_tree.status, FormStatus.ADJUST)
        self.assertTrue(any("SUPPORT KNEE BENT" in r or "HIPS" in r for r in eval_tree.reasons))

        # 6. Triangle Pose: Top arm dropped, bent knees
        tri = TrianglePose()
        lm_tri_defects = create_mock_pose_landmarks(
            sh_l=(0.28, 0.45), sh_r=(0.42, 0.28),
            el_l=(0.22, 0.62), el_r=(0.48, 0.30),
            wr_l=(0.18, 0.80), wr_r=(0.55, 0.35), # top arm dropped down
            hp_l=(0.44, 0.55), hp_r=(0.56, 0.55),
            kn_l=(0.25, 0.72), kn_r=(0.70, 0.72), # left knee bent
            ak_l=(0.20, 0.90), ak_r=(0.80, 0.90),
            nose=(0.32, 0.35),
        )
        f_tri = extract_yoga_features(lm_tri_defects)
        eval_tri = tri.evaluate_form(f_tri)
        self.assertEqual(eval_tri.status, FormStatus.ADJUST)
        self.assertTrue(any("TOP ARM" in r or "KNEE BENT" in r for r in eval_tri.reasons))


class TestClassifierAndTimer(unittest.TestCase):
    def test_classifier_debouncing_and_confirmation(self):
        classifier = YogaPoseClassifier(confirm_frames=5, lost_frames=10)
        lm = create_mock_pose_landmarks(
            sh_l=(0.45, 0.3), sh_r=(0.55, 0.3),
            el_l=(0.43, 0.45), el_r=(0.57, 0.45),
            wr_l=(0.43, 0.6), wr_r=(0.57, 0.6),
            hp_l=(0.46, 0.5), hp_r=(0.54, 0.5),
            kn_l=(0.46, 0.7), kn_r=(0.54, 0.7),
            ak_l=(0.46, 0.9), ak_r=(0.54, 0.9),
            nose=(0.50, 0.15),
        )
        f = extract_yoga_features(lm)

        # Frames 1 to 4: Candidate state, not yet confirmed
        for i in range(4):
            conf_pose, state, form_eval = classifier.update(f)
            self.assertIsNone(conf_pose)
            self.assertEqual(state, YogaPoseClassifier.STATE_CANDIDATE)

        # Frame 5: Reaches confirm_frames -> Transitions to CONFIRMED
        conf_pose, state, form_eval = classifier.update(f)
        self.assertIsNotNone(conf_pose)
        self.assertEqual(conf_pose.pose_id, "tadasana")
        self.assertEqual(state, YogaPoseClassifier.STATE_CONFIRMED)
        self.assertEqual(form_eval.status, FormStatus.CORRECT)

    def test_hold_timer_pause_and_reset(self):
        timer = YogaHoldTimer()
        tadasana = TadasanaPose()

        correct_eval = FormEvaluation(status=FormStatus.CORRECT)
        adjust_eval = FormEvaluation(status=FormStatus.ADJUST, reasons=["KNEES BENT"])

        # 1. Ticks forward when CORRECT
        t1 = timer.update(dt=0.5, confirmed_pose=tadasana, form_eval=correct_eval)
        self.assertAlmostEqual(t1, 0.5)
        self.assertEqual(timer.timer_status_str, "RUNNING")

        t2 = timer.update(dt=0.5, confirmed_pose=tadasana, form_eval=correct_eval)
        self.assertAlmostEqual(t2, 1.0)

        # 2. Pauses when ADJUST (does not increment or reset)
        t3 = timer.update(dt=0.5, confirmed_pose=tadasana, form_eval=adjust_eval)
        self.assertAlmostEqual(t3, 1.0)
        self.assertEqual(timer.timer_status_str, "PAUSED")

        # 3. Resumes when CORRECT again
        t4 = timer.update(dt=0.5, confirmed_pose=tadasana, form_eval=correct_eval)
        self.assertAlmostEqual(t4, 1.5)
        self.assertEqual(timer.timer_status_str, "RUNNING")

        # 4. Resets to 0.0 when confirmed pose is lost
        t5 = timer.update(dt=0.5, confirmed_pose=None, form_eval=FormEvaluation(status=FormStatus.LOST))
        self.assertEqual(t5, 0.0)
        self.assertEqual(timer.timer_status_str, "IDLE")


class TestHUDAndLogger(unittest.TestCase):
    def test_hud_rendering(self):
        hud = YogaHUD()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = TadasanaPose()
        form_eval = FormEvaluation(status=FormStatus.ADJUST, reasons=["TORSO LEANING", "KNEES BENT"])
        features = YogaFeatures(
            left_knee_angle=160.0, right_knee_angle=160.0, avg_knee_angle=160.0,
            left_hip_angle=170.0, right_hip_angle=170.0, avg_hip_angle=170.0,
            left_elbow_angle=170.0, right_elbow_angle=170.0, avg_elbow_angle=170.0,
            left_shoulder_angle=30.0, right_shoulder_angle=30.0, avg_shoulder_angle=30.0,
            torso_angle=12.0, stance_width_ratio=1.0, horizontal_stance_ratio=0.8, feet_distance_ratio=1.1,
            shoulder_level_diff=0.02, hip_level_diff=0.02,
            shoulder_mid_y=0.3, hip_mid_y=0.5, knee_mid_y=0.7, ankle_mid_y=0.9,
            left_wrist_y=0.6, right_wrist_y=0.6, left_shoulder_y=0.3, right_shoulder_y=0.3,
            left_ankle_y=0.9, right_ankle_y=0.9, left_knee_y=0.7, right_knee_y=0.7,
            hands_above_head=False, visibility=0.95
        )

        hud.draw(
            frame=frame,
            confirmed_pose=pose,
            lifecycle_state="CONFIRMED",
            form_eval=form_eval,
            hold_seconds=5.2,
            timer_status_str="PAUSED",
            features=features,
        )
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertTrue(np.any(frame > 0))

    def test_logger_csv_output(self):
        csv_test_path = "test_yoga_dataset.csv"
        if os.path.exists(csv_test_path):
            os.remove(csv_test_path)

        logger = YogaDatasetLogger(csv_path=csv_test_path, flush_every=2)
        pose = TadasanaPose()
        form_eval = FormEvaluation(status=FormStatus.CORRECT)
        features = YogaFeatures(
            left_knee_angle=175.0, right_knee_angle=175.0, avg_knee_angle=175.0,
            left_hip_angle=175.0, right_hip_angle=175.0, avg_hip_angle=175.0,
            left_elbow_angle=175.0, right_elbow_angle=175.0, avg_elbow_angle=175.0,
            left_shoulder_angle=20.0, right_shoulder_angle=20.0, avg_shoulder_angle=20.0,
            torso_angle=3.0, stance_width_ratio=1.0, horizontal_stance_ratio=0.8, feet_distance_ratio=1.1,
            shoulder_level_diff=0.01, hip_level_diff=0.01,
            shoulder_mid_y=0.3, hip_mid_y=0.5, knee_mid_y=0.7, ankle_mid_y=0.9,
            left_wrist_y=0.6, right_wrist_y=0.6, left_shoulder_y=0.3, right_shoulder_y=0.3,
            left_ankle_y=0.9, right_ankle_y=0.9, left_knee_y=0.7, right_knee_y=0.7,
            hands_above_head=False, visibility=0.98
        )

        logger.log_frame(timestamp_ms=0, confirmed_pose=pose, form_eval=form_eval, features=features, hold_time=0.0)
        logger.log_frame(timestamp_ms=33, confirmed_pose=pose, form_eval=form_eval, features=features, hold_time=0.033)
        logger.flush()

        self.assertTrue(os.path.exists(csv_test_path))
        with open(csv_test_path, "r") as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 3)

        if os.path.exists(csv_test_path):
            os.remove(csv_test_path)


if __name__ == "__main__":
    unittest.main()

