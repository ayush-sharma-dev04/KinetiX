"""
Abstract Base Class and evaluation data models for Yoga Poses.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple
from core.geometry import YogaFeatures


class FormStatus:
    CORRECT = "CORRECT"
    ADJUST = "ADJUST"
    LOST = "POSE LOST"


@dataclass
class FormEvaluation:
    """Detailed form check results for a pose."""
    status: str                         # CORRECT, ADJUST, POSE LOST
    reasons: List[str] = field(default_factory=list)  # Actionable feedback messages
    metrics: Dict[str, Any] = field(default_factory=dict) # Pose-specific metrics
    error_joints: Set[int] = field(default_factory=set)   # Landmark indices with form defects


class BaseYogaPose(ABC):
    """
    Abstract Base Class for modular yoga pose detection and form evaluation.
    Every pose implements:
      - is_candidate: fast signature classification to detect the pose
      - evaluate_form: detailed geometric rule checks for form feedback
    """

    @property
    @abstractmethod
    def pose_id(self) -> str:
        """Unique identifier, e.g. 'tadasana'."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """English name, e.g. 'Mountain Pose'."""
        pass

    @property
    @abstractmethod
    def sanskrit_name(self) -> str:
        """Sanskrit name, e.g. 'Tadasana'."""
        pass

    @abstractmethod
    def is_candidate(self, f: YogaFeatures) -> Tuple[bool, float]:
        """
        Determines whether the given feature frame matches the general shape of this pose.
        Returns:
            Tuple[bool, float]: (is_candidate, match_confidence [0.0 - 1.0])
        """
        pass

    @abstractmethod
    def evaluate_form(self, f: YogaFeatures) -> FormEvaluation:
        """
        Performs in-depth biomechanical form checks on the pose.
        Returns:
            FormEvaluation with status (CORRECT/ADJUST), feedback reasons, and metrics.
        """
        pass
