"""
Yoga Pose catalog and registration.
"""
from typing import List
from yoga.base_pose import BaseYogaPose
from yoga.poses.tadasana import TadasanaPose
from yoga.poses.warrior_ii import WarriorIIPose
from yoga.poses.warrior_i import WarriorIPose
from yoga.poses.tree_pose import TreePose
from yoga.poses.triangle import TrianglePose
from yoga.poses.downward_dog import DownwardDogPose


def get_default_pose_registry() -> List[BaseYogaPose]:
    """
    Returns an ordered list of pose detector instances.
    Evaluated in priority order:
      1. DownwardDogPose (inverted V shape)
      2. TrianglePose (wide stance, lateral tilt, vertical arm line)
      3. WarriorIPose (wide stance, bent knee, arms overhead)
      4. WarriorIIPose (wide stance, bent knee, arms horizontal T)
      5. TreePose (narrow base, single leg balance)
      6. TadasanaPose (standing upright, feet together)
    """
    return [
        DownwardDogPose(),
        TrianglePose(),
        WarriorIPose(),
        WarriorIIPose(),
        TreePose(),
        TadasanaPose(),
    ]
