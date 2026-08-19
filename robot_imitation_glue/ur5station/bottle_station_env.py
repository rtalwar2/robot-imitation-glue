"""Two-arm bottle-opening station.

`UR5eStation.ROBOT_IP` is already the bottle experiment's "left"/gripper arm, so the wrist
camera, the microphone spectrogram and left-arm control all come for free via `super()`. This
adds the right arm (holds the bottle) and the cap's 3-channel voltage sensor (see
`bottle_experiment/open_bottle_demo_analyze_sensors.py` for where these two arms and the
sensor were originally driven directly, without `BaseEnv`/DDS).
"""

import sys
from pathlib import Path

import loguru
from airo_robots.manipulators.hardware.ur_rtde import URrtde

from robot_imitation_glue.hardware.bottle_sensor import BottleSensorSubscriber
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOTTLE_EXPERIMENT_PATH = str(_REPO_ROOT / "bottle_experiment")
if _BOTTLE_EXPERIMENT_PATH not in sys.path:
    sys.path.insert(0, _BOTTLE_EXPERIMENT_PATH)

from calibrate_and_hover_bottle import (  # noqa: E402
    DEFAULT_TCP_RIGHT_TO_BOTTLE,
    RIGHT_ROBOT_IP,
    get_bottle_cap_pose_in_base_left,
)
from open_bottle_agent import is_opening_motion_reachable  # noqa: E402

logger = loguru.logger


class BottleStation(UR5eStation):
    def __init__(self, schunk, **kwargs):
        super().__init__(schunk, **kwargs)
        logger.info("connecting to right robot (bottle arm).")
        self.robot_right = URrtde(RIGHT_ROBOT_IP, URrtde.UR3E_CONFIG, gripper=None)
        logger.info("creating bottle sensor subscriber")
        self.bottle_sensor = BottleSensorSubscriber("Bottle")

    def get_observations(self):
        obs_dict = super().get_observations()
        obs_dict["bottle_sensor"] = self.bottle_sensor.get_bottle_sensor()
        return obs_dict

    def get_bottle_cap_pose(self):
        """Current bottle cap pose in this station's (ur_left's) base frame, recomputed from
        ur_right's live TCP pose -- valid as long as ur_right hasn't re-grasped the bottle."""
        return get_bottle_cap_pose_in_base_left(DEFAULT_TCP_RIGHT_TO_BOTTLE, self.robot_right)

    def move_right_to_tcp_pose(self, pose, joint_speed=0.2):
        self.robot_right.move_to_tcp_pose(pose, joint_speed=joint_speed).wait()

    def move_right_to_joint_configuration(self, joint_configuration, joint_speed=0.2):
        self.robot_right.move_to_joint_configuration(joint_configuration, joint_speed=joint_speed).wait()

    def is_bottle_pose_reachable(self, cap_pose):
        """Whether the left arm can hover above and fully execute the opening motion for this
        cap pose (checked against a ring of hypothetical touch points -- the real touch point
        is only known after detection)."""
        return is_opening_motion_reachable(cap_pose, self.robot)

    def close(self):
        super().close()
        self.robot_right.rtde_control.stopScript()
