from copy import deepcopy
import os
from pathlib import Path
import sys

from airo_teleop_agents.phone_teleop_agents import Phone4PositionManipulator
from airo_teleop_phone.config_phone import PhoneConfig, PhoneOS
from robot_imitation_glue.ur5station.data_collection import abs_se3_to_policy_action_converter
from robot_imitation_glue.button_detector.ButtonDetector import ButtonDetector
from robot_imitation_glue.eval_agent_delta_z import generate_deterministic_poses

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Prefer local workspace versions over globally visible packages from other projects.
_LOCAL_PATHS = [
    _REPO_ROOT / "airo-mono" / "airo-typing",
    _REPO_ROOT / "airo-mono" / "airo-spatial-algebra",
    _REPO_ROOT / "airo-mono" / "airo-robots",
    _REPO_ROOT / "airo-mono" / "airo-camera-toolkit",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-agents",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-devices",
]
for _path in _LOCAL_PATHS:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from airo_teleop_agents.gello_teleop_agents import Gello4UR
from airo_teleop_devices.gello_teleop_device import GelloTeleopDevice
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent
import numpy as np  # Missing import for np
from robot_imitation_glue.agents.gello.gello_agent import DynamixelConfig
from scipy.spatial.transform import Rotation as R

# from robot_imitation_glue.agents.spacemouse_agent import SpaceMouseAgent
from robot_imitation_glue.collect_data import collect_data, teleoperate
from robot_imitation_glue.collect_data_delta import collect_data_xyz, collect_data_xyz_red_switch, collect_data_xyz_white_switch
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation



def delta_action_to_abs_se3_converter(robot_pose_se3, gripper_state, action):
    # convert spacemouse action to ur3e action
    # we take the action to consist of a delta position, delta rotation and delta gripper width.
    # the delta rotation is interpreted as expressed in a frame with the same orientation as the base frame but with the origin at the EEF.
    # in this way, when rotating the spacemouse, the robot eef will not move around, while at the same time the axes of orientation
    # do not depend on the current orientation of the EEF.

    # the delta position is intepreted in the world frame and also applied on the EEF frame.

    delta_pos = action[:3]
    delta_rot = action[3:6]
    gripper_action = action[6]

    robot_trans = robot_pose_se3[:3, 3]
    robot_SO3 = robot_pose_se3[:3, :3]

    new_robot_trans = robot_trans + delta_pos
    # rotation is now interpreted as euler and not as rotvec
    # similar to Diffusion Policy.
    # however, rotvec seems more principled (related to twist)
    new_robot_SO3 = R.from_euler("xyz", delta_rot).as_matrix() @ robot_SO3

    new_robot_SE3 = np.eye(4)
    new_robot_SE3[:3, :3] = new_robot_SO3
    new_robot_SE3[:3, 3] = new_robot_trans

    new_gripper_state = gripper_state + gripper_action
    new_gripper_state = np.clip(new_gripper_state, 0, 0.085)

    return new_robot_SE3, new_gripper_state


def abs_se3_to_relative_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
    relative_se3 = np.linalg.inv(robot_pose) @ abs_se3_action

    relative_pos = relative_se3[:3, 3]
    relative_euler = R.from_matrix(relative_se3[:3, :3]).as_euler("xyz")
    relative_gripper = gripper_action - gripper_pose

    return np.concatenate((relative_pos, relative_euler, relative_gripper), axis=0).astype(np.float32)




if __name__ == "__main__":

    env = UR5eStation(with_instrumentation=True,with_spectogram_model=False,use_internal_ft=True)

    dataset_name = "testing_phone_datacollection"

    # create dummy env, agent and recorder to test flow.
    PHONE_OS = PhoneOS.ANDROID  # or PhoneOS.IOS
    ENABLE_TRANSLATIONS = True
    ENABLE_ROTATIONS = True
    phone_config = PhoneConfig(phone_os=PHONE_OS)
    
    teleop_agent = Phone4PositionManipulator(
    position_manipulator=env.robot,
    phone_config=phone_config,
    translation_scale=0.5,
    rotation_scale=1.0,
    phone_forward_axis = "-x",
    enable_settle_time_s= 0.25,
    max_translation_step_m= 0.03,
    max_rotation_step_rad= 0.35,
    enabled_axes=[ENABLE_TRANSLATIONS] * 3 + [ENABLE_ROTATIONS] * 3,
    auto_connect=True,
    )
    # if not os.path.exists("datasets"):
    #     os.makedirs("datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((10,), dtype=np.float64),
        root_dataset_dir=Path(f"datasets/expert/{dataset_name}"),
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )

    # input("are you ready?")
    collect_data(
        env,
        teleop_agent,
        dataset_recorder,
        10,
        delta_action_to_abs_se3_converter,
        abs_se3_to_policy_action_converter,
    )


    env.close()
    # agent.close()