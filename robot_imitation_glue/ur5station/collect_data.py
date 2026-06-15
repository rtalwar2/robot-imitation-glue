from copy import deepcopy
import os
from pathlib import Path
import sys

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
    # create dummy env, agent and recorder to test flow.

    env = UR5eStation(with_instrumentation=True,with_spectogram_model=False,use_internal_ft=True)

    dataset_name = "red_round_button_small_train"

    # gello_config = deepcopy(GelloTeleopDevice.GELLO1_DEFAULT_CONFIG)
    # agent = Gello4UR(
    #     gello_usb_port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT792AL6-if00-port0",
    #     gello_config=gello_config,
    #     ur_robot=env.robot,
    #     use_joint_space=True,
    # )
    input("are you ready?")

    # if not os.path.exists("datasets"):
    #     os.makedirs("datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        # For collect_data_xyz_red_switch: policy action = delta_xyz(3) + rotation_6d(6)
        example_action=np.zeros((9,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/expert/{dataset_name}"),
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )

    # print(dataset_recorder.state_keys)

    # collect_data(
    #     env,
    #     agent,
    #     dataset_recorder,
    #     10,
    #     delta_action_to_abs_se3_converter,
    #     abs_se3_to_relative_policy_action_converter,
    # )
    button_detector = ButtonDetector(env.get_camera_intrinsics(),None)

    env.robot.rtde_control.teachMode()
    input("go above button position")
    # detected_3d_button_positions = env.get_robot_pose_se3()[:3,3]
    # detected_3d_button_positions =[-0.48606219, -0.02978124 , 0.10837631] #red_round_button_big

    # detected_3d_button_positions = [-0.48683923, -0.04199,  0.07724089] #blue round button small
    # detected_3d_button_positions = [-0.49176621, -0.00877,  0.07627462]#white_switch
    # detected_3d_button_positions = [-0.48892, -0.05575,  0.05628161]#red_switch
    detected_3d_button_positions = [-0.48853 ,-0.03907   ,0.06231323]#red round button small
    # env.get_robot_pose_se3()[:3,3]
    # print("current robot position", detected_3d_button_positions)
    env.robot.rtde_control.endTeachMode()
    pre_touching_pose = button_detector.get_pretouch_position(detected_3d_button_positions)

    if "train" in dataset_name:
        deterministic_poses = generate_deterministic_poses(pre_touching_pose, env, count=100,seed=18)
    elif "val" in dataset_name:
        deterministic_poses = generate_deterministic_poses(pre_touching_pose, env, count=25,seed=19)
    collect_data_xyz(
        env,
        dataset_recorder,
        10,
        detected_3d_button_positions = detected_3d_button_positions,
        pre_touching_pose=pre_touching_pose,
        deterministic_poses=deterministic_poses
    )
    # collect_data(
    #     env,
    #     agent,
    #     dataset_recorder,
    #     10,
    #     delta_action_to_abs_se3_converter,
    #     abs_se3_to_relative_policy_action_converter,
    # )
    # teleoperate(
    #     env,
    #     agent
    # )

    env.close()
    # agent.close()