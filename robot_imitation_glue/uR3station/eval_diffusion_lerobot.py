import numpy as np
from regex import D
from robot_imitation_glue.uR3station.robot_env import UR3eStation
import torch
import cv2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.gello.gello_agent import DynamixelConfig
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval

image_size=224

if __name__ == "__main__":
    checkpoint_path = "/home/rtalwar/robot-imitation-glue/outputs/train/2025-11-17/15-52-29_rgb_joints_no_instrumentation_bottom/checkpoints/last/pretrained_model"
    train_dataset_path = (
        "/home/rtalwar/robot-imitation-glue/datasets/rgb_joints_no_instrumentation_bottom_resized"
    )
    eval_scenarios_dataset_path = train_dataset_path

    eval_dataset_name = "eval_rgb_joints23446598d7445684743rtyjtyjjy94"

    def preprocessor(obs_dict):
        spectogram_image = obs_dict["spectogram_image"]
        wrist_image = obs_dict["wrist_image"]
        state = obs_dict["state"]
        button = obs_dict["btn_state"]
        # state = np.concatenate((state,button))
        # state = np.zeros(state.shape,dtype=np.float32)
        resized_wrist_image = cv2.resize(np.array(wrist_image), (image_size, image_size), interpolation=cv2.INTER_AREA)
        resized_spectogram_image = cv2.resize(np.array(spectogram_image), (image_size, image_size), interpolation=cv2.INTER_AREA)

        state = torch.tensor(state).float().unsqueeze(0)
        spectogram_image = torch.tensor(resized_spectogram_image).float() / 255.0
        wrist_image = torch.tensor(resized_wrist_image).float() / 255.0
        spectogram_image = spectogram_image.permute(2, 0, 1)
        wrist_image = wrist_image.permute(2, 0, 1)

        # unsqueeze images
        spectogram_image = spectogram_image.unsqueeze(0)
        wrist_image = wrist_image.unsqueeze(0)

        return {
            "observation.images.spectogram_image": spectogram_image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
        }

    env = UR3eStation(with_instrumentation=True)
    env.reset()
    config = DynamixelConfig(
        joint_ids=(1, 2, 3, 4, 5, 6),
        joint_offsets=[
            4 * np.pi / 2,
            2 * np.pi / 2,
            0 * np.pi / 2,
            -3 * np.pi / 2,
            2 * np.pi / 2,
            7 * np.pi / 2,
        ],
        joint_signs=(1, 1, -1, 1, 1, 1),
        gripper_config=(7, 195, 154),
    )
    start_joints = np.concatenate(
        (env.robot.get_joint_configuration(), env.get_gripper_opening()), axis=0
    )
    agent = GelloAgent(config, "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT792AL6-if00-port0",start_joints)


    policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
    lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

    # create a dataset recorder

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=f"datasets/{eval_dataset_name}",
        dataset_name=eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    eval_scenarios_dataset = LeRobotDataset(
        repo_id="", root=eval_scenarios_dataset_path
    )
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    eval(
        env,
        agent,
        lerobot_agent,
        dataset_recorder,
        policy_to_pose_converter=None,
        teleop_to_pose_converter=None,
        fps=10,
        eval_dataset=eval_scenarios_dataset,
        eval_dataset_image_key="observation.images.wrist_image",
        env_observation_image_key="wrist_image",
        env_spectogram_key = "spectogram_image"
    )
