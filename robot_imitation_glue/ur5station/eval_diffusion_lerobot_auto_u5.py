import numpy as np
from regex import D
import torch
import cv2

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.gello.gello_agent import DynamixelConfig
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference

# from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent_delta_z import eval_xyz, eval_xyz_auto
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation

image_size=224


import torch
import numpy as np
from transformers import ASTForAudioClassification, AutoConfig



if __name__ == "__main__":
    # checkpoint_path = "/home/rtalwar/robot-imitation-glue/outputs/ramen-noodels/red_round_button_small_audio_pretrained"
    checkpoint_path = "/home/rtalwar/robot-imitation-glue/outputs/ramen-noodels/red_round_button_small_n75_button"
    # train_dataset_path = (
    #     "/home/rtalwar/robot-imitation-glue/datasets/delta_xyz_final_rgb"
    # )
    # eval_scenarios_dataset_path = train_dataset_path

    # eval_dataset_name = "eval_delta_xyz_final_rgb_audio_mit_frozen_intermediate_fixed_button"
    eval_dataset_name = "./eval_curve/red_round_button_small_n75_button_30s"

    def preprocessor(obs_dict):
        spectogram_values_image = obs_dict["spectogram_values"]
        spectogram_image = obs_dict["spectogram_image"]
        wrist_image = obs_dict["wrist_image"]
        state = obs_dict["state"]
        button = obs_dict["btn_state"]
        # pred_button = obs_dict["pred_button_state"]
        # state = np.concatenate((state,button))
        state = np.zeros(button.shape,dtype=np.float32)
        resized_wrist_image = cv2.resize(np.array(wrist_image), (320, 240))
        resized_spectogram_image = cv2.resize(np.array(spectogram_image), (image_size, image_size))

        # state = torch.tensor(button).float().unsqueeze(0)
        # state = torch.tensor(pred_button).float().unsqueeze(0)
        state = torch.tensor(state).float().unsqueeze(0)
        spectogram_image = torch.tensor(resized_spectogram_image).float() / 255.0
        wrist_image = torch.tensor(resized_wrist_image).float() / 255.0
        spectogram_image = spectogram_image.permute(2, 0, 1)
        wrist_image = wrist_image.permute(2, 0, 1)
        spectogram_values_image = torch.tensor(spectogram_values_image)
        spectogram_values_image = spectogram_values_image.permute(2, 0, 1)
        
        # unsqueeze images
        spectogram_image = spectogram_image.unsqueeze(0) 
        wrist_image = wrist_image.unsqueeze(0)
        spectogram_values_image = spectogram_values_image.unsqueeze(0) 
        return {
            # "observation.images.spectogram_image": spectogram_image,
            "observation.audio.spectogram_values" : spectogram_values_image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
        }

    env = UR5eStation(with_instrumentation=True,with_spectogram_model=False,use_internal_ft=True)
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


    # policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
    policy, lerobnot_preprocessor, lerobot_postprocessor = make_lerobot_policy_for_inference(checkpoint_path)
    lerobot_agent = LerobotAgent(policy,lerobnot_preprocessor, lerobot_postprocessor, "cuda", preprocessor)

    # create a dataset recorder

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((9,), dtype=np.float32),
        root_dataset_dir=f"datasets/{eval_dataset_name}",
        dataset_name=eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    # eval_scenarios_dataset = LeRobotDataset(
    #     repo_id="", root=eval_scenarios_dataset_path
    # )
    input("Press Enter to start evaluation")
    eval_xyz_auto(
        env,
        lerobot_agent,
        dataset_recorder,
        fps=10,
        env_observation_image_key="wrist_image",
        env_spectogram_key = "spectogram_image",
    )
