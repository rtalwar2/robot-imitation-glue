from os import name
import time
import cv2
import loguru
from robot_imitation_glue.forward_kinematics_helper import forward_kinematics_ur3e
from robot_imitation_glue.uR3station.robot_env import UR3eStation
import rerun as rr

from robot_imitation_glue.utils import precise_wait
import torch
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
)
import csv
import os
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    logger = loguru.logger
    
    checkpoint_path = "/home/rtalwar/robot-imitation-glue/outputs/train/2025-11-16/21-08-22_rgb_only_no_instrumentation_no_crop_bottom/checkpoints/last/pretrained_model"
    train_dataset_path = (
        "/home/rtalwar/robot-imitation-glue/datasets/rgb_only_no_instrumentation_bottom_resized"
    )
    # --- Paths ---
    checkpoint_path = "/home/rtalwar/robot-imitation-glue/outputs/train/2025-11-16/21-08-22_rgb_only_no_instrumentation_no_crop_bottom/checkpoints/last/pretrained_model"
    train_dataset_path = (
        "/home/rtalwar/robot-imitation-glue/datasets/rgb_only_no_instrumentation_bottom_resized"
    )

    def preprocessor(obs_dict):
        spectogram_image = obs_dict["spectogram_image"]
        wrist_image = obs_dict["wrist_image"]
        state = obs_dict["state"]
        button = obs_dict["btn_state"]
        # state = np.concatenate((state,button))
        # state = np.zeros(state.shape, dtype=np.float32)

        image_size = 224
        resized_wrist_image = cv2.resize(
            np.array(wrist_image),
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        resized_spectogram_image = cv2.resize(
            np.array(spectogram_image),
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

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
    rr.init("robot_imitation_glue", spawn=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "inference_logs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"inference_run_{timestamp}.csv")
    rows = []
    env = UR3eStation(with_instrumentation=True)

    device = get_safe_torch_device("cuda", log=True)
    dataset = LeRobotDataset(repo_id="", root=train_dataset_path)
    policy = make_lerobot_policy(checkpoint_path, train_dataset_path)
    policy_agent = LerobotAgent(policy, "cuda", preprocessor)

    episode_indices = dataset.episode_data_index
    episode_idx = 0
    start_idx = episode_indices["from"][episode_idx].item()
    end_idx = episode_indices["to"][episode_idx].item()
    output_dir2 = f"image_comparison_logs/episode_{episode_idx}_{timestamp}"
    os.makedirs(output_dir2, exist_ok=True)

    frame = dataset[start_idx]
    initial_action = frame["action"].cpu().numpy()
    control_period = 1 / 10
    env.reset()

    env.act(initial_action[0:6], initial_action[-1], time.time() + control_period)

    input("you are in initial_config")

    # reset to clear action buffers for chunking agents
    policy_agent.reset()
    tool_positions = []

    try:

        # --- Loop through frames in episode ---
        for idx in range(start_idx, end_idx):
            frame = dataset[idx]

            obs_ds = {
                "observation.images.wrist_image": frame[
                    "observation.images.wrist_image"
                ]
                .unsqueeze(0)
                .to(device),
                "observation.state": torch.tensor(frame["observation.state"])
                .unsqueeze(0)
                .to(device)
                .float(),
            }

            cycle_end_time = time.time() + control_period
            observations = env.get_observations()

            preprocessed_obs = preprocessor(observations)
            vis_image = preprocessed_obs["observation.images.wrist_image"]

            # --- Get live camera image ---
            live_img = vis_image
            
            # --- Get dataset wrist image ---
            # gt_images = (ds_img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ds_img = frame["observation.images.wrist_image"].unsqueeze(0)
            print(f"live image shape= {live_img.shape}")
            print(f"dataset image.shape {ds_img.shape}")
            # for PyTorch tensor live_image of shape [1, 3, 224, 224]
            live_img_np = (
                live_img.squeeze(0)  # remove batch dim -> [3, 224, 224]
                .permute(1, 2, 0)  # to [224, 224, 3]
                .cpu()
                .numpy()
            )

            # same for dataset image
            ds_img_np = ds_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # print(ds_img_np)
            # print(live_img_np)
            live_img_np = (live_img_np * 255).astype(np.uint8)
            ds_img_np = (ds_img_np * 255).astype(np.uint8)
            action, used_images = policy_agent.get_action(observations)
            if used_images is not None:
                # used_images shape: [batch, n_obs_steps, C, H, W]
                # assuming batch size 1:
                obs_imgs = used_images[0]          # shape [2, C, H, W]

                for i, img in enumerate(obs_imgs):
                    img = img.squeeze(0)  # remove batch dim -> [C, H, W]
                    # convert tensor → numpy and reshape to HWC
                    img_np = img.cpu().numpy().transpose(1, 2, 0)

                    rr.log(f"prediction_inputs/obs_{i}", rr.Image(img_np))

            logger.debug(f"policy action: {action}")
            X_B_TCP_virtual = forward_kinematics_ur3e(action[0:6])
            rr.log(
                "world/robot/tool_pose",
                rr.Transform3D(
                    translation=X_B_TCP_virtual[:3, 3],
                    mat3x3=X_B_TCP_virtual[:3, :3],
                ),
            )
            pos = X_B_TCP_virtual[:3, 3]
            tool_positions.append(pos)

            rr.log(
                "world/robot/trajectory",
                rr.LineStrips3D(np.array(tool_positions, dtype=np.float32)[None, :, :]),
            )
            timestep = len(rows)
            rows.append(np.concatenate(([timestep], action)))

            next_height = X_B_TCP_virtual[2, 3] - 0.19
            if next_height < 0.01:
                print(
                    f"emergency break, almost hitting table with next height:{next_height}"
                )
                break
            gt_action = frame["action"].cpu().numpy()

            env.act(
                robot_joints=gt_action[0:6],
                gripper_pose=gt_action[-1],
                timestamp=time.time() + control_period,
            )
            cv2.imwrite(
                os.path.join(output_dir2, f"live_{idx}.png"),
                cv2.cvtColor(live_img_np, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(output_dir2, f"dataset_{idx}.png"),
                cv2.cvtColor(ds_img_np, cv2.COLOR_RGB2BGR),
            )

            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)
    finally:
        # --- Save CSV after loop ---
        header = ["timestep"] + [f"action_{i}" for i in range(7)]
        rows_np = np.stack(rows)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows_np)

        print(f"✅ Saved inference data to {csv_path}")
