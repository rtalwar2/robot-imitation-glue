import time

import cv2
import loguru
import numpy as np
import rerun as rr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.base import BaseAgent, BaseDatasetRecorder, BaseEnv
from robot_imitation_glue.utils import precise_wait
from robot_imitation_glue.forward_kinematics_helper import forward_kinematics_ur3e
# create type for callable that takes obs and returns action


logger = loguru.logger


class State:
    rollout_active = False
    is_stopped = False
    is_paused = False


class Event:
    start_rollout = False
    stop_rollout = False
    delete_last = False
    pause = False
    resume = False
    quit = False

    def clear(self):
        for attr in self.__dict__:
            setattr(self, attr, False)


def init_keyboard_listener(event: Event, state: State):
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            # "space bar"
            if key == keyboard.Key.enter and not state.rollout_active:
                event.start_rollout = True

            elif key == keyboard.Key.enter and state.rollout_active:
                event.stop_rollout = True

            elif hasattr(key, "char") and key.char == "p" and not state.rollout_active and not state.is_paused:
                # pause the episode
                event.pause = True

            elif hasattr(key, "char") and key.char == "p" and state.is_paused:
                # resume the episode
                event.resume = True

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and not state.rollout_active:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener

import torch
def overlay_all_keypoints(image, attn_tensor, alpha=0.5):
    """
    image_tensor: (C, H, W) torch tensor, values 0–1 or 0–255
    attn_tensor:  (K, h, w) torch tensor, typically (32,7,7)
    alpha: overlay transparency
    """

    # Convert image to uint8
    # if image.max() <= 1.0:
    #     image = (image * 255).astype(np.uint8)
    # else:
    #     image = image.astype(np.uint8)

    H_img, W_img = image.shape[:2]

    # Upsample all heatmaps at once
    heat = 1-attn_tensor.unsqueeze(0)  # (1, K, 7, 7)
    # heat = attn_tens|or
    heat_upsampled = torch.nn.functional.interpolate(
        heat, size=(H_img, W_img), mode='bilinear', align_corners=False
    )[0]  # (K, H, W)

    # Combine all into one heatmap by summing
    combined = heat_upsampled.sum(dim=0).cpu().numpy()  # (H, W)

    # Normalize to 0–255
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min() + 1e-6)
    combined_uint8 = (combined_norm * 255).astype(np.uint8)

    # Colorize using a heatmap (JET)
    combined_color = (cv2.applyColorMap(combined_uint8, cv2.COLORMAP_JET)/255.0).astype(np.float32)

    # Overlay the heatmap onto the RGB image
    overlay = cv2.addWeighted(image, 1 - alpha, combined_color, alpha, 0)

    return image,overlay, combined_uint8, combined_color

def eval(  # noqa: C901
    env: BaseEnv,
    teleop_agent: BaseAgent,
    policy_agent: BaseAgent,
    recorder: BaseDatasetRecorder,
    policy_to_pose_converter,
    teleop_to_pose_converter,
    fps=10,
    eval_dataset: LeRobotDataset = None,
    eval_dataset_image_key: str = "scene",
    env_observation_image_key: str = "scene",
    env_spectogram_key:str = "spectogram_image"
):
    """
    Evalulate a (policy) agent on a robot environment.

    You should also specify a teleop agent which allows to move the robot arm between policy rollouts to set the initial state.

    Rollouts are recorded using the provided dataset recorder.

    You can provide a dataset to load the initial scene image from. This is useful to evaluate the policy on a specific scene.

    Args:
        env: robot environment
        teleop_agent: teleop agent
        policy_agent: policy agent
        recorder: dataset recorder
        policy_to_pose_converter: function to convert policy action to robot pose
        teleop_to_pose_converter: function to convert teleop action to robot pose
        fps: frames per second for the dataset recorder
        eval_dataset: dataset to load initial scene image from
        eval_dataset_image_key: key in the dataset to load the image from

    """
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    rr.init("robot_imitation_glue", spawn=True)

    control_period = 1 / fps
    num_rollouts = recorder.n_recorded_episodes
    action = teleop_agent.get_action(env.get_observations())
    gripper_target = (1-action[-1])*0.085
    env.act(action[0:6],gripper_target,time.time() +5)
    while not state.is_stopped:

        initial_scene_image = None
        instruction = None

        # load initial image from the dataset if provided. display it on top of the current scene image,
        # this allows to set the initial state of the scene.
        if eval_dataset is not None:
            n_dataset_episodes = eval_dataset.num_episodes
            if num_rollouts < n_dataset_episodes:
                eval_dataset_episode = num_rollouts
            else:
                eval_dataset_episode = -1

            if eval_dataset_episode > -1:
                # get initial scene image
                print(eval_dataset.episode_data_index)
                step_idx = eval_dataset.episode_data_index["from"][eval_dataset_episode].item()
                print(eval_dataset[step_idx].keys())
                initial_scene_image = eval_dataset[step_idx][eval_dataset_image_key]

                # convert to numpy array of uint8 values
                initial_scene_image = initial_scene_image.permute(1, 2, 0).numpy()
                initial_scene_image *= 255
                initial_scene_image = initial_scene_image.astype(np.uint8)

                instruction = eval_dataset[step_idx]["task"]
                logger.info(
                    f"Loading initial state of episode {eval_dataset_episode} from eval dataset with instruction: {instruction}."
                )

            if initial_scene_image is not None:
                # show initial scene image
                rr.log("initial_scene_image", rr.Image(initial_scene_image))


        logger.info("Start teleop")
        logger.debug("moveL robot to current teleop pose")
        # first move slowly to the initial pose of the teleop device
        action = teleop_agent.get_action(env.get_observations())
        gripper_target = (1-action[-1])*0.085
        env.act(action[0:6],gripper_target,time.time() + control_period)
        logger.debug("robot moved to teleop pose")
        if initial_scene_image is not None:
            # blend initial scene image with current scene image
            # blended_image = cv2.addWeighted(initial_scene_image, 0.5, vis_image, 0.5, 0)
            rr.log("initial_scene_image", rr.Image(initial_scene_image))
        while not state.rollout_active:
            
            cycle_end_time = time.time() + control_period

            observations = env.get_observations()

            vis_image = observations[env_observation_image_key].copy()
            spectogram_image = observations[env_spectogram_key]
            rr.log("scene", rr.Image(vis_image))
            rr.log("spectogram", rr.Image(spectogram_image))


            action = teleop_agent.get_action(observations)
            logger.debug(f"teleop action: {action}")

            # convert teleop action to env action
            gripper_target = (1-action[-1])*0.085

            env.act(
                robot_joints=action[0:6],
                gripper_pose=gripper_target,
                timestamp=time.time() + control_period,
            )

            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)

            if event.quit:
                state.is_stopped = True
                listener.stop()
                recorder.finish_recording()
                logger.info("Stop evaluation")
                return

            if event.start_rollout:
                state.rollout_active = True
            event.clear()

        logger.info("Start rollout")
        recorder.start_episode()

        # reset to clear action buffers for chunking agents
        policy_agent.reset()
        tool_positions = []

        while not state.is_stopped and state.rollout_active:
            cycle_end_time = time.time() + control_period

            observations = env.get_observations()

            vis_image = observations[env_observation_image_key].copy()
            spectogram_image = observations[env_spectogram_key]

            ## print number of episodes to image
            cv2.putText(
                vis_image,
                f"Episode: {recorder.n_recorded_episodes}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            
            rr.log("scene", rr.Image(vis_image))
            rr.log("spectogram", rr.Image(spectogram_image))
            action, used_images,attn_maps = policy_agent.get_action(observations)

            if used_images is not None:
                # used_images shape: [batch, n_obs_steps, C, H, W]
                # assuming batch size 1:
                obs_imgs = used_images[0]          # shape [2, C, H, W]

                for i, img in enumerate(obs_imgs):
                    # convert tensor → numpy and reshape to HWC
                    img = img.squeeze(0)  # remove batch dim -> [C, H, W]

                    img_np = img.cpu().numpy().transpose(1, 2, 0)

                    rr.log(f"prediction_inputs/obs_{i}/raw", rr.Image(img_np))
                    image, overlay, heat_gray, heat_color = overlay_all_keypoints(img_np, attn_maps[0][0][i],0.7)
                    rr.log(f"prediction_inputs/obs_{i}/attention", rr.Image(overlay))


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

            next_height = X_B_TCP_virtual[2,3]
            if next_height<0.01:
                print(f"emergency break, almost hitting table with next height:{next_height}")
                return
            # convert  action to env action
            # new_robot_target_pose, new_target_gripper_state = policy_to_pose_converter(
            #     target_pose, target_gripper_state, action
            # )
            # logger.debug(f"new_robot_target_pose: {new_robot_target_pose}")
            # logger.debug(f"current robot pose: {env.get_robot_pose_se3()}")

            env.act(
                robot_joints=action[0:6],
                gripper_pose=action[-1],
                timestamp=time.time() + control_period,
            )

            recorder.record_step(observations, action.astype(np.float64))


            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)

            if event.stop_rollout:
                state.rollout_active = False
                num_rollouts += 1
                logger.info(f"Stop rollout {num_rollouts}")
                recorder.save_episode()
                event.clear()
                logger.info(f"Saved episode {recorder.n_recorded_episodes}")
                action = teleop_agent.get_action(env.get_observations())
                gripper_target = (1-action[-1])*0.085
                env.act(action[0:6],gripper_target,time.time() +5)


if __name__ == "__main__":
    """example of how to use the eval function"""
    import os

    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.mock import MockAgent, MockEnv, mock_agent_to_pose_converter

    env = MockEnv()
    env.reset()
    teleop_agent = MockAgent()
    policy_agent = MockAgent()

    if os.path.exists("datasets/demo"):
        dataset = LeRobotDataset(repo_id="mock", root="datasets/demo")
    else:
        dataset = None
    # create a dataset recorder

    if os.path.exists("datasets/test_dataset"):
        os.system("rm -rf datasets/test_dataset")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir="datasets/test_dataset",
        dataset_name="test_dataset",
        fps=10,
        use_videos=True,
    )

    eval(
        env,
        teleop_agent,
        policy_agent,
        dataset_recorder,
        mock_agent_to_pose_converter,
        mock_agent_to_pose_converter,
        fps=10,
        eval_dataset=dataset,
    )
