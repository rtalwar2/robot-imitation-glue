import os
import shutil
import time
from loguru import logger

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.base import BaseEnv

def replay_episode(  # noqa: C901
    env: BaseEnv,
    dataset: LeRobotDataset,
    action_to_env_converter,
    image_key,
    dataset_image_key,
    episode_idx: int = 0,
    fps=10,
):
    """Transform a LeRobot dataset using custom mapping functions.

    This function creates a new LeRobot dataset by applying transformation functions
    to both the dataset features and each frame in the original dataset.
    The original dataset remains unchanged.

    Args:
        repo_id: Repository ID for the original dataset. Either repo_id or root_dir must be provided.
        root_dir: Path to the original LeRobot dataset. Either repo_id or root_dir must be provided.
        new_root_dir: Path to save the transformed LeRobot dataset.
        new_repo_id: Repository ID for the new dataset (defaults to original repo_id + '_transformed').
        transform_fn: Function that takes a frame dictionary and returns a transformed frame dictionary.
        transform_features_fn: Optional function that takes a features dictionary and returns a transformed features dictionary.
        features_to_drop: List of feature names to drop from the new dataset.
        use_videos: Whether to use videos for the new dataset.
        image_writer_processes: Number of processes for image writing.
        image_writer_threads: Number of threads for image writing.
        verbose: Whether to print progress information.

    Returns:
        The transformed LeRobot dataset.

    Raises:
        ValueError: If neither repo_id nor root_dir is provided.
    """

    # episode_idx=0
    # fps=10
    if episode_idx==-1:
        n_episodes = len(dataset.episode_data_index["from"])

        for episode_idx in range(0,n_episodes):
            episode_indices = dataset.episode_data_index
            episode_start_idx = episode_indices["from"][episode_idx].item()
            episode_to_idx = episode_indices["to"][episode_idx].item()

            action = dataset[episode_start_idx]["state"].cpu().numpy()
            input(f"Press Enter to move robot to initial pose")
            env.act(
                robot_joints=action[0:6],
                gripper_pose=action[-1],
                timestamp=time.time() +  1.0 / fps,
            )

            input("Press Enter to start replay")

            duration = 1.0 / fps
            for i in range(episode_start_idx, episode_to_idx):
                action = dataset[i]["action"].cpu().numpy()
                obs = env.get_observations()
                # print(f"current obs = {obs}")
                # print(f"dataset obs = {dataset[i]}")
                # robot_pose, gripper = action_to_env_converter(env.get_robot_pose_se3(), env.get_gripper_opening(), action)
                logger.debug(f"delta z = {action}")
                logger.debug(f"current robot pose = {env.get_robot_pose_se3()}")
                logger.debug(f"current state observation = {obs['state']}")
                
                env.act_deltaz(action,env.get_robot_pose_se3(), time.time() + duration)
                time.sleep(duration)

            print("replay finished")
