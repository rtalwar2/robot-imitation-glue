import cv2
import numpy as np
from robot_imitation_glue.lerobot_dataset.replay_episode2_deltaz import replay_episode
from robot_imitation_glue.uR3station.robot_env import UR3eStation

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


if __name__ == "__main__":

    env = UR3eStation()

    dataset = LeRobotDataset(repo_id="", root="/home/rtalwar/robot-imitation-glue/datasets/delta_z")

    replay_episode(env,dataset, None, "scene_image", "observation.images.scene_image", -1)