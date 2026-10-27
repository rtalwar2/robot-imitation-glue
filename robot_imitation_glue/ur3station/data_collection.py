import argparse
import os
from pathlib import Path

import numpy as np

from robot_imitation_glue.agents.airo_teleop_agent import make_spacemouse_agent
from robot_imitation_glue.collect_data import collect_data
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

# the delta action encoding is shared with the dataset preparation and the eval scripts,
# see robot_imitation_glue.preprocessing.
from robot_imitation_glue.preprocessing import (
    abs_se3_to_relative_policy_action_converter,
    delta_action_to_abs_se3_converter,
)
from robot_imitation_glue.ur3station.ur3_robot_env import UR3eStation

# default of the command line argument: edit it here or override it on the command line.
DATASET_NAME = "test_dataset"


def main():
    parser = argparse.ArgumentParser(description="Collect SpaceMouse demonstrations on the UR3 station.")
    parser.add_argument("--dataset-name", default=DATASET_NAME, help="name of the recorded dataset")
    args = parser.parse_args()

    # connects to the robot, the cameras and the SpaceMouse: only run this at the station.
    env = UR3eStation()
    agent = make_spacemouse_agent(env)

    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/{args.dataset_name}"),
        dataset_name=args.dataset_name,
        fps=10,
        use_videos=True,
    )

    collect_data(
        env,
        agent,
        dataset_recorder,
        10,
        delta_action_to_abs_se3_converter,
        abs_se3_to_relative_policy_action_converter,
    )

    env.close()
    agent.close()


if __name__ == "__main__":
    main()
