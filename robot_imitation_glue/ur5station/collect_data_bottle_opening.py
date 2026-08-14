import sys
from pathlib import Path

import numpy as np
from airo_robots.grippers.hardware.schunk_process import SchunkGripperProcess

from robot_imitation_glue.collect_data_bottle import collect_data_bottle_opening
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.ur5station.bottle_station_env import BottleStation

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

# bottle_station_env.py already put bottle_experiment/ on sys.path.
from open_bottle_agent import RANDOM_SEED, generate_reachable_bottle_poses  # noqa: E402

if __name__ == "__main__":
    schunk = SchunkGripperProcess(usb_interface="/dev/serial/by-path/pci-0000:00:14.0-usb-0:7:1.0-port0,11,115200,8E1")

    # with_spectogram records the mel spectrogram as an observation -- audio is one of the modalities
    # under evaluation and cannot be recovered after the fact, so it must be on from the first episode.
    # with_instrumentation refers to the button subscriber from the button experiment, not the bottle's
    # cap sensor, which BottleStation always records.
    env = BottleStation(
        schunk,
        with_instrumentation=False,
        with_spectogram_model=False,
        with_spectogram=True,
        use_internal_ft=True,
    )

    dataset_name = "bottle_opening_test_1"

    input("are you ready?")

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        # policy action = delta_xyz(3) + rotation_6d(6), same format collect_data_xyz records
        example_action=np.zeros((9,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/bottle_experiment/expert/{dataset_name}"),
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )

    # One seeded stream per split: sharing a single stream made the val poses the first 25
    # train poses, i.e. a train/test leak on the pose distribution.
    SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}
    if "train" in dataset_name:
        split, n_poses = "train", 100
    elif "val" in dataset_name:
        split, n_poses = "val", 25
    else:
        split, n_poses = "test", 5

    # Pose indices to skip, per split. Indices are per-split because each split is seeded
    # differently, so index 0 of "test" is a different pose from index 0 of "train". Excluding an
    # index does not shift the others (see generate_reachable_bottle_poses), so the generator just
    # samples one extra pose to make up the count.
    #
    # Only exclude poses that are impractical to *collect* on -- the bottle cannot be re-closed by
    # hand between episodes, the right arm fouls something, and so on. Excluding a pose because the
    # robot finds the task hard there biases the success rate upward, and for the test split that
    # directly inflates the headline result. Reachability is already filtered by
    # is_tcp_pose_reachable / is_opening_motion_reachable, so anything reaching this list is a
    # judgement call worth writing down.
    POSE_BLACKLIST = {"train": [], "val": [], "test": [0]}

    rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS[split])
    bottle_poses = generate_reachable_bottle_poses(
        n_poses, env.robot, env.robot_right, rng, blacklist=POSE_BLACKLIST[split]
    )

    collect_data_bottle_opening(
        env,
        dataset_recorder,
        frequency=10,
        bottle_poses=bottle_poses,
    )

    env.close()
