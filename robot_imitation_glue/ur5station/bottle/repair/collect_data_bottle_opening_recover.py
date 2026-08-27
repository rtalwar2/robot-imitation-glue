"""TEMPORARY recovery script -- re-collects episodes for bottle poses 32, 33 and 34, which were
lost from `bottle_opening_train` (present in meta/info.json's episode count but missing from both
meta/episodes/ and data/ -- most likely a collection run that crashed mid-`save_episode()`).

This is a thin wrapper around the normal `collect_data_bottle_opening.py` entrypoint: identical
setup, identical seeded pose generation (so `bottle_poses[32/33/34]` reproduce the exact same
poses the missing episodes were supposed to use), but passes `pose_indices=[32, 33, 34]` instead
of letting the collector cycle through all 100 train poses by `n_recorded_episodes`.

Resuming the dataset as-is is safe even though its `total_episodes` is currently inflated by the
gap: `LeRobotDataset.resume()` only reads meta/info.json et al, it does not need to load the
actual (incomplete) frame data. The 3 recovered episodes will land as new episode indices
appended after the current (inflated) count -- expect them at 51, 52, 53, not at 32, 33, 34. A
follow-up repair pass renumbers everything >=35 down by 3 (closing the original gap) and folds
these 3 new episodes in at 32, 33, 34, fixing `total_episodes` to the true count of 51.

Delete this file once the recovery + repair are done -- it is not part of the normal collection
pipeline and its whole reason to exist is this one gap.
"""

import sys
from pathlib import Path

import numpy as np
from airo_robots.grippers.hardware.schunk_process import SchunkGripperProcess

from robot_imitation_glue.collect_data_bottle import collect_data_bottle_opening
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.ur5station.bottle_station_env import BottleStation

_REPO_ROOT = Path(__file__).resolve().parents[2]

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

from open_bottle_agent import RANDOM_SEED, generate_reachable_bottle_poses  # noqa: E402

# The 3 missing episode indices, exactly as printed by the earlier gap diagnosis.
MISSING_POSE_INDICES = [32, 33, 34]

if __name__ == "__main__":
    schunk = SchunkGripperProcess(usb_interface="/dev/serial/by-path/pci-0000:00:14.0-usb-0:7:1.0-port0,11,115200,8E1")

    env = BottleStation(
        schunk,
        with_instrumentation=False,
        with_spectogram_model=False,
        with_spectogram=True,
        use_internal_ft=True,
    )

    dataset_name = "bottle_opening_train"

    input("are you ready?")

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((9,), dtype=np.float32),
        root_dataset_dir=Path(f"datasets/bottle_experiment/expert/{dataset_name}"),
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )
    print(
        f"[recover] resumed '{dataset_name}' at n_recorded_episodes={dataset_recorder.n_recorded_episodes} "
        f"(inflated by the gap -- the {len(MISSING_POSE_INDICES)} recovered episodes will land after this, "
        "not at their original indices; a repair pass fixes the numbering afterwards)"
    )

    # Must exactly match collect_data_bottle_opening.py's train-split pose generation, or
    # bottle_poses[32/33/34] would not be the same poses the missing episodes were supposed to use.
    SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}
    split, n_poses = "train", 100
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
        pose_indices=MISSING_POSE_INDICES,
    )

    env.close()
