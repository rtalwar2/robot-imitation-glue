"""Prepare the bottle-opening datasets for the instrumentation experiment.

Separate from `prepare_datasets.py` (pour-cup-v2 joint/EEF remapping) and from
`uR3station/prepare_datasets.py` (button experiment), following the same one-script-per-experiment
pattern as those.

Turns one raw recording into the 8 datasets the 4-arm x 4-data-level experiment needs:
{9-dim action, 12-dim action} x {100, 75, 50, 25}% of the successful episodes.

The 12-dim variant appends the 3-channel cap sensor to the action vector. That is all design C
("predict the instrumentation alongside the action") requires: lerobot reads the denoised width from
`config.action_feature.shape[0]`, so the U-Net, the sampling prior and the MIN_MAX normalizer all
widen on their own. The three extra channels take 3/12 of the denoising loss, and at inference
`LerobotAgent(n_env_action_dims=9)` drops them again.

Every dataset is built directly from the raw root, never from another prepared dataset: each
`transform_dataset` call re-encodes video, so deriving the 12-dim set from the 9-dim one would give
it two generations of compression against the others' one -- an image-quality difference aligned
exactly with the treatment. Building per level (rather than passing `episodes=` to the trainer) also
recomputes `meta/stats.json` per level, so normalization never sees episodes the run doesn't train
on, and it sidesteps a lerobot bug: `EpisodeAwareSampler` emits absolute frame indices while
`DatasetReader.get_item` expects relative ones, so a non-prefix `episodes=` list either raises or
silently trains on the wrong frames.
"""

import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset

RAW_ROOT = Path("datasets/bottle_experiment/expert/bottle_opening_train")
OUTPUT_ROOT = Path("datasets/bottle_experiment/prepared")
OUTPUT_PREFIX = "bottle"

# Fraction of the successful episodes each level trains on. Subsets are nested, so the 25% set is a
# subset of the 50% set, and so on.
DATA_LEVELS = (1.0, 0.75, 0.5, 0.25)

# Fixes which episodes land in which level. Do NOT reuse the collection script's RANDOM_SEED -- that
# one draws bottle poses, and sharing a stream between unrelated draws is how the train/val pose leak
# happened.
EPISODE_ORDER_SEED = 20260812

WRIST_IMAGE_SHAPE = (3, 240, 320)
N_ACTION_DIMS = 9
N_SENSOR_CHANNELS = 3

# Dropped: the 720p original (too large, and the policy trains on the resized copy), the RGB
# rendering of the spectrogram (the AST consumes the raw values instead), and the state components
# that `observation.state` does not carry.
FEATURES_TO_DROP = [
    "wrist_image_original",
    "scene_image_original",
    "scene_image",
    "spectogram_image",
    "joints",
    "gripper_state",
]


def _rename_features(features: dict) -> dict:
    """Raw recording keys -> the keys lerobot's Diffusion Policy looks for."""
    features["observation.images.wrist_image"] = features.pop("wrist_image")
    features["observation.images.wrist_image"]["shape"] = WRIST_IMAGE_SHAPE

    # `observation.state` is the ONLY state key the policy consumes: `robot_state_feature` matches on
    # that exact name, and any other observation.* vector is typed but never reaches global_cond. So
    # anything the policy should see as proprioception has to be concatenated in here -- hence TCP
    # pose (6) + internal force-torque (6). The consequence for the paper is that FT is not encoded
    # separately, so per-modality attribution for it comes from the offline screening runs.
    features["observation.state"] = features.pop("robot_pose")
    features["observation.state"]["shape"] = (12,)
    features.pop("ft")
    features.pop("ft_bias")

    # The fork's audio branch keys off exactly this name.
    features["observation.audio.spectogram_values"] = features.pop("spectogram_values")

    # bottle_sensor is deliberately kept: the design A pretraining reads it as its target, and it can
    # never leak into the policy because lerobot's `batch_to_transition` drops every key that is not
    # observation.*, action, or bookkeeping.
    return features


def _features_transform(action_dims: int):
    def transform(features: dict) -> dict:
        features = _rename_features(features)
        features["action"] = {**features["action"], "shape": (action_dims,), "dtype": "float32"}
        return features

    return transform


def _frame_transform(action_dims: int):
    def transform(frame: dict) -> dict:
        new_frame = frame.copy()
        new_frame["observation.images.wrist_image"] = new_frame.pop("wrist_image")
        # `ft - ft_bias` applies the per-episode drift correction here rather than at collection
        # time, so the recording keeps the raw signal and the correction stays inspectable and
        # redoable. ft_bias is constant within an episode (captured at the fixed home pose before
        # anything is in contact) and is all-zeros for episodes recorded before this was added,
        # where it degrades to a no-op.
        new_frame["observation.state"] = np.concatenate(
            (
                np.asarray(new_frame.pop("robot_pose"), dtype=np.float32),
                np.asarray(new_frame.pop("ft"), dtype=np.float32)
                - np.asarray(new_frame.pop("ft_bias"), dtype=np.float32),
            )
        )
        new_frame["observation.audio.spectogram_values"] = new_frame.pop("spectogram_values")

        action = np.asarray(frame["action"], dtype=np.float32)
        if action_dims == N_ACTION_DIMS + N_SENSOR_CHANNELS:
            action = np.concatenate((action, np.asarray(frame["bottle_sensor"], dtype=np.float32)))
        new_frame["action"] = action

        for key in FEATURES_TO_DROP:
            new_frame.pop(key, None)
        return new_frame

    return transform


def successful_episodes(dataset: LeRobotDataset) -> list[int]:
    """Episodes containing at least one frame flagged successful.

    Per-episode rather than per-frame because the collection loop sets `next.success` only once the
    final sensor checkpoint passes, so the earlier frames of a successful episode still carry False.

    Anything recorded before the `leg_6_end` fix in collect_data_bottle.py is labelled False
    regardless of outcome and will be dropped here -- re-label those from the sensor logs first, or
    they are silently lost.
    """
    successful = []
    for episode_index in range(dataset.meta.total_episodes):
        bounds = dataset.meta.episodes[episode_index]
        for frame_index in range(bounds["dataset_from_index"], bounds["dataset_to_index"]):
            if bool(np.asarray(dataset[frame_index]["next.success"]).any()):
                successful.append(episode_index)
                break
    return successful


def nested_level_subsets(episodes: list[int], seed: int) -> dict[str, list[int]]:
    """Shuffle once, then take nested prefixes -- the 25% set inside the 50% set, and so on.

    Shuffling matters because appearance configurations get recorded in blocks: a subset taken in
    recording order would give the smallest level only the first sticker set, confounding data volume
    with appearance diversity.
    """
    order = list(episodes)
    np.random.default_rng(seed).shuffle(order)

    subsets = {}
    for level in DATA_LEVELS:
        count = max(1, round(len(order) * level))
        subsets[f"{round(level * 100)}"] = sorted(order[:count])
    return subsets


def prepare(raw_root: Path, output_root: Path) -> None:
    raw_dataset = LeRobotDataset(repo_id=None, root=str(raw_root))
    all_episodes = list(range(raw_dataset.meta.total_episodes))
    keep = successful_episodes(raw_dataset)
    print(f"{len(keep)}/{len(all_episodes)} episodes successful; dropping {sorted(set(all_episodes) - set(keep))}")
    if not keep:
        raise RuntimeError(
            "no successful episodes found -- if these were recorded before the leg_6_end fix, every "
            "frame carries next.success=False and the episodes must be re-labelled first"
        )

    subsets = nested_level_subsets(keep, EPISODE_ORDER_SEED)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "raw_root": str(raw_root),
        "episode_order_seed": EPISODE_ORDER_SEED,
        "successful_episodes": keep,
        "levels": subsets,
    }
    (output_root / "episode_order.json").write_text(json.dumps(manifest, indent=2))

    for action_dims in (N_ACTION_DIMS, N_ACTION_DIMS + N_SENSOR_CHANNELS):
        for level, episodes in subsets.items():
            name = f"{OUTPUT_PREFIX}_{action_dims}d_{level}"
            print(f"\n=== {name}: {len(episodes)} episodes ===")
            transform_dataset(
                root_dir=str(raw_root),
                new_root_dir=str(output_root / name),
                new_repo_id=name,
                transform_fn=_frame_transform(action_dims),
                transform_features_fn=_features_transform(action_dims),
                features_to_drop=FEATURES_TO_DROP,
                episodes_to_drop=sorted(set(all_episodes) - set(episodes)),
            )
            verify(output_root / name, action_dims, len(episodes))


def verify(root: Path, action_dims: int, expected_episodes: int) -> None:
    """Catch the failure modes that would otherwise only surface as a confusing training run."""
    dataset = LeRobotDataset(repo_id=None, root=str(root))
    assert dataset.meta.total_episodes == expected_episodes, (
        f"{root.name}: {dataset.meta.total_episodes} episodes, expected {expected_episodes}"
    )

    stats_min = dataset.meta.stats["action"]["min"]
    assert len(stats_min) == action_dims, f"{root.name}: action stats have {len(stats_min)} dims"

    frame = dataset[0]
    assert frame["action"].shape[-1] == action_dims, f"{root.name}: action is {frame['action'].shape}"
    if action_dims > N_ACTION_DIMS:
        # The whole point of the 12-dim variant: the tail must be the sensor reading, in order.
        sensor = np.asarray(frame["bottle_sensor"], dtype=np.float32)
        tail = np.asarray(frame["action"][N_ACTION_DIMS:], dtype=np.float32)
        assert np.allclose(tail, sensor, atol=1e-5), f"{root.name}: action tail {tail} != sensor {sensor}"

    assert frame["observation.state"].shape[-1] == 12, f"{root.name}: state is {frame['observation.state'].shape}"
    print(f"{root.name}: OK ({expected_episodes} episodes, action {action_dims}d)")


if __name__ == "__main__":
    prepare(RAW_ROOT, OUTPUT_ROOT)
