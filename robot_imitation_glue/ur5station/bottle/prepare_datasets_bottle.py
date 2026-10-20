"""Prepare the bottle-opening datasets for the instrumentation experiment.

Separate from `prepare_datasets.py` (pour-cup-v2 joint/EEF remapping) and from
`uR3station/prepare_datasets.py` (button experiment), following the same one-script-per-experiment
pattern as those.

Turns one raw recording into the 8 datasets the two-arm x 4-data-level design needs:
{9-dim action, 12-dim action} x {100, 75, 50, 25}% of the successful episodes. (The 9-dim sets
serve both curve arms and the from_scratch verification cell; the 12-dim sets serve generic_c.)

The 12-dim variant appends the 3-channel cap sensor to the action vector. That is all the
treatment ("predict the instrumentation alongside the action") requires: lerobot reads the denoised width from
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

Levels are fractions of the successful episodes by default (the protocol's 100/75/50/25 %); with
--episode-counts they are fixed episode counts instead, named `<count>ep` -- still nested prefixes of the
same fixed-seed shuffle, so the smaller sets are always subsets of the larger ones:

    python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle                       # protocol levels, 9d + 12d
    python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle --episode-counts 50,70,90 --action-dims 9

Two additions (2026-10-13), both opt-in so the datasets above rebuild unchanged:

--extra-raw-root R adds EVERY successful episode of a second recording (the recovery demonstrations of
collect_recovery_bottle) to every level, on top of the level's expert episodes -- which stay exactly the expert
subsets above (same successful episodes, same seeded shuffle). The two recordings are first merged with
lerobot's aggregate_datasets (video concatenated by stream copy, no re-encode), so the prepared videos are
still one encoding generation away from the camera like every other prepared set.

--relative-actions stores the action as the ABSOLUTE target pose [xyz, rot6d] (the recorded per-step tool-frame
delta composed onto that frame's measured pose with policy_action_to_tcp_pose, exactly what the robot was sent)
and replaces the action statistics with those of UMI-style RELATIVE actions -- every action of a training chunk
expressed in the frame of the chunk's latest observed pose, which is what the policy (DiffusionConfig.
use_relative_actions -> SE3RelativeActionsProcessorStep) is trained on. An action_representation.json marker in
the dataset root says so; generate_configs reads it to switch the policy over.

--suffix names the levels "<count>ep_<suffix>" and their manifest episode_order_<suffix>.json, so these sit next
to the plain sets instead of replacing them:

    python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle --episode-counts 50,70,90 \
        --action-dims 9 --extra-raw-root bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train_recovery \
        --relative-actions --suffix recovery_rel
"""

import argparse
import json
from pathlib import Path

import shutil

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import absolute_to_relative_pose_actions

from robot_imitation_glue.collect_data_delta import policy_action_to_tcp_pose

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset

# The 2026-10-09 recollection (90 episodes). The collector writes to a RELATIVE path and was started from
# bottle_experiment/, so it landed under bottle_experiment/datasets/ -- the repo-root copy of the same name
# is the older 9-episode recording. Pass --raw-root for any other recording.
RAW_ROOT = Path("/home/rtalwar/robot-imitation-glue/bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train")
OUTPUT_ROOT = Path("/home/rtalwar/robot-imitation-glue/datasets/bottle_experiment/prepared")
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

    # bottle_sensor is deliberately kept: screening (screen_instrumentation) and offline success
    # re-labelling read it from the dataset. It can never leak into the policy because lerobot's
    # `batch_to_transition` drops every key that is not observation.*, action, or bookkeeping.
    return features


def _features_transform(action_dims: int):
    def transform(features: dict) -> dict:
        features = _rename_features(features)
        features["action"] = {**features["action"], "shape": (action_dims,), "dtype": "float32"}
        return features

    return transform


def absolute_target_action(robot_pose, delta_action) -> np.ndarray:
    """The recorded tool-frame delta composed onto the frame's measured pose -> absolute target [xyz, rot6d]."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", np.asarray(robot_pose[3:6], dtype=np.float64)).as_matrix()
    pose[:3, 3] = np.asarray(robot_pose[:3], dtype=np.float64)
    target = policy_action_to_tcp_pose(pose, np.asarray(delta_action[:N_ACTION_DIMS], dtype=np.float64))
    return np.concatenate((target[:3, 3], target[:3, 0], target[:3, 1])).astype(np.float32)


# Free-air frames at the start of an expert episode (hover + descent, before any contact) used to fit the
# FT offset of --ft-orientation-correction. Recovery episodes start at the jam, in contact: never fit those.
FT_FIT_FRAMES = 30
EPISODE_INDEX_KEY = "_episode_index"


def fit_ft_orientation_offsets(raw_root: Path, episodes: list[int]) -> dict[int, np.ndarray]:
    """Per episode, the FT reading's free-air model raw(t) = R_t @ o + c (o fixed in the tool frame, c in the base
    frame), least-squares over its first FT_FIT_FRAMES frames. Returns {episode: (6,) [o, c]}.

    Why: the expert collector did not re-zero the UR's FT sensor before capturing the home bias (fixed
    2026-10-19). The sensor's offset is fixed in the SENSOR frame, so `ft - ft_bias` leaves
    (R_t - R_home) @ offset at every other orientation: 6.7 N median, up to 17 N, in free air in the 90
    expert demos, vs ~2 N in the eval rollouts. Fitted on frames 0-11 this model brings frames 18-29 (34 deg
    further into the motion) from 9.1 N to 1.2 N median (p90 20.7 -> 2.1 N)."""
    files = sorted((raw_root / "data").glob("*/*.parquet"))
    table = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "robot_pose", "ft"]) for f in files)
    offsets = {}
    for episode in episodes:
        rows = table[table.episode_index == episode].sort_values("frame_index").iloc[:FT_FIT_FRAMES]
        pose = np.stack(rows.robot_pose.values).astype(np.float64)
        raw = np.stack(rows.ft.values)[:, :3].astype(np.float64)
        rotations = Rotation.from_euler("xyz", pose[:, 3:6]).as_matrix()
        design = np.concatenate((rotations, np.broadcast_to(np.eye(3), rotations.shape)), axis=2).reshape(-1, 6)
        offsets[episode], *_ = np.linalg.lstsq(design, raw.reshape(-1), rcond=None)
    return offsets


def _frame_transform(action_dims: int, relative_actions: bool = False, ft_offsets: dict | None = None):
    def transform(frame: dict) -> dict:
        new_frame = frame.copy()
        episode = new_frame.pop(EPISODE_INDEX_KEY, None)
        new_frame["observation.images.wrist_image"] = new_frame.pop("wrist_image")
        # `ft - ft_bias` applies the per-episode drift correction here rather than at collection
        # time, so the recording keeps the raw signal and the correction stays inspectable and
        # redoable. ft_bias is constant within an episode (captured at the fixed home pose before
        # anything is in contact) and is all-zeros for episodes recorded before this was added,
        # where it degrades to a no-op.
        ft = np.asarray(new_frame.pop("ft"), dtype=np.float32) - np.asarray(new_frame.pop("ft_bias"), dtype=np.float32)
        if ft_offsets is not None and episode in ft_offsets:
            # --ft-orientation-correction: the orientation-dependent free-air offset instead of the home bias.
            # Forces only; the torques keep the plain home-bias correction.
            o, c = ft_offsets[episode][:3], ft_offsets[episode][3:]
            rotation = Rotation.from_euler("xyz", np.asarray(frame["robot_pose"][3:6], dtype=np.float64)).as_matrix()
            ft[:3] = np.asarray(frame["ft"][:3], dtype=np.float64) - rotation @ o - c
        new_frame["observation.state"] = np.concatenate((np.asarray(new_frame.pop("robot_pose"), dtype=np.float32), ft))
        new_frame["observation.audio.spectogram_values"] = new_frame.pop("spectogram_values")

        action = np.asarray(frame["action"], dtype=np.float32)
        if relative_actions:
            action = absolute_target_action(frame["robot_pose"], action)
        if action_dims == N_ACTION_DIMS + N_SENSOR_CHANNELS:
            action = np.concatenate((action, np.asarray(frame["bottle_sensor"], dtype=np.float32)))
        new_frame["action"] = action

        for key in FEATURES_TO_DROP:
            new_frame.pop(key, None)
        return new_frame

    return transform


def successful_episodes(raw_root: Path) -> list[int]:
    """Episodes containing at least one frame flagged successful.

    Per-episode rather than per-frame because the collection loop sets `next.success` only once the
    final sensor checkpoint passes, so the earlier frames of a successful episode still carry False.
    Read straight from the parquet files: going through LeRobotDataset[frame] would decode every video
    frame just to look at a boolean.

    Anything recorded before the `leg_6_end` fix in collect_data_bottle.py is labelled False
    regardless of outcome and will be dropped here -- re-label those from the sensor logs first, or
    they are silently lost.
    """
    files = sorted((raw_root / "data").glob("*/*.parquet"))
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "next.success"]) for f in files])
    flagged = frames.groupby("episode_index")["next.success"].apply(lambda s: any(np.any(v) for v in s))
    return sorted(int(e) for e in flagged[flagged].index)


def nested_level_subsets(episodes: list[int], seed: int, counts: list[int] | None = None) -> dict[str, list[int]]:
    """Shuffle once, then take nested prefixes -- the 25% set inside the 50% set, and so on.

    Shuffling matters because appearance configurations get recorded in blocks: a subset taken in
    recording order would give the smallest level only the first sticker set, confounding data volume
    with appearance diversity. `counts` replaces the DATA_LEVELS fractions with fixed episode counts
    (levels named "<count>ep"); the same seeded order is used either way.
    """
    order = list(episodes)
    np.random.default_rng(seed).shuffle(order)

    if counts is not None:
        too_many = [c for c in counts if c > len(order)]
        if too_many:
            raise ValueError(f"asked for {too_many} episodes but only {len(order)} are successful")
        return {f"{count}ep": sorted(order[:count]) for count in counts}
    subsets = {}
    for level in DATA_LEVELS:
        count = max(1, round(len(order) * level))
        subsets[f"{round(level * 100)}"] = sorted(order[:count])
    return subsets


def _episode_spectrogram_moments(raw_dataset: LeRobotDataset, episodes: list[int]) -> dict[int, tuple]:
    """Per-episode (n, sum, sum_of_squares, time_dimension) of the spectrogram, one decode pass.

    Computed per episode so the per-LEVEL statistics combine from these without re-decoding: the
    levels are nested subsets of the same episodes, and the spectrogram lives in an AV1 video
    stream whose decode dominates the cost of this whole script.
    """
    moments = {}
    for episode_index in episodes:
        bounds = raw_dataset.meta.episodes[episode_index]
        count, total, total_squared, time_dimension = 0, 0.0, 0.0, None
        for frame_index in range(bounds["dataset_from_index"], bounds["dataset_to_index"]):
            values = np.asarray(raw_dataset[frame_index]["spectogram_values"], dtype=np.float64)
            if values.ndim == 3:  # stored as a 3-channel video with the signal repeated per channel
                values = values[0]
            time_dimension = values.shape[0]
            count += values.size
            total += values.sum()
            total_squared += (values**2).sum()
        moments[episode_index] = (count, total, total_squared, time_dimension)
        print(f"  episode {episode_index}: {count} spectrogram values")
    return moments


def write_relative_action_stats(root: Path, horizon: int = 16, n_obs_steps: int = 2, drop_n_last_frames: int = 7) -> None:
    """Replace meta/stats.json's action entry with the statistics of the RELATIVE actions the policy trains on.

    Mirrors what training samples: every frame t that can start a sample (the last drop_n_last_frames of each
    episode cannot), with the action chunk t-(n_obs_steps-1) .. t-(n_obs_steps-1)+horizon-1 (indices outside
    the episode copy-padded with its first/last frame, as lerobot pads) expressed in the frame of the state at t.
    The defaults are the bottle configs' DiffusionConfig values; a config with another horizon needs new stats.
    """
    files = sorted((root / "data").glob("*/*.parquet"))
    table = pd.concat(pd.read_parquet(f, columns=["episode_index", "action", "observation.state"]) for f in files)
    actions = torch.tensor(np.stack(table["action"].to_numpy()), dtype=torch.float64)
    states = torch.tensor(np.stack(table["observation.state"].to_numpy()), dtype=torch.float64)
    episodes = table["episode_index"].to_numpy()

    chunks = []
    offsets = np.arange(horizon) - (n_obs_steps - 1)
    for episode in np.unique(episodes):
        index = np.flatnonzero(episodes == episode)
        first, last = index[0], index[-1]
        for t in index[: max(len(index) - drop_n_last_frames, 1)]:
            chunk = actions[np.clip(t + offsets, first, last)]
            chunks.append(absolute_to_relative_pose_actions(chunk.unsqueeze(0), states[t].unsqueeze(0))[0])
    relative = torch.cat(chunks).numpy()

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = {
        "min": relative.min(0).tolist(),
        "max": relative.max(0).tolist(),
        "mean": relative.mean(0).tolist(),
        "std": relative.std(0).tolist(),
        "count": [int(relative.shape[0])],
        **{f"q{q:02d}": np.quantile(relative, q / 100, axis=0).tolist() for q in (1, 10, 50, 90, 99)},
    }
    stats_path.write_text(json.dumps(stats, indent=4))
    (root / "action_representation.json").write_text(json.dumps({
        "action": "absolute target pose [x, y, z, rot6d(6)] (first two rotation columns), base frame",
        "policy_input": "relative: SE3RelativeActionsProcessorStep, frame of the latest observation.state pose",
        "stats": "meta/stats.json action = RELATIVE action statistics",
        "use_relative_actions": True,
        "horizon": horizon, "n_obs_steps": n_obs_steps, "drop_n_last_frames": drop_n_last_frames,
    }, indent=2))
    print(f"{root.name}: relative action stats over {len(chunks)} chunks, "
          f"position range {np.round(relative[:, :3].min(0), 3)} .. {np.round(relative[:, :3].max(0), 3)} m")


def merge_raw_recordings(roots: list[Path], merged_root: Path) -> Path:
    """One raw dataset from several (episodes renumbered in order), reused if it already holds all of them."""
    from lerobot.datasets.aggregate import aggregate_datasets

    expected = sum(LeRobotDataset(repo_id=None, root=str(r)).meta.total_episodes for r in roots)
    if merged_root.exists():
        try:
            if LeRobotDataset(repo_id=None, root=str(merged_root)).meta.total_episodes == expected:
                print(f"reusing merged recording {merged_root} ({expected} episodes)")
                return merged_root
        except Exception as error:  # noqa: BLE001 -- an unreadable leftover is rebuilt
            print(f"merged recording {merged_root} unreadable ({error}) -- rebuilding")
        shutil.rmtree(merged_root)
    aggregate_datasets(repo_ids=[r.name for r in roots], aggr_repo_id=merged_root.name, roots=roots, aggr_root=merged_root)
    print(f"merged {[str(r) for r in roots]} -> {merged_root} ({expected} episodes)")
    return merged_root


def write_audio_stats(raw_root: Path, output_root: Path, action_dims_list=(N_ACTION_DIMS, N_ACTION_DIMS + N_SENSOR_CHANNELS),
                      manifest_name: str = "episode_order.json") -> None:
    """Per-LEVEL spectrogram mean/std, written as audio_stats.json into every prepared dataset root.

    Exists to prevent a normalization leak: `audio_norm_mean/std` in the training configs must be
    computed from the episodes that level actually trains on. Using one global value (e.g. from the
    100% pretraining run) would hand the 25% run statistics from episodes it never sees -- and since
    all arms at a level share these values, the leak would be invisible in the arm comparison while
    quietly contaminating the data-efficiency claim, which is the paper's headline.

    Stats are over each level's WHOLE subset (policy training has no validation split, so the subset
    is exactly the training data). The 9d and 12d variants of a level share identical audio, so they
    receive identical files. Reads the RAW dataset -- one decode pass total, per-episode moments
    combined per level -- with the pyav backend, since torchcodec crashes decoding the small AV1
    spectrogram stream inside worker processes and is fragile on it generally.
    """
    manifest = json.loads((output_root / manifest_name).read_text())
    raw_dataset = LeRobotDataset(repo_id=None, root=str(raw_root), video_backend="pyav")

    all_needed = sorted({e for episodes in manifest["levels"].values() for e in episodes})
    print(f"computing spectrogram moments for {len(all_needed)} episodes (one decode pass)...")
    moments = _episode_spectrogram_moments(raw_dataset, all_needed)

    for level, episodes in manifest["levels"].items():
        count = sum(moments[e][0] for e in episodes)
        total = sum(moments[e][1] for e in episodes)
        total_squared = sum(moments[e][2] for e in episodes)
        time_dimensions = {moments[e][3] for e in episodes}
        assert len(time_dimensions) == 1, f"inconsistent spectrogram time dimension: {time_dimensions}"

        mean = total / count
        std = float(np.sqrt(max(total_squared / count - mean**2, 1e-12)))
        stats = {
            "audio_norm_mean": float(mean),
            "audio_norm_std": std,
            "time_dimension": int(time_dimensions.pop()),
            "n_frames_worth": count,
            "episodes": episodes,
            "raw_root": str(raw_root),
        }
        for action_dims in action_dims_list:
            root = output_root / f"{OUTPUT_PREFIX}_{action_dims}d_{level}"
            if not root.exists():
                raise FileNotFoundError(f"{root} does not exist -- run the full prepare first")
            (root / "audio_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"level {level}: mean={mean:.6f} std={std:.6f} over {len(episodes)} episodes")


def prepare(raw_root: Path, output_root: Path, action_dims_list=(N_ACTION_DIMS, N_ACTION_DIMS + N_SENSOR_CHANNELS),
            counts: list[int] | None = None, extra_raw_root: Path | None = None, relative_actions: bool = False,
            suffix: str | None = None, pool_count: int | None = None, ft_orientation_correction: bool = False) -> None:
    """`pool_count` (protocol 1.6 N with extra episodes): the pool is the first pool_count expert episodes of
    the seeded order (exactly the `<pool_count>ep` set) plus every successful extra episode; that pool is
    shuffled with the same seed and cut into the 100/75/50/25% levels, so the extra (recovery) episodes are
    spread over the levels like any other episode instead of joining all of them."""
    expert_root = raw_root
    keep = successful_episodes(expert_root)
    extra = []
    if extra_raw_root is not None:
        n_expert = LeRobotDataset(repo_id=None, root=str(expert_root)).meta.total_episodes
        extra = [n_expert + e for e in successful_episodes(extra_raw_root)]
        raw_root = merge_raw_recordings([expert_root, extra_raw_root],
                                        expert_root.parent / f"{expert_root.name}_plus_{extra_raw_root.name}")
        print(f"{len(extra)} successful extra episodes ({extra_raw_root.name}), "
              + ("pooled with the expert episodes" if pool_count else "added to every level"))
    raw_dataset = LeRobotDataset(repo_id=None, root=str(raw_root))
    all_episodes = list(range(raw_dataset.meta.total_episodes))
    print(f"{len(keep) + len(extra)}/{len(all_episodes)} episodes successful; dropping "
          f"{sorted(set(all_episodes) - set(keep) - set(extra))}")
    if not keep:
        raise RuntimeError(
            "no successful episodes found -- if these were recorded before the leg_6_end fix, every "
            "frame carries next.success=False and the episodes must be re-labelled first"
        )

    # the expert part of every level is exactly the plain sets' subset: same episodes, same seeded shuffle
    if pool_count is not None:
        if counts is not None:
            raise ValueError("--pool-count builds the percentage levels; it cannot be combined with --episode-counts")
        pool = nested_level_subsets(keep, EPISODE_ORDER_SEED, [pool_count])[f"{pool_count}ep"] + extra
        subsets = nested_level_subsets(pool, EPISODE_ORDER_SEED)
        if suffix:
            subsets = {f"{level}_{suffix}": episodes for level, episodes in subsets.items()}
        print(f"pool of {len(pool)} episodes ({pool_count} expert + {len(extra)} extra): "
              + ", ".join(f"{level}: {len(episodes)} ({len(set(episodes) & set(extra))} extra)"
                          for level, episodes in subsets.items()))
    else:
        subsets = nested_level_subsets(keep, EPISODE_ORDER_SEED, counts)
        if suffix:
            subsets = {f"{level}_{suffix}": sorted(episodes + extra) for level, episodes in subsets.items()}
        elif extra:
            raise ValueError("--extra-raw-root needs --suffix: the levels would otherwise overwrite the plain sets")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_name = (f"episode_order_{suffix}.json" if suffix
                     else f"episode_order_pool{pool_count}.json" if pool_count else "episode_order.json")
    manifest = {
        "raw_root": str(raw_root),
        "episode_order_seed": EPISODE_ORDER_SEED,
        "successful_episodes": keep + extra,
        "extra_raw_root": str(extra_raw_root) if extra_raw_root else None,
        "extra_episodes": extra,
        "pool_count": pool_count,
        "relative_actions": relative_actions,
        "levels": subsets,
    }
    ft_offsets = fit_ft_orientation_offsets(raw_root, keep) if ft_orientation_correction else None  # expert episodes only
    manifest["ft_orientation_correction"] = ft_orientation_correction
    (output_root / manifest_name).write_text(json.dumps(manifest, indent=2))

    for action_dims in action_dims_list:
        for level, episodes in subsets.items():
            name = f"{OUTPUT_PREFIX}_{action_dims}d_{level}"
            print(f"\n=== {name}: {len(episodes)} episodes ===")
            transform_dataset(
                root_dir=str(raw_root),
                new_root_dir=str(output_root / name),
                new_repo_id=name,
                transform_fn=_frame_transform(action_dims, relative_actions, ft_offsets),
                transform_features_fn=_features_transform(action_dims),
                features_to_drop=FEATURES_TO_DROP,
                episodes_to_drop=sorted(set(all_episodes) - set(episodes)),
                episode_index_key=EPISODE_INDEX_KEY,
            )
            if relative_actions:
                write_relative_action_stats(output_root / name)
            verify(output_root / name, action_dims, len(episodes))

    write_audio_stats(raw_root, output_root, action_dims_list, manifest_name)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="only (re)compute audio_stats.json for already-prepared datasets, without rebuilding "
        "them -- rebuilding re-encodes all video, so use this to retrofit stats onto an existing "
        "prepare run",
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="the raw recording to prepare from")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--episode-counts", default=None,
                        help="comma-separated fixed episode counts (e.g. 50,70,90) instead of the 100/75/50/25%% levels")
    parser.add_argument("--extra-raw-root", type=Path, default=None,
                        help="a second recording (e.g. the recovery demos) whose successful episodes join every level")
    parser.add_argument("--relative-actions", action="store_true",
                        help="absolute target-pose actions + UMI-style relative action stats (DiffusionConfig.use_relative_actions)")
    parser.add_argument("--pool-count", type=int, default=None,
                        help="pool the first N expert episodes (seeded order) with every --extra-raw-root episode, then "
                             "cut the pool into the 100/75/50/25%% levels (the curve datasets once N is fixed)")
    parser.add_argument("--ft-orientation-correction", action="store_true",
                        help="replace the home-bias FT correction of the EXPERT episodes by a per-episode fit of the "
                             "orientation-dependent free-air offset (for recordings made without zeroFtSensor, i.e. "
                             "the 2026-10-09/10 expert demos); use a new --output-root or --suffix")
    parser.add_argument("--suffix", default=None, help="level/dataset name suffix, e.g. recovery_rel")
    parser.add_argument("--action-dims", default=f"{N_ACTION_DIMS},{N_ACTION_DIMS + N_SENSOR_CHANNELS}",
                        help="which variants to build: 9 = no instrumentation (generic, from_scratch), "
                             "12 = with the sensor channels (generic_c)")
    args = parser.parse_args()
    action_dims_list = tuple(int(d) for d in args.action_dims.split(","))
    if set(action_dims_list) - {N_ACTION_DIMS, N_ACTION_DIMS + N_SENSOR_CHANNELS}:
        raise SystemExit(f"--action-dims must be {N_ACTION_DIMS} and/or {N_ACTION_DIMS + N_SENSOR_CHANNELS}")
    counts = [int(c) for c in args.episode_counts.split(",")] if args.episode_counts else None
    if args.stats_only:
        write_audio_stats(args.raw_root, args.output_root, action_dims_list,
                          f"episode_order_pool{args.pool_count}.json" if args.pool_count else "episode_order.json")
    else:
        prepare(args.raw_root, args.output_root, action_dims_list, counts, args.extra_raw_root, args.relative_actions,
                args.suffix, args.pool_count, args.ft_orientation_correction)
