from pathlib import Path
import shutil

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import pyarrow.parquet as pq
import torch

from robot_imitation_glue.base import BaseDatasetRecorder


class DummyDatasetRecorder(BaseDatasetRecorder):
    def start_episode(self):
        print("starting dataset episode recording")

    def record_step(self, obs, action):
        print("recording step")
        print("saving obs:", obs)
        print("saving action:", action)

    def save_episode(self):
        print("saving dataset episode")

    def set_episode_success(self, success):
        print("labeling dataset episode success:", success)

    @property
    def n_recorded_episodes(self):
        return 0


class LeRobotDatasetRecorder(BaseDatasetRecorder):
    DEFAULT_FEATURES = {
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }

    @staticmethod
    def _is_parquet_footer_corruption_error(exc: Exception) -> bool:
        current = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if "Parquet magic bytes not found in footer" in str(current):
                return True
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        return False

    @staticmethod
    def _find_corrupted_episode_metadata_parquets(dataset_root: Path) -> list[Path]:
        episodes_dir = dataset_root / "meta" / "episodes"
        if not episodes_dir.exists():
            return []

        corrupted_paths: list[Path] = []
        for parquet_path in sorted(episodes_dir.glob("*/*.parquet")):
            try:
                pq.read_metadata(parquet_path)
            except Exception:
                corrupted_paths.append(parquet_path)
        return corrupted_paths

    @staticmethod
    def _quarantine_corrupted_episode_metadata(dataset_root: Path, corrupted_paths: list[Path]) -> list[Path]:
        if len(corrupted_paths) == 0:
            return []

        episodes_dir = dataset_root / "meta" / "episodes"
        quarantine_dir = dataset_root / "meta" / "episodes_corrupt"
        moved_paths: list[Path] = []

        for parquet_path in corrupted_paths:
            rel_path = parquet_path.relative_to(episodes_dir)
            target_path = quarantine_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(parquet_path), str(target_path))
            moved_paths.append(target_path)

        return moved_paths

    def _resume_dataset_with_auto_repair(self) -> LeRobotDataset:
        try:
            return LeRobotDataset.resume(
                repo_id=self.dataset_name,
                root=self.root_dataset_dir,
                image_writer_processes=0,
                image_writer_threads=16,
            )
        except Exception as exc:
            if not self._is_parquet_footer_corruption_error(exc):
                raise

            print("Resume failed due to corrupted parquet footer in meta/episodes. Attempting repair.")
            corrupted_paths = self._find_corrupted_episode_metadata_parquets(self.root_dataset_dir)
            if len(corrupted_paths) == 0:
                raise

            moved_paths = self._quarantine_corrupted_episode_metadata(self.root_dataset_dir, corrupted_paths)
            print(
                "Quarantined corrupted metadata parquet files:\n"
                + "\n".join([str(path) for path in moved_paths])
            )
            print("Retrying dataset resume after metadata repair.")
            return LeRobotDataset.resume(
                repo_id=self.dataset_name,
                root=self.root_dataset_dir,
                image_writer_processes=0,
                image_writer_threads=16,
            )

    def _cleanup_interrupted_episode_image_dirs(self, episode_index: int) -> None:
        # If a previous run crashed mid-episode, stale frame PNGs can remain on disk.
        # Remove pending temp image dirs for the next episode index before recording.
        for image_key in self.image_keys:
            image_dir = (
                self.root_dataset_dir
                / "images"
                / image_key
                / f"episode-{episode_index:06d}"
            )
            if image_dir.is_dir():
                shutil.rmtree(image_dir)
                print(f"Removed stale interrupted image directory: {image_dir}")

    def __init__(
        self,
        example_obs_dict: dict,
        example_action: np.array,
        root_dataset_dir: Path,
        dataset_name: str,
        fps: int,
        use_videos=True,
    ):

        self.root_dataset_dir = Path(root_dataset_dir)
        self.dataset_name = dataset_name
        self.fps = fps

        self._n_recorded_episodes = 0
        self.key_mapping_dict = {}

        self.image_keys = []
        self.state_keys = []
        self.dataset_meta_info_path = self.root_dataset_dir / "meta" / "info.json"

        # create features  using the example dict. assume all numpy arrays. 0D is a scalar, 1D with size 1 is a scalar, 1D with size > 1 is a vector, 3D is an image.
        features = self.DEFAULT_FEATURES.copy()
        for key, value in example_obs_dict.items():
            shape = value.shape
            if len(shape) == 0:
                features[key] = {"dtype": str(value.dtype), "shape": (1,), "names": None}
            elif len(shape) == 1:
                features[key] = {"dtype": str(value.dtype), "shape": shape, "names": None}
            elif len(shape) == 3:
                if not shape[0] == 3:
                    # not channel first! reorder shape
                    shape = (shape[2], shape[0], shape[1])
                if use_videos:
                    features[key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}
                else:
                    features[key] = {"dtype": "image", "shape": shape, "names": None}
            else:
                raise ValueError(f"Unsupported shape for feature {key}: {shape}")

            if len(shape) == 3:
                self.image_keys.append(key)
            else:
                self.state_keys.append(key)
        
        # add action to features
        features["action"] = {"dtype": "float64", "shape": example_action.shape, "names": None}
        print(f"Features: {features}")

        if self.dataset_meta_info_path.exists():
            print(f"Dataset {dataset_name} already exists. Resuming it.")
            self.lerobot_dataset = self._resume_dataset_with_auto_repair()
            self._n_recorded_episodes = self.lerobot_dataset.meta.total_episodes
            self._cleanup_interrupted_episode_image_dirs(self._n_recorded_episodes)
            print(f"Loaded {self._n_recorded_episodes} episodes.")
        else:
            print(f"Dataset {dataset_name} does not exist. Creating it.")
            self.lerobot_dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=self.fps,
                root=self.root_dataset_dir,
                features=features,
                use_videos=use_videos,
                image_writer_processes=0,
                image_writer_threads=8,
            )
        print("Dataset created:", self.lerobot_dataset)

    def start_episode(self):
        pass

    def record_step(self, obs, action, success=False):
        frame = {
            "action": torch.from_numpy(action),
            "next.reward": torch.tensor([0.0]),
            "next.success": torch.tensor([bool(success)]),
            "seed": torch.tensor([0]),  # TODO: store the seed
            "task": "",
        }
        for key in self.image_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = obs[key]

        for key in self.state_keys:
            frame[key] = torch.tensor(obs[key])
        self.lerobot_dataset.add_frame(frame)

    def delete_episode(self):
        self.lerobot_dataset.clear_episode_buffer()

    def set_episode_success(self, success):
        # matches add_frame()'s own torch->numpy conversion, so every entry in the buffer
        # stays the same type whether it was set live or patched retroactively here.
        episode_buffer = self.lerobot_dataset.writer.episode_buffer
        episode_buffer["next.success"] = [np.array([bool(success)])] * episode_buffer["size"]

    def save_episode(self):
        self.lerobot_dataset.save_episode()
        self._n_recorded_episodes += 1

    def finish_recording(self):
        self.lerobot_dataset.finalize()

    @property
    def n_recorded_episodes(self):
        return self._n_recorded_episodes


if __name__ == "__main__":

    example_obs = {
        "robot_pose": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "image": np.random.rand(800, 400, 3).astype(np.float32),
        "image1": np.random.rand(3, 800, 400).astype(np.float32),
        "image2": np.random.rand(3, 800, 400).astype(np.float32),
        "image3": np.random.rand(3, 800, 400).astype(np.float32),
        "gripper_state": np.array([0.1], dtype=np.float32),
    }

    example_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

    import os

    # remove entire dataset directory
    os.system("rm -rf datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=example_obs,
        example_action=example_action,
        root_dataset_dir=Path("datasets"),
        dataset_name="test_dataset",
        fps=30,
        use_videos=True,
    )

    for j in range(3):
        dataset_recorder.start_episode()
        for i in range(10 - j):
            dataset_recorder.record_step(example_obs, example_action)
        dataset_recorder.save_episode()

    dataset_recorder.finish_recording()
    print(f"Recorded {dataset_recorder.n_recorded_episodes} episodes.")

    dataset = LeRobotDataset(repo_id="test_dataset", root=Path("datasets"), episodes=[0, 1])
    print(f"Loaded {len(dataset)} steps.")
    print(dataset.episode_data_index)
