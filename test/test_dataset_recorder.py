import numpy as np
import pytest

pytest.importorskip("lerobot.common.datasets.lerobot_dataset")

from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder

EXAMPLE_OBS = {
    "scene_image": np.zeros((16, 16, 3), dtype=np.uint8),
    "state": np.array([0.1, 0.2, 0.3], dtype=np.float32),
}
EXAMPLE_ACTION = np.array([0.1, 0.2, 0.3], dtype=np.float32)


@pytest.fixture
def recorder(tmp_path):
    return LeRobotDatasetRecorder(
        example_obs_dict=EXAMPLE_OBS,
        example_action=EXAMPLE_ACTION,
        root_dataset_dir=tmp_path / "dataset",
        dataset_name="test_dataset",
        fps=10,
        use_videos=False,
    )


def _record_steps(recorder, n):
    recorder.start_episode()
    for _ in range(n):
        recorder.record_step(EXAMPLE_OBS, EXAMPLE_ACTION)


def test_set_episode_success_labels_all_recorded_frames(recorder):
    _record_steps(recorder, 3)
    recorder.set_episode_success(True)

    episode_buffer = recorder.lerobot_dataset.episode_buffer
    assert episode_buffer["size"] == 3
    assert all(bool(value) for value in episode_buffer["next.success"])

    recorder.save_episode()
    assert recorder.n_recorded_episodes == 1


def test_record_step_defaults_to_failure(recorder):
    _record_steps(recorder, 2)
    assert not any(bool(value) for value in recorder.lerobot_dataset.episode_buffer["next.success"])


def test_delete_episode_discards_current_episode(recorder):
    _record_steps(recorder, 2)
    recorder.save_episode()

    _record_steps(recorder, 2)
    recorder.delete_episode()

    assert recorder.lerobot_dataset.episode_buffer["size"] == 0
    assert recorder.n_recorded_episodes == 1
