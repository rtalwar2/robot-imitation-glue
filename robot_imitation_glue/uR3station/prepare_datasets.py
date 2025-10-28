import cv2
import numpy as np

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper


features_to_drop = ["wrist_image_original", "scene_image_original"]


def features_transform(features):
    features["observation.state"] = features.pop("state")
    features["observation.state"]["shape"] = (7,)
    features["observation.images.wrist_image"] = features.pop("wrist_image")
    features["observation.images.spectogram_image"] = features.pop("spectogram_image")
    features["observation.images.wrist_image"]["shape"] = (3,64,64)
    features["observation.images.spectogram_image"]["shape"] = (3,64,64)
    features["action"]["shape"] = (7,)

    print("processed features:")
    print(features)
    return features


# def eef_features_transform(features):
#     features["observation.state"] = features.pop("state")
#     features["observation.state"]["shape"] = (7,)
#     features["observation.images.wrist_image"] = features.pop("wrist_image")
#     features["observation.images.scene_image"] = features.pop("scene_image")
#     features["action"]["shape"] = (10,)

#     print("processed features:")
#     print(features)
#     return features


def joints_frame_transform(frame):
    spectogram_image = frame["spectogram_image"]
    wrist_image = frame["wrist_image"]
    state = frame["state"]

    new_frame = frame.copy()
    new_frame.pop("spectogram_image")
    new_frame.pop("wrist_image")
    new_frame.pop("state")
    new_frame["observation.state"] = state
    new_frame["action"] = np.array(frame["action"]).astype(np.float64)

    resized_wrist_image = np.clip(cv2.resize(np.array(wrist_image),(64,64), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    resized_spectogram_image = np.clip(cv2.resize(np.array(spectogram_image),(64,64), interpolation=cv2.INTER_AREA), 0.0, 1.0)
    new_frame["observation.images.spectogram_image"] = resized_spectogram_image
    new_frame["observation.images.wrist_image"] = resized_wrist_image

    return new_frame


def ee_pose_frame_transform(frame):
    new_frame = frame.copy()
    new_frame["observation.state"] = frame["state"]
    new_frame["action"] = frame["action"]
    new_frame["observation.images.scene_image"] = frame["scene_image"]
    new_frame["observation.images.wrist_image"] = frame["wrist_image"]
    new_frame.pop("scene_image")
    new_frame.pop("wrist_image")
    new_frame.pop("state")

    return new_frame


transform_dataset(
    root_dir="/home/raman/imitation_button/datasets/test_5_episodes",
    new_root_dir="/home/raman/imitation_button/datasets/test_5_episodes_resized",
    transform_fn=joints_frame_transform,
    transform_features_fn=features_transform
)
