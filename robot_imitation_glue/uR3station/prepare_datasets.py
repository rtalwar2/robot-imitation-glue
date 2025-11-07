import cv2
import numpy as np

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper


features_to_drop = ["wrist_image_original", "scene_image_original"]

image_size = 224

def features_transform(features):
    features["observation.state"] = features.pop("state")
    # features["observation.state"]["shape"] = (7,)
    features["observation.state"]["shape"] = (8,)
    features["observation.images.wrist_image"] = features.pop("wrist_image")
    features["observation.images.spectogram_image"] = features.pop("spectogram_image")
    features["observation.images.wrist_image"]["shape"] = (3, image_size, image_size)
    features["observation.images.spectogram_image"]["shape"] = (3, image_size, image_size)
    features["action"]["shape"] = (7,)
    features.pop("wrist_image_original")
    print("processed features:")
    print(features)
    return features


def joints_frame_transform(frame):
    spectogram_image = frame["spectogram_image"]
    wrist_image = frame["wrist_image"]
    state = frame["state"]
    button = frame["btn_state"]

    new_frame = frame.copy()
    new_frame.pop("spectogram_image")
    new_frame.pop("wrist_image")
    new_frame.pop("wrist_image_original")
    new_frame.pop("state")
    # new_frame["observation.state"] = state
    new_frame["observation.state"] = np.concatenate((state,button))

    new_frame["action"] = np.array(frame["action"]).astype(np.float64)

    resized_wrist_image = np.clip(
        cv2.resize(np.array(wrist_image), (image_size, image_size), interpolation=cv2.INTER_AREA),
        0.0,
        1.0,
    )
    resized_spectogram_image = np.clip(
        cv2.resize(np.array(spectogram_image), (image_size, image_size), interpolation=cv2.INTER_AREA),
        0.0,
        1.0,
    )
    new_frame["observation.images.spectogram_image"] = resized_spectogram_image
    new_frame["observation.images.wrist_image"] = resized_wrist_image

    return new_frame



transform_dataset(
    root_dir="/home/rtalwar/robot-imitation-glue/datasets/height_one_part2",
    new_root_dir="/home/rtalwar/robot-imitation-glue/datasets/height_one_part2_resized",
    transform_fn=joints_frame_transform,
    transform_features_fn=features_transform,
    episodes_to_drop=[23,26,29,33,38,41,45,50,65]
)
