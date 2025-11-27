import cv2
import numpy as np

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper


features_to_drop = ["wrist_image_original", "scene_image_original"]

image_size = 224

def features_transform(features):
    features["observation.state"] = features.pop("state")
    features["observation.state"]["shape"] = (1,)
    # features["observation.state"]["shape"] = (7,)
    # features["observation.state"]["shape"] = (8,)
    features["observation.images.wrist_image"] = features.pop("wrist_image")
    # features["observation.images.spectogram_image"] = features.pop("spectogram_image")
    features["observation.images.wrist_image"]["shape"] = (3, image_size, image_size)
    # features["observation.images.spectogram_image"]["shape"] = (3, image_size, image_size)
    features["action"]["shape"] = (1,) #deltaz
    features.pop("wrist_image_original")
    features.pop("spectogram_image")  # rgb only
    # features["observation.images.spectogram_values"] = features.pop("spectogram_values")  # rgb only
    features.pop("spectogram_values")
    print("processed features:")
    print(features)
    return features


def joints_frame_transform(frame):
    spectogram_image = frame["spectogram_image"]
    spectogram_values_image = frame["spectogram_values"]
    wrist_image = frame["wrist_image"]
    state = frame["state"]
    button = frame["btn_state"]

    new_frame = frame.copy()
    new_frame.pop("spectogram_image")
    new_frame.pop("spectogram_values")
    new_frame.pop("wrist_image")
    new_frame.pop("wrist_image_original")
    new_frame.pop("state")
    # new_frame["observation.state"] = np.zeros(state.shape, dtype=np.float32)
    # new_frame["observation.state"] = state
    # new_frame["observation.state"] = np.concatenate((state,button))
    new_frame["observation.state"] = np.array([0.0],dtype=np.float32)
    new_frame["action"] = np.array(frame["action"]).astype(np.float64)

    resized_wrist_image = cv2.resize(
        np.array(wrist_image), (image_size, image_size), interpolation=cv2.INTER_AREA
    )
    resized_spectogram_image = cv2.resize(
        np.array(spectogram_image),
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    # new_frame["observation.images.spectogram_image"] = resized_spectogram_image
    # new_frame["observation.images.spectogram_values"] = spectogram_values_image
    new_frame["observation.images.wrist_image"] = resized_wrist_image

    return new_frame

l = [25,27,30,33,37,45,50,52,65,80,82,85,89,98]
l_bottom = [x for x in range(0,201) if x>=100 or x in l]

delta_z_automated_final_2=[3,6,7,9,10,11,19,20,22,24,26,28,29,31,32,35,36,37,39,40,42,43,45,46,50,52,54,55,56,57,60,62,64,65,67,68,69,70,70,72,73,74,75,76,77,78,79]

transform_dataset(
    root_dir="/home/rtalwar/robot-imitation-glue/datasets/delta_z_automated_final2_dynamic",
    new_root_dir="/home/rtalwar/robot-imitation-glue/datasets/delta_z_dynamic_rgb",
    transform_fn=joints_frame_transform,
    transform_features_fn=features_transform,
    episodes_to_drop=[2,3,14,18,23,31,34,35,36,37,38,43,45,50],
)
