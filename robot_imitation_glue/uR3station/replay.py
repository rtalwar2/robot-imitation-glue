from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.lerobot_dataset.replay_episode import replay_episode
from robot_imitation_glue.uR3station.robot_env import UR3eStation
# Force TorchCodec to load before multiprocessing starts
from torchcodec.decoders import VideoDecoder
print("TorchCodec preloaded successfully.")

# if __name__ == "__main__":
#     env = UR3eStation()

#     try:
#         dataset = LeRobotDataset(repo_id="", root="/home/rtalwar/robot-imitation-glue/datasets/height_zero")
#         replay_episode(
#             env, dataset, None, "scene_image", "observation.images.scene_image", 0
#         )

#     finally:
#         env.close()
