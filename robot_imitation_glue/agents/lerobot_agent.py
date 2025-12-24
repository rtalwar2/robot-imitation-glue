import time

import numpy as np
import torch
from loguru import logger

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.policies.diffusion.configuration_diffusion import PreTrainedConfig
# from lerobot.common.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from robot_imitation_glue.base import BaseAgent
import json
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# def make_lerobot_policy(pretrained_path, dataset_path):
#     """ """
#     # TODO: try to omit the need to load the dataset, bc it is not always on the same machine and is a source of errors..
#     # not sure why Lerobot has not simply stored the metadata in an additional file.
#     # with open(pretrained_path+"/config.json", "r") as f:
#     #     cfg = json.load(f)

#     # # Wrap it with type info if missing
#     # if "type" not in cfg:
#     #     cfg = {"type": "diffusion", **cfg}

#     policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
#     dataset = LeRobotDataset(repo_id="dataset", root=dataset_path)

#     # important! this actually loads the weight instead of random initialization.
#     policy_config.pretrained_path = pretrained_path
    # policy = make_policy(policy_config, ds_meta=dataset.meta)
#     policy.eval()
#     return policy
def make_lerobot_policy_for_inference(pretrained_path, device="cuda"):
    """
    Loads a policy and its processors for inference without requiring the original dataset.
    """
    # 1. Load the policy configuration to ensure we have the right type (optional if you know it's Diffusion)
    # config_path = Path(pretrained_path) / "config.json"
    # with open(config_path, "r") as f:
    #     cfg_dict = json.load(f)
    
    # 2. Load the Policy Weights directly
    # This automatically loads the config and weights.
    # Note: Replace DiffusionPolicy with the class matching your model (ACTPolicy, etc.)
    policy = DiffusionPolicy.from_pretrained(pretrained_path)
    policy.eval()
    policy.to(device)

    # 3. Load the Pre/Post Processors
    # passing pretrained_path allows it to load policy_preprocessor.json and .safetensors stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, 
        pretrained_path=pretrained_path
    )
    
    # # Move processors to device
    # if preprocessor:
    #     preprocessor.to(device)
    # if postprocessor:
    #     postprocessor.to(device)

    return policy, preprocessor, postprocessor
# def make_lerobor_pre_post_processor():

#     preprocessor, postprocessor = make_pre_post_processors()

class LerobotAgent(BaseAgent):
    """
    agent for inference on a policy trained with Lerobot.

    """

    def __init__(self,policy, preprocessor, postprocessor, device, observation_preprocessor):
        """
        processor must take the env obs dict and do
        1) numpy to tensor
        2) batchifying the observation
        3) renaming the keys to the policy expected keys
        4) (optional) do any other preprocessing, such as image resizing/cropping...


        """
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.observation_preprocessor = observation_preprocessor

    def get_action(self, observation):
        start_time = time.time()
        observation = self.observation_preprocessor(observation)
        end_time = time.time()
        logger.info(f"Lerobot agent observation preprocessor took {((end_time - start_time)*1000):.2f} ms")
        observation = {k: v.to(self.device) for k, v in observation.items()}
        with torch.no_grad():
            time_start = time.time()
            observation = self.preprocessor(observation)
            action,used_images ,attn_maps= self.policy.select_action(observation)
            action = self.postprocessor(action)
            time_end = time.time()
            logger.info(f"Lerobot agent inference took {((time_end - time_start)*1000):.2f} ms")
        return action.squeeze(0).cpu().numpy(),used_images,attn_maps

    def reset(self):
        self.policy.reset()


if __name__ == "__main__":

    path = "/home/tlips/Code/robot-imitation-glue/outputs/train/2025-04-10/13-15-24_pick-cube_diffusion/checkpoints/030000/pretrained_model"
    dataset_path = "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-remapped"

    policy = make_lerobot_policy(path, dataset_path).cpu()

    dataset = LeRobotDataset(repo_id="dataset", root=dataset_path)

    # test policy

    batch = dataset[0]
    batch.pop("action")
    # unsqueeze all tensors
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)

    time_before = time.time()
    action = policy.select_action(batch).squeeze(0)
    time_after = time.time()
    print(f"inferece took {time_after-time_before} s")
    print(f"policy action: {action}")
    print(f"dataset ['action']: {dataset[0]['action']}")

    print("testing inference on dummy observations")

    def observation_preprocessor(observation):
        observation["img1"] = observation["img1"].transpose(2, 0, 1)
        observation["img2"] = observation["img2"].transpose(2, 0, 1)
        observation["img1"] = observation["img1"].astype(np.float32) / 255.0
        observation["img2"] = observation["img2"].astype(np.float32) / 255.0

        observation["observation.images.scene_image"] = torch.from_numpy(observation["img1"]).float()
        observation["observation.images.wrist_image"] = torch.from_numpy(observation["img2"]).float()
        observation["observation.state"] = torch.from_numpy(observation["state"]).float()

        # drop old keys
        observation.pop("img1")
        observation.pop("img2")
        observation.pop("state")

        # resize images to 224x224
        from torchvision import transforms

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(196)])
        observation["observation.images.scene_image"] = transform(observation["observation.images.scene_image"])
        observation["observation.images.wrist_image"] = transform(observation["observation.images.wrist_image"])

        # add batch dimension
        observation["observation.images.scene_image"] = observation["observation.images.scene_image"].unsqueeze(0)
        observation["observation.images.wrist_image"] = observation["observation.images.wrist_image"].unsqueeze(0)
        observation["observation.state"] = observation["observation.state"].unsqueeze(0)
        return observation

    policy = make_lerobot_policy(path, dataset_path)
    policy = policy.to("cuda")
    agent = LerobotAgent(policy, "cuda", observation_preprocessor)
    for _ in range(20):
        test_obs = {
            "img1": np.random.randint(0, 255, (256, 256, 3)),
            "img2": np.random.randint(0, 255, (256, 256, 3)),
            "state": np.random.randn(7),
        }
        action = agent.get_action(test_obs)
    print(f"agent action: {action}")
