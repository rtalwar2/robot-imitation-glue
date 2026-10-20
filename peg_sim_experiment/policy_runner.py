"""Run a trained checkpoint in the plug sim through the SAME inference path as the robot (LerobotAgent).

Relative-action chunks are converted back to absolute poses with the pose cached at prediction time, and the
instrumented arm's auxiliary channel is split off here (kept as `last_aux` for logging) -- the environment only
ever receives the first 9 dims, exactly as LerobotAgent(n_env_action_dims=9) does on the rig.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference  # noqa: E402

N_ENV_ACTION_DIMS = 9


def observation_preprocessor(obs: dict) -> dict:
    image = torch.from_numpy(np.ascontiguousarray(obs["image"])).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    return {"observation.images.wrist_image": image,
            "observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0)}


class PolicyRunner:
    def __init__(self, checkpoint: Path, num_inference_steps: int = 10, n_action_steps: int = 8, device: str = "cuda"):
        policy, pre, post = make_lerobot_policy_for_inference(str(checkpoint), device=device)
        policy.config.num_inference_steps = num_inference_steps
        policy.diffusion.num_inference_steps = num_inference_steps
        policy.config.n_action_steps = n_action_steps
        # n_env_action_dims=None: keep the full vector to log the auxiliary prediction, slice below instead
        self.agent = LerobotAgent(policy, pre, post, device, observation_preprocessor, n_env_action_dims=None)
        self.action_dims = policy.config.action_feature.shape[0]
        self.last_aux: np.ndarray | None = None

    def reset(self):
        self.agent.reset()
        self.last_aux = None

    def __call__(self, obs: dict) -> np.ndarray:
        action, _, _ = self.agent.get_action(obs)
        self.last_aux = action[N_ENV_ACTION_DIMS:] if self.action_dims > N_ENV_ACTION_DIMS else None
        return action[:N_ENV_ACTION_DIMS]
