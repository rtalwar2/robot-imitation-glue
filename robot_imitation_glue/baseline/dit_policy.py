"""A Diffusion Transformer (DiT) policy with flow matching, next to the behavior-cloning baseline.

The model follows the two ingredients of "The Ingredients for Robotic Diffusion Transformers"
(Dasari et al., https://dit-policy.github.io):

1. a *separate* image encoder per camera (here the same resnet18 `ImageEncoder` as the BC baseline),
2. **adaLN-Zero** conditioning instead of cross-attention: the observation/timestep vector modulates the
   layer norms of the transformer blocks and gates their residual branches (see `AdaLNBlock`).

Instead of DDPM we train with flow matching (as in https://github.com/Robot-Learning-Collective/pusht-slim):
the network predicts the constant velocity `action - noise` of the straight path between the noise and the
action chunk, and inference is a handful of Euler steps. This removes the noise schedule and the separate
scheduler object, and gives usable samples in ~10 network evaluations.

Unlike the BC baseline, the policy predicts a *chunk* of `prediction_horizon` future actions at once.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from robot_imitation_glue.baseline.policy import ImageEncoder


class TimestepEncoder(nn.Module):
    """Maps the scalar flow time t in [0,1] to an embedding: sinusoidal features followed by an MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        assert hidden_dim % 2 == 0, "the sinusoidal embedding needs an even dimension"
        half = hidden_dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000.0) * torch.arange(half).float() / half))
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # the frequencies are the usual diffusion ones, which assume timesteps in [0, 1000]; our flow time lives
        # in [0,1] so we rescale it, otherwise all but the highest frequency channels are constant.
        x = (1000.0 * t)[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


class AdaLNBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (Peebles & Xie, DiT).

    The conditioning vector is projected to 6 modulation parameters: a (shift, scale) pair for the layer norm of
    each sublayer and a gate that scales the output of each residual branch. The projection is zero-initialized,
    which is the "Zero" of adaLN-Zero: at initialization scale=shift=gate=0, so every block is exactly the
    identity and the network starts as a well-behaved shallow model that grows depth as the gates open. This is
    the stability ingredient of the DiT-Policy paper.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        # affine parameters are disabled, the (shift, scale) below take their role
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, T, hidden_dim) action tokens, cond: (B, hidden_dim) conditioning vector."""
        # (B, 1, hidden_dim) each, so they broadcast over the action tokens
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln_modulation(cond).unsqueeze(1).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale1) + shift1
        h, _ = self.attention(h, h, h, need_weights=False)
        x = x + gate1 * h

        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class FinalLayer(nn.Module):
    """adaLN-Zero output head: modulated layer norm + a zero-initialized linear to the action dimension."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.linear = nn.Linear(hidden_dim, action_dim)
        for layer in (self.adaln_modulation[1], self.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class DiTPolicy(nn.Module):
    """Predicts a chunk of `prediction_horizon` normalized actions with a flow-matching diffusion transformer.

    The network is the velocity field of the flow: it maps (noisy action chunk, flow time, observation) to the
    velocity that transports the noise to the action chunk.
    """

    def __init__(
        self,
        camera_keys: list[str],
        proprio_dim: int,
        action_dim: int,
        prediction_horizon: int = 16,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        assert len(camera_keys) > 0, "the policy needs at least one camera"
        # keep the config around so that a checkpoint can be loaded without knowing the hyperparameters
        self.config = {
            "camera_keys": list(camera_keys),
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "prediction_horizon": prediction_horizon,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
        }
        self.camera_keys = list(camera_keys)
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon

        # one resnet18 per camera, as in the DiT-Policy paper. A ModuleList instead of a ModuleDict, because the
        # lerobot camera keys contain dots.
        self.encoders = nn.ModuleList([ImageEncoder(hidden_dim, pretrained_backbone) for _ in self.camera_keys])
        self.state_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.timestep_encoder = TimestepEncoder(hidden_dim)
        # all cameras + proprioception + flow time are fused into the single adaLN conditioning vector
        self.cond_proj = nn.Sequential(
            nn.Linear((len(self.camera_keys) + 2) * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_pos_emb = nn.Parameter(torch.zeros(1, prediction_horizon, hidden_dim))
        nn.init.normal_(self.action_pos_emb, std=0.02)
        self.blocks = nn.ModuleList([AdaLNBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.final_layer = FinalLayer(hidden_dim, action_dim)

    def encode_observation(self, images: dict[str, torch.Tensor], proprio: torch.Tensor) -> torch.Tensor:
        """The camera and proprioception features. They do not depend on the flow time, so the sampler encodes
        the observation once and reuses it for every Euler step (the resnets dominate the inference cost)."""
        features = [encoder(images[key]) for key, encoder in zip(self.camera_keys, self.encoders)]
        features.append(self.state_encoder(proprio))
        return torch.cat(features, dim=-1)

    def predict_velocity(self, noisy_actions: torch.Tensor, t: torch.Tensor, obs_features: torch.Tensor):
        """noisy_actions: (B, horizon, action_dim), t: (B,), obs_features from `encode_observation`."""
        cond = self.cond_proj(torch.cat([obs_features, self.timestep_encoder(t)], dim=-1))
        x = self.action_proj(noisy_actions) + self.action_pos_emb
        for block in self.blocks:
            x = block(x, cond)
        return self.final_layer(x, cond)

    def forward(
        self, noisy_actions: torch.Tensor, t: torch.Tensor, images: dict[str, torch.Tensor], proprio: torch.Tensor
    ) -> torch.Tensor:
        """noisy_actions: (B, horizon, action_dim), t: (B,), images: dict of (B,3,H,W) tensors in [0,1],
        proprio: (B, proprio_dim) normalized. Returns the predicted velocity, shaped like `noisy_actions`."""
        return self.predict_velocity(noisy_actions, t, self.encode_observation(images, proprio))

    def compute_loss(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow matching loss on normalized action chunks.

        We sample a flow time t ~ U(0,1) and a Gaussian noise chunk, interpolate linearly between them
        (`x_t = (1-t) * noise + t * actions`) and regress the velocity of that straight path, which is the
        constant `actions - noise`.

        `mask` is a (B, horizon) float tensor that is 0 for timesteps beyond the end of the episode. The loss is
        the *mean* over the valid entries, so it stays on the same scale as an unmasked mean.
        """
        t = torch.rand(actions.shape[0], device=actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity = self.forward(x_t, t, images, proprio)

        error = nn.functional.mse_loss(velocity, actions - noise, reduction="none")
        if mask is None:
            return error.mean()
        mask = mask[..., None]
        return (error * mask).sum() / (mask.expand_as(error).sum().clamp(min=1.0))

    @torch.no_grad()
    def sample(self, images: dict[str, torch.Tensor], proprio: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Integrate the flow from noise to an action chunk with `n_steps` Euler steps of size 1/n_steps.

        Returns a (B, prediction_horizon, action_dim) tensor of normalized actions.
        """
        batch_size = proprio.shape[0]
        obs_features = self.encode_observation(images, proprio)
        x = torch.randn(batch_size, self.prediction_horizon, self.action_dim, device=proprio.device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((batch_size,), step * dt, device=proprio.device)
            x = x + dt * self.predict_velocity(x, t, obs_features)
        return x
