"""Screening tests that decide whether a task's instrumentation is worth pretraining on.

Two tests, with different standing:

**Privilege test (`--input proprio`) -- mandatory, run first, on every task.** Predicts the
instrumentation from proprioception alone. If it succeeds, the robot already knows the signal from its
own joint state, nothing is privileged, and no encoder work can help. This is a go/no-go on the task
itself and is independent of which training mechanism is used. It is what disqualifies a wiping task
instrumented by grid coverage: under a non-revisiting path the grid index is a deterministic function
of end-effector position, so proprioception predicts it perfectly and the "instrumentation" is just a
noisier proprioception sensor.

**Modality suitability (`--input image|audio|ft`) -- mandatory for the pilot, diagnostic after.** The
pilot needs it because the encoder-pretraining arm has to pick *which* encoder to pretrain, and that
choice must follow a pre-registered rule rather than intuition. Once the mechanism is settled, if it
turns out to be the one that predicts instrumentation alongside the action, nothing branches on these
scores any more and they become a results-section table explaining why the method works on this task.

Scores are R2 against a mean-predictor, so 0 means "no better than always guessing the average".
**Fix the threshold before looking at the numbers** -- that is the whole point of pre-registering.
Read each modality's score as how much it adds over the proprioception floor.

Usage:
    python -m robot_imitation_glue.ur5station.bottle.screen_instrumentation \\
        --dataset-root datasets/bottle_experiment/prepared/bottle_9d_100 --input proprio --report 
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionAudioEncoder, DiffusionRgbEncoder

from robot_imitation_glue.ur5station.bottle.train_ast_bottle import (
    CALIBRATED_RANGE,
    episode_split,
    normalize_sensor,
    spectrogram_from_frame,
    spectrogram_stats,
)

IMAGE_KEY = "observation.images.wrist_image"
STATE_KEY = "observation.state"
SENSOR_KEY = "bottle_sensor"

# observation.state is [tcp_pose(6), drift-corrected ft(6)] -- see ur5station/prepare_datasets_bottle.py.
TCP_POSE_SLICE = slice(0, 6)
FT_SLICE = slice(6, 12)

WRIST_IMAGE_SHAPE = (3, 240, 320)
CROP_SHAPE = (216, 288)


class ScreeningDataset(Dataset):
    """Yields (encoder input, normalized 3-channel sensor target) for one modality."""

    def __init__(self, dataset: LeRobotDataset, frame_indices: list[int], input_mode: str, image_stats=None):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.input_mode = input_mode
        self.image_stats = image_stats

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int):
        frame = self.dataset[self.frame_indices[index]]
        target = torch.from_numpy(normalize_sensor(frame[SENSOR_KEY]))

        if self.input_mode == "image":
            image = frame[IMAGE_KEY].float()
            # Match the policy's own pipeline: MEAN_STD with dataset stats, no resize. Using ImageNet
            # stats or a 224 resize here (as the resnet-binary script does) would tune the encoder to
            # an input distribution the policy never produces, handicapping the pretrained variant for
            # a preprocessing reason rather than a representational one.
            mean, std = self.image_stats
            return (image - mean) / (std + 1e-8), target
        if self.input_mode == "audio":
            return torch.from_numpy(spectrogram_from_frame(frame)), target
        if self.input_mode == "proprio":
            return frame[STATE_KEY][TCP_POSE_SLICE].float(), target
        if self.input_mode == "ft":
            return frame[STATE_KEY][FT_SLICE].float(), target
        raise ValueError(f"unknown input mode {self.input_mode}")


def build_encoder(input_mode: str, dataset: LeRobotDataset, audio_stats: tuple[float, float], device: str):
    """Return (module, feature_dim). Image and audio use the policy's real encoder classes.

    Using the policy's classes matters for more than convenience: weights trained here load into the
    policy with strict=True and no key remapping, because the module structure is identical.
    """
    if input_mode == "image":
        config = DiffusionConfig(
            audio_backbone=None,
            use_group_norm=False,
            crop_shape=CROP_SHAPE,
            crop_is_random=True,
            device=device,
        )
        config.input_features = {
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=WRIST_IMAGE_SHAPE),
            STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        }
        config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
        encoder = DiffusionRgbEncoder(config)
        # DiffusionRgbEncoder returns (features, attention) in this fork.
        return _Unwrap(encoder), encoder.feature_dim

    if input_mode == "audio":
        mean, std = audio_stats
        time_dimension = spectrogram_from_frame(dataset[0]).shape[0]
        config = DiffusionConfig(
            audio_norm_mean=mean,
            audio_norm_std=std,
            time_dimension=time_dimension,
            audio_feature_type="embedding",
            freeze_audio_encoder=False,
            device=device,
        )
        encoder = DiffusionAudioEncoder(config)
        return encoder, encoder.feature_dim

    # Proprioception and force-torque have no encoder in the policy -- they are concatenated raw into
    # the conditioning vector. A small MLP is the fairest stand-in: it asks "is the signal recoverable
    # from these numbers at all", which is exactly what the gate is testing.
    hidden = 128
    return (
        nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()),
        hidden,
    )


class _Unwrap(nn.Module):
    """Drops the attention map from DiffusionRgbEncoder's 2-tuple return."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)[0]


def r2_per_channel(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    residual = ((labels - predictions) ** 2).sum(axis=0)
    total = ((labels - labels.mean(axis=0)) ** 2).sum(axis=0)
    return np.where(total > 0, 1.0 - residual / np.maximum(total, 1e-12), 0.0)


@torch.no_grad()
def evaluate(model, head, loader, device) -> dict:
    model.eval()
    head.eval()
    predictions, labels = [], []
    for inputs, targets in loader:
        predictions.append(head(model(inputs.to(device))).cpu().numpy())
        labels.append(targets.numpy())
    predictions = np.concatenate(predictions).astype(np.float64)
    labels = np.concatenate(labels).astype(np.float64)

    r2 = r2_per_channel(predictions, labels)
    return {
        **{f"r2_channel_{index}": float(value) for index, value in enumerate(r2)},
        "r2_mean": float(r2.mean()),
        "mse": float(((labels - predictions) ** 2).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--input", required=True, choices=["proprio", "image", "audio", "ft"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="defaults to 1e-5 for audio (finetuning a pretrained transformer) and 1e-4 otherwise",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save-encoder", type=Path, default=None, help="write an encoder_init checkpoint")
    parser.add_argument("--report", type=Path, default=None, help="append the scores to this JSON file")
    args = parser.parse_args()

    learning_rate = args.learning_rate or (1e-5 if args.input == "audio" else 1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # torchcodec's AV1 decoder crashes on the small (128x298) spectogram_values stream as soon
    # as it runs in any forked DataLoader worker process, at num_workers=4 AND at
    # num_workers=1 alike (RuntimeError: Could not push packet to decoder) -- confirmed not a
    # concurrency race between workers, and confirmed not corrupted data (a full sequential
    # ffmpeg decode and a single-process frame-by-frame torchcodec decode both pass cleanly).
    # pyav is the older, more conservative backend and doesn't hit this.
    video_backend = "pyav" if args.input == "audio" else None
    dataset = LeRobotDataset(repo_id=None, root=str(args.dataset_root), video_backend=video_backend)
    train_indices, validation_indices = episode_split(dataset, args.seed)

    audio_stats = (0.0, 1.0)
    if args.input == "audio":
        audio_stats = spectrogram_stats(dataset, train_indices)
        print(f"spectrogram stats (train only): mean={audio_stats[0]:.6f} std={audio_stats[1]:.6f}")

    image_stats = None
    if args.input == "image":
        stats = dataset.meta.stats[IMAGE_KEY]
        image_stats = (
            torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(3, 1, 1),
            torch.as_tensor(stats["std"], dtype=torch.float32).reshape(3, 1, 1),
        )

    model, feature_dim = build_encoder(args.input, dataset, audio_stats, device)
    model = model.to(device)
    head = nn.Linear(feature_dim, len(CALIBRATED_RANGE)).to(device)

    train_loader = DataLoader(
        ScreeningDataset(dataset, train_indices, args.input, image_stats),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    validation_loader = DataLoader(
        ScreeningDataset(dataset, validation_indices, args.input, image_stats),
        batch_size=args.batch_size,
        num_workers=4,
    )

    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=learning_rate)
    best = {"r2_mean": -float("inf")}
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        head.train()
        running = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(head(model(inputs.to(device))), targets.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()

        metrics = evaluate(model, head, validation_loader, device)
        print(
            f"epoch {epoch + 1}/{args.epochs}  train_mse={running / max(len(train_loader), 1):.6f}  "
            f"val_r2={metrics['r2_mean']:+.4f}  " + "  ".join(f"{k}={v:+.4f}" for k, v in metrics.items() if "channel" in k)
        )
        if metrics["r2_mean"] > best["r2_mean"]:
            best = {**metrics, "epoch": epoch + 1}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    print(f"\nbest validation: {json.dumps(best, indent=2)}")
    if args.input == "proprio":
        print(
            "PRIVILEGE GATE: a high score here means the instrumentation is recoverable from "
            "proprioception alone, i.e. not privileged, and the task is unsuitable."
        )

    if args.report:
        report = json.loads(args.report.read_text()) if args.report.exists() else {}
        report.setdefault(str(args.dataset_root), {})[args.input] = best
        args.report.write_text(json.dumps(report, indent=2))
        print(f"appended scores to {args.report}")

    if args.save_encoder:
        if args.input not in ("image", "audio"):
            raise ValueError("--save-encoder only applies to image or audio; the MLPs are diagnostics only")
        # Restore the best epoch first, then unwrap so the saved state dict is the encoder's own --
        # unprefixed keys, matching exactly what DiffusionModel expects to load with strict=True.
        if best_state is not None:
            model.load_state_dict(best_state)
        encoder = model.encoder if isinstance(model, _Unwrap) else model
        args.save_encoder.parent.mkdir(parents=True, exist_ok=True)
        payload = {"encoder_state_dict": encoder.state_dict(), "metrics": best, "dataset_root": str(args.dataset_root)}
        if args.input == "audio":
            payload |= {
                "audio_norm_mean": audio_stats[0],
                "audio_norm_std": audio_stats[1],
                "time_dimension": spectrogram_from_frame(dataset[0]).shape[0],
            }
        torch.save(payload, args.save_encoder)
        print(f"saved encoder to {args.save_encoder}")


if __name__ == "__main__":
    main()
