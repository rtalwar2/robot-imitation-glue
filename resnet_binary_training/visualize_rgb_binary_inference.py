#!/usr/bin/env python3
"""Visualize inference examples from a LeRobot validation dataset.

This script:
1. Loads a trained ResNet18 binary classifier checkpoint.
2. Samples frames from a LeRobot validation dataset.
3. Runs inference and writes a figure with per-image GT/prediction annotations.

Example:
    python resnet_binary_training/visualize_rgb_binary_inference.py \
      --val-root /home/rtalwar/robot-imitation-glue/datasets/delta_xyz_final_rgb_audio \
      --checkpoint /home/rtalwar/robot-imitation-glue/resnet_binary_training/models/best_model.pt \
      --output /home/rtalwar/robot-imitation-glue/resnet_binary_training/models/inference_examples.png
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torchvision import models, transforms

# Repo path setup so `lerobot` can be imported when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


class InferenceDataset:
    """Thin wrapper to fetch model-ready tensors and display-ready images from LeRobotDataset."""

    def __init__(self, root: str, image_key: str, label_key: str, video_backend: str):
        self.ds = LeRobotDataset(
            repo_id="local",
            root=root,
            download_videos=False,
            video_backend=video_backend,
        )
        self.image_key = image_key
        self.label_key = label_key

        if image_key not in self.ds.features:
            keys = ", ".join(sorted(self.ds.features.keys()))
            raise ValueError(f"image_key '{image_key}' not found. Available keys: {keys}")
        if label_key not in self.ds.features:
            keys = ", ".join(sorted(self.ds.features.keys()))
            raise ValueError(f"label_key '{label_key}' not found. Available keys: {keys}")

        # Decode only the requested image stream if dataset has multiple video keys.
        if image_key in self.ds.meta.video_keys and len(self.ds.meta.video_keys) > 1:
            filtered_features = {
                key: value
                for key, value in self.ds.meta.info["features"].items()
                if key not in self.ds.meta.video_keys or key == image_key
            }
            self.ds.meta.info["features"] = filtered_features
            warnings.warn(
                f"Restricted video decoding to '{image_key}' to avoid unrelated stream decode failures.",
                stacklevel=2,
            )

        self.model_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _to_chw_float(image: Tensor) -> Tensor:
        if not isinstance(image, Tensor):
            image = torch.as_tensor(image)

        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got {tuple(image.shape)}")
        if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image

    @staticmethod
    def _to_binary_scalar(value: Tensor) -> float:
        if isinstance(value, Tensor):
            if value.numel() == 0:
                raise ValueError("Label tensor is empty")
            value = float(value.reshape(-1)[0].item())
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("Label sequence is empty")
            value = float(value[0])
        else:
            value = float(value)
        return 1.0 if value >= 0.5 else 0.0

    def get_item(self, idx: int, max_decode_retries: int) -> tuple[Tensor, Tensor, float]:
        sample = None
        last_error = None
        for attempt in range(max_decode_retries):
            try:
                sample = self.ds[idx]
                break
            except RuntimeError as err:
                last_error = err
                idx = random.randint(0, len(self.ds) - 1)
                if attempt == max_decode_retries - 1:
                    raise RuntimeError(
                        "Failed to decode sample after retries. "
                        f"Last error: {type(err).__name__}: {err}"
                    ) from err

        if sample is None:
            raise RuntimeError(f"Sample loading failed unexpectedly. Last error: {last_error}")

        raw_image = self._to_chw_float(sample[self.image_key])
        label = self._to_binary_scalar(sample[self.label_key])
        model_image = self.model_transform(raw_image)
        return raw_image, model_image, label


def build_resnet18_binary() -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = build_resnet18_binary().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a state dict or a dict containing 'model_state_dict'.")

    # Handle DataParallel checkpoints.
    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()
    }

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            "Checkpoint load summary:",
            f"missing_keys={len(result.missing_keys)}",
            f"unexpected_keys={len(result.unexpected_keys)}",
        )

    model.eval()
    return model


def visualize(
    val_root: str,
    checkpoint: str,
    output: str,
    image_key: str,
    label_key: str,
    num_examples: int,
    threshold: float,
    seed: int,
    video_backend: str,
    max_decode_retries: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = InferenceDataset(
        root=val_root,
        image_key=image_key,
        label_key=label_key,
        video_backend=video_backend,
    )
    model = load_model(checkpoint, device)

    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    num_examples = min(num_examples, len(dataset))
    indices = random.sample(range(len(dataset)), k=num_examples)

    cols = min(4, max(1, num_examples))
    rows = (num_examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.ravel().tolist()

    correct = 0
    with torch.no_grad():
        for ax, idx in zip(axes, indices, strict=False):
            raw_image, model_image, gt_label = dataset.get_item(idx, max_decode_retries=max_decode_retries)

            logits = model(model_image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).item()
            pred_label = 1.0 if prob >= threshold else 0.0
            is_correct = pred_label == gt_label
            correct += int(is_correct)

            np_image = raw_image.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            ax.imshow(np_image)
            ax.axis("off")

            title_color = "green" if is_correct else "red"
            ax.set_title(
                (
                    f"idx={idx} | GT={int(gt_label)} | Pred={int(pred_label)}\n"
                    f"p(class=1)={prob:.3f}"
                ),
                color=title_color,
                fontsize=10,
            )

        for ax in axes[num_examples:]:
            ax.axis("off")

    acc = correct / num_examples
    fig.suptitle(
        f"Binary Inference Examples | n={num_examples} | threshold={threshold:.2f} | sample_acc={acc:.3f}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved inference visualization to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize binary model inference on LeRobot validation frames")
    parser.add_argument(
        "--val-root",
        type=str,
        required=True,
        help="Path to validation LeRobot dataset root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pt or last_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "resnet_binary_training" / "models" / "inference_examples.png"),
        help="Path to output visualization image",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="observation.images.wrist_image",
        help="Image feature key in LeRobot dataset",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="btn_state",
        help="Binary label key in LeRobot dataset",
    )
    parser.add_argument("--num-examples", type=int, default=16, help="Number of random validation examples")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class 1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decoder backend for LeRobotDataset",
    )
    parser.add_argument(
        "--max-decode-retries",
        type=int,
        default=8,
        help="Retries with alternate indices if video decoding fails for a sample",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")
    if args.num_examples <= 0:
        raise ValueError("num_examples must be > 0")
    if args.max_decode_retries <= 0:
        raise ValueError("max_decode_retries must be > 0")

    return args


if __name__ == "__main__":
    cli = parse_args()
    visualize(
        val_root=cli.val_root,
        checkpoint=cli.checkpoint,
        output=cli.output,
        image_key=cli.image_key,
        label_key=cli.label_key,
        num_examples=cli.num_examples,
        threshold=cli.threshold,
        seed=cli.seed,
        video_backend=cli.video_backend,
        max_decode_retries=cli.max_decode_retries,
    )

"""python /home/rtalwar/robot-imitation-glue/resnet_binary_training/visualize_rgb_binary_inference.py \
  --val-root /home/rtalwar/robot-imitation-glue/datasets/delta_xyz_final_rgb_audio \
  --checkpoint /home/rtalwar/robot-imitation-glue/resnet_binary_training/models/best_model.pt \
  --output /home/rtalwar/robot-imitation-glue/resnet_binary_training/models/inference_examples.png \
  --num-examples 32 \
  --video-backend pyav"""