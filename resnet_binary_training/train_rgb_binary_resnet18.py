#!/usr/bin/env python3
"""Train a ResNet18 binary classifier on frame-level labels from a local LeRobot dataset.

Example:
    python train_rgb_binary_resnet18.py \
      --train-root /home/rtalwar/robot-imitation-glue/datasets/delta_xyz_final_rgb_instrumentation \
      --val-root /home/rtalwar/robot-imitation-glue/datasets/delta_xyz_val_rgb_instrumentation \
      --image-key observation.images.wrist_image \
      --label-key btn_state \
      --output-dir /home/rtalwar/robot-imitation-glue/outputs/rgb_binary_resnet18

Notes:
- This script reads frames via LeRobotDataset and expects a binary label per frame.
- It saves `best_model.pt` with `model_state_dict` compatible with the diffusion RGB binary encoder.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Allow running without pip-installing lerobot if this repo layout is used.
REPO_ROOT = Path(__file__).resolve().parent
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


@dataclass
class TrainConfig:
    train_root: str
    val_root: str
    image_key: str
    label_key: str
    output_dir: str
    epochs: int
    batch_size: int
    num_workers: int
    video_backend: str
    max_decode_retries: int
    learning_rate: float
    weight_decay: float
    pos_weight: float | None
    use_imagenet_pretrained: bool
    freeze_backbone: bool
    seed: int


class LeRobotBinaryFrameDataset(Dataset):
    """Wrap a LeRobotDataset to expose (image, binary_label) pairs."""

    def __init__(
        self,
        dataset_root: str | Path,
        image_key: str,
        label_key: str,
        is_train: bool,
        video_backend: str,
        max_decode_retries: int,
    ):
        self.dataset = LeRobotDataset(
            repo_id="local",
            root=str(dataset_root),
            download_videos=False,
            video_backend=video_backend,
        )
        self.image_key = image_key
        self.label_key = label_key
        self.max_decode_retries = max_decode_retries

        if image_key not in self.dataset.features:
            available = ", ".join(sorted(self.dataset.features.keys()))
            raise ValueError(f"image_key '{image_key}' not found. Available keys: {available}")
        if label_key not in self.dataset.features:
            available = ", ".join(sorted(self.dataset.features.keys()))
            raise ValueError(f"label_key '{label_key}' not found. Available keys: {available}")

        # LeRobotDataset decodes all video keys in __getitem__. If the dataset contains additional
        # video streams (e.g., spectrogram video), decoding those can fail even though we only need
        # one RGB key. Restrict video decoding to the selected image key.
        if image_key in self.dataset.meta.video_keys and len(self.dataset.meta.video_keys) > 1:
            filtered_features = {
                key: value
                for key, value in self.dataset.meta.info["features"].items()
                if key not in self.dataset.meta.video_keys or key == image_key
            }
            self.dataset.meta.info["features"] = filtered_features
            warnings.warn(
                "Dataset has multiple video keys; restricted decoding to "
                f"'{image_key}' for binary RGB training.",
                stacklevel=2,
            )

        common = [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                    *common,
                ]
            )
        else:
            self.transform = transforms.Compose(common)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_chw_float_image(image: Tensor) -> Tensor:
        if not isinstance(image, Tensor):
            image = torch.as_tensor(image)

        # Convert HWC -> CHW if needed.
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {tuple(image.shape)}")
        if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image

    @staticmethod
    def _to_scalar_float(label: Tensor) -> float:
        if isinstance(label, Tensor):
            if label.numel() == 0:
                raise ValueError("Label tensor is empty")
            return float(label.reshape(-1)[0].item())
        if isinstance(label, (list, tuple)):
            if len(label) == 0:
                raise ValueError("Label sequence is empty")
            return float(label[0])
        return float(label)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample = None
        last_error = None
        # Retry with a random index to avoid crashing on a corrupted video packet/frame.
        for attempt in range(self.max_decode_retries):
            try:
                sample = self.dataset[idx]
                break
            except RuntimeError as err:
                last_error = err
                idx = random.randint(0, len(self.dataset) - 1)
                if attempt == self.max_decode_retries - 1:
                    raise RuntimeError(
                        "Failed to decode sample after retries. "
                        f"Last error: {type(err).__name__}: {err}"
                    ) from err

        if sample is None:
            raise RuntimeError(f"Sample loading failed unexpectedly. Last error: {last_error}")

        image = self._to_chw_float_image(sample[self.image_key])
        label_value = self._to_scalar_float(sample[self.label_key])

        # Clamp to binary range in case source label is float-like.
        label_value = 1.0 if label_value >= 0.5 else 0.0

        image = self.transform(image)
        label = torch.tensor([label_value], dtype=torch.float32)
        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(use_imagenet_pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model


def compute_pos_weight(dataset: Dataset) -> float:
    positives = 0.0
    total = 0
    for _, label in tqdm(dataset, desc="Counting labels", leave=False):
        positives += float(label.item())
        total += 1

    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return negatives / positives


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = 0

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        correct += (preds == labels).sum().item()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LeRobotBinaryFrameDataset(
        dataset_root=cfg.train_root,
        image_key=cfg.image_key,
        label_key=cfg.label_key,
        is_train=True,
        video_backend=cfg.video_backend,
        max_decode_retries=cfg.max_decode_retries,
    )
    val_ds = LeRobotBinaryFrameDataset(
        dataset_root=cfg.val_root,
        image_key=cfg.image_key,
        label_key=cfg.label_key,
        is_train=False,
        video_backend=cfg.video_backend,
        max_decode_retries=cfg.max_decode_retries,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.use_imagenet_pretrained, cfg.freeze_backbone).to(device)

    if cfg.pos_weight is None:
        computed_pos_weight = compute_pos_weight(train_ds)
    else:
        computed_pos_weight = cfg.pos_weight

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([computed_pos_weight], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = train_loss_sum / max(train_total, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "best_val_f1": best_f1,
                    "image_key": cfg.image_key,
                    "label_key": cfg.label_key,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path} (val_f1={best_f1:.4f})")

    last_ckpt = output_dir / "last_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "best_val_f1": best_f1,
            "image_key": cfg.image_key,
            "label_key": cfg.label_key,
        },
        last_ckpt,
    )

    with (output_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val_f1={best_f1:.4f}")
    print(f"Last checkpoint: {last_ckpt}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet18 binary classifier on LeRobot frames")
    parser.add_argument("--train-root", type=str, required=True, help="Path to training LeRobot dataset root")
    parser.add_argument("--val-root", type=str, required=True, help="Path to validation LeRobot dataset root")
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
        help="Binary label key in LeRobot dataset (e.g. btn_state)",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints/logs")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decoder backend for LeRobotDataset. `pyav` is often more robust with mixed video files.",
    )
    parser.add_argument(
        "--max-decode-retries",
        type=int,
        default=8,
        help="How many times to retry with a different sample if video decoding fails.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive-class weight for BCEWithLogitsLoss. If omitted, computed from train labels.",
    )

    parser.add_argument(
        "--use-imagenet-pretrained",
        action="store_true",
        help="Initialize ResNet18 from ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze all layers except final FC.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return TrainConfig(
        train_root=args.train_root,
        val_root=args.val_root,
        image_key=args.image_key,
        label_key=args.label_key,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        video_backend=args.video_backend,
        max_decode_retries=args.max_decode_retries,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        use_imagenet_pretrained=args.use_imagenet_pretrained,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
