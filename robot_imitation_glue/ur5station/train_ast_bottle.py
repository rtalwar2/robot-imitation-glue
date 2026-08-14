"""Pretrain the bottle experiment's audio encoder on the cap instrumentation signal.

Forked from `train_ast_single.py` (the button experiment, left untouched). Four things differ, and
each one exists for a reason:

1. It trains the policy's own `DiffusionAudioEncoder`, not a bare `ASTForAudioClassification`. The
   whole point of this script is to produce weights that load into the policy, and building the same
   class from the same `DiffusionConfig` makes `load_state_dict(strict=True)` match by construction --
   no prefix remapping, no risk of a silently partial load. It also inherits the encoder's
   normalization formula and the position-embedding resize for free.
2. It reads a prepared LeRobot dataset instead of a HuggingFace hub repo, so "pretrain on the same
   episode subset as the fine-tune" is structural rather than remembered. Point it at the same
   prepared dataset the corresponding training arm uses.
3. The target is the 3-channel continuous cap sensor with an MSE loss, scored by per-channel R2
   against a mean-predictor baseline -- the button script's binary/BCE/f1 setup does not apply.
4. Spectrogram normalization stats are computed on the training split only and **written into the
   checkpoint**, because `DiffusionConfig.audio_norm_mean/std` default to 0.0/1.0 (i.e. no
   normalization at all). If the arm config does not carry these values, the policy will normalize
   audio differently than this pretraining did.

Usage:
    python -m robot_imitation_glue.ur5station.train_ast_bottle \\
        --dataset-root datasets/bottle_experiment/prepared/bottle_9d_100 \\
        --output outputs/pretrain/bottle_audio_100.pt
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionAudioEncoder
from transformers import Trainer, TrainingArguments

AUDIO_KEY = "observation.audio.spectogram_values"
SENSOR_KEY = "bottle_sensor"

# Per-channel operating range, measured as the global min/max of each channel across
# bottle_experiment/sensor_logs/run_{0003,0005,0006}.json -- the three runs recorded under the
# current sensor layout and motion parameters. Runs 0000-0002 used a different layout and are not
# comparable; the earlier constants here were derived from them.
#
# Used instead of the raw 0-3.3V ADC range, which would squeeze the whole useful signal into the
# top fraction of [0,1], and instead of dataset min/max, which would differ between data levels and
# leak. These come from separate calibration runs rather than the demonstration set and are fixed
# before training, so there is no leakage and the scaling is identical at every data level.
#
# Re-derive together with PER_CHANNEL_THRESHOLDS whenever the cap, sensor mounting or opening
# motion changes.
CALIBRATED_RANGE = ((2.96, 3.25), (2.48, 3.29), (2.08, 3.25))

VALIDATION_FRACTION = 0.2


def normalize_sensor(readings: np.ndarray) -> np.ndarray:
    lo = np.array([low for low, _ in CALIBRATED_RANGE], dtype=np.float32)
    hi = np.array([high for _, high in CALIBRATED_RANGE], dtype=np.float32)
    return ((np.asarray(readings, dtype=np.float32) - lo) / (hi - lo)).astype(np.float32)


def spectrogram_from_frame(frame: dict) -> np.ndarray:
    """(time, mel) spectrogram from a prepared-dataset frame.

    Stored as a 3-channel image with the values repeated across channels, so channel 0 is the signal.
    This mirrors what the policy's `_prepare_global_conditioning` does, which is what matters: the
    encoder must see the same tensor here as it will during policy training.
    """
    values = np.asarray(frame[AUDIO_KEY], dtype=np.float32)
    return values[0] if values.ndim == 3 else values


class BottleAudioDataset(Dataset):
    """Raw spectrograms and normalized sensor targets.

    Deliberately does NOT normalize the spectrogram: DiffusionAudioEncoder.forward applies
    (x - mean) / (std * 2) from its config, so normalizing here too would apply it twice, and the
    encoder would then expect a different distribution at policy-training time than it saw here.
    """

    def __init__(self, dataset: LeRobotDataset, frame_indices: list[int]):
        self.dataset = dataset
        self.frame_indices = frame_indices

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> dict:
        frame = self.dataset[self.frame_indices[index]]
        return {
            "input_values": torch.from_numpy(spectrogram_from_frame(frame)),
            "labels": torch.from_numpy(normalize_sensor(frame[SENSOR_KEY])),
        }


class AudioInstrumentationRegressor(nn.Module):
    """The policy's audio encoder plus a throwaway regression head.

    Only `self.encoder` is saved; the head exists to give the encoder a gradient signal and is
    discarded, exactly like the classifier head in the button script.
    """

    def __init__(self, config: DiffusionConfig, n_channels: int):
        super().__init__()
        self.encoder = DiffusionAudioEncoder(config)
        self.head = nn.Linear(self.encoder.feature_dim, n_channels)

    def forward(self, input_values, labels=None):
        predictions = self.head(self.encoder(input_values))
        loss = nn.functional.mse_loss(predictions, labels) if labels is not None else None
        return {"loss": loss, "logits": predictions}


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def compute_metrics(eval_prediction):
    """Per-channel R2 against a mean-predictor baseline, plus the mean across channels.

    R2 rather than raw error because the suitability threshold in the protocol is defined relative to
    a trivial baseline: R2 = 0 is "no better than always predicting the mean", which is the floor a
    modality has to clear to be worth pretraining on.
    """
    predictions = np.asarray(eval_prediction.predictions, dtype=np.float64)
    labels = np.asarray(eval_prediction.label_ids, dtype=np.float64)

    residual = ((labels - predictions) ** 2).sum(axis=0)
    total = ((labels - labels.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(total > 0, 1.0 - residual / np.maximum(total, 1e-12), 0.0)

    metrics = {f"r2_channel_{index}": float(value) for index, value in enumerate(r2)}
    metrics["r2_mean"] = float(r2.mean())
    metrics["mse"] = float(((labels - predictions) ** 2).mean())
    return metrics


def episode_split(dataset: LeRobotDataset, seed: int) -> tuple[list[int], list[int]]:
    """Split by episode, not by frame.

    Adjacent frames within an episode are near-duplicates at 10 Hz, so a frame-level split would put
    almost-identical samples on both sides and report a validation score that is really a training
    score -- which would then feed straight into the suitability decision.
    """
    episodes = list(range(dataset.meta.total_episodes))
    np.random.default_rng(seed).shuffle(episodes)
    n_validation = max(1, round(len(episodes) * VALIDATION_FRACTION))
    validation_episodes = set(episodes[:n_validation])

    train_indices, validation_indices = [], []
    for episode_index in range(dataset.meta.total_episodes):
        bounds = dataset.meta.episodes[episode_index]
        target = validation_indices if episode_index in validation_episodes else train_indices
        target.extend(range(bounds["dataset_from_index"], bounds["dataset_to_index"]))
    return train_indices, validation_indices


def spectrogram_stats(dataset: LeRobotDataset, frame_indices: list[int]) -> tuple[float, float]:
    """Mean/std over the training frames only, streamed rather than stacked.

    The button script did `np.stack(train_ds["input_values"])`, which materialises the whole split in
    RAM to produce two scalars. Welford-style accumulation keeps it constant-memory.
    """
    count = 0
    total = 0.0
    total_squared = 0.0
    for frame_index in frame_indices:
        values = spectrogram_from_frame(dataset[frame_index]).astype(np.float64)
        count += values.size
        total += values.sum()
        total_squared += (values**2).sum()
    mean = total / count
    return float(mean), float(np.sqrt(max(total_squared / count - mean**2, 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    # 1e-5, not the 1e-4 used for the resnet: this finetunes a pretrained transformer, where the AST
    # paper's convention is an order of magnitude lower.
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2025, help="match the training arm's seed")
    parser.add_argument("--audio-backbone", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="skip AudioSet init. Design A starts from generic weights, so leave this off unless you "
        "deliberately want the random-init contrast.",
    )
    parser.add_argument("--wandb-project", default=None, help="omit to disable wandb")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = LeRobotDataset(repo_id=None, root=str(args.dataset_root))
    train_indices, validation_indices = episode_split(dataset, args.seed)
    print(
        f"{dataset.meta.total_episodes} episodes -> {len(train_indices)} train / "
        f"{len(validation_indices)} validation frames"
    )

    mean, std = spectrogram_stats(dataset, train_indices)
    time_dimension = spectrogram_from_frame(dataset[train_indices[0]]).shape[0]
    print(f"spectrogram stats (train only): mean={mean:.6f} std={std:.6f}, time_dimension={time_dimension}")

    config = DiffusionConfig(
        audio_backbone=args.audio_backbone,
        audio_norm_mean=mean,
        audio_norm_std=std,
        time_dimension=time_dimension,
        audio_feature_type="embedding",
        pretrained_audio_weights=not args.from_scratch,
        freeze_audio_encoder=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = AudioInstrumentationRegressor(config, n_channels=len(CALIBRATED_RANGE))

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    elif "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_DISABLED"] = "true"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer = RegressionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output.parent / f"{args.output.stem}_hf"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            load_best_model_at_end=True,
            metric_for_best_model="r2_mean",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=20,
            remove_unused_columns=False,
            seed=args.seed,
            report_to="wandb" if args.wandb_project else "none",
        ),
        train_dataset=BottleAudioDataset(dataset, train_indices),
        eval_dataset=BottleAudioDataset(dataset, validation_indices),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("final validation metrics:", json.dumps(metrics, indent=2, default=float))

    # Everything the policy config must agree with travels alongside the weights, so a mismatch can be
    # asserted rather than discovered from a bad training curve.
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "audio_norm_mean": mean,
            "audio_norm_std": std,
            "time_dimension": time_dimension,
            "audio_backbone": args.audio_backbone,
            "audio_feature_type": "embedding",
            "pretrained_audio_weights": not args.from_scratch,
            "dataset_root": str(args.dataset_root),
            "steps": int(trainer.state.global_step),
            "metrics": {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        },
        args.output,
    )
    print(f"saved encoder to {args.output}")
    print(
        "set these in the arm config: "
        f'"audio_norm_mean": {mean}, "audio_norm_std": {std}, "time_dimension": {time_dimension}, '
        f'"audio_encoder_init_checkpoint": "{args.output}"'
    )


if __name__ == "__main__":
    main()
