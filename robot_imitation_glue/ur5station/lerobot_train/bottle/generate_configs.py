"""Generate the 16 bottle-experiment training configs (4 arms x 4 data levels).

Written as a generator rather than 16 hand-maintained JSON files because the entire experiment rests
on the arms being identical except where they are meant to differ. Hand-editing 16 files makes silent
drift a matter of time; here every shared value has exactly one definition, and `_assert_arms_differ_
only_in` fails the build if an unintended key ever diverges.

What each arm is:

    from_scratch  random encoders
    generic       ImageNet + AudioSet
    design_a      generic, then finetuned on the instrumentation signal (an increment on `generic`)
    design_c      random encoders, but the denoised vector is [action(9), instrumentation(3)]

design_c differs only by its dataset: lerobot reads the denoised width from the dataset's `action`
feature, so pointing it at the 12-dim prepared dataset is the whole mechanism.

Normalization is matched to whatever each encoder was last trained under, which is why it is an
intended difference rather than a parity violation: `generic` gets ImageNet/AudioSet stats because
that is what its weights expect, while `design_a`'s encoder last saw data during instrumentation
pretraining on this dataset, so it gets dataset stats like the random-init arms.

Usage:
    python -m robot_imitation_glue.ur5station.lerobot_train.bottle.generate_configs \\
        --audio-norm-mean <from pretraining> --audio-norm-std <from pretraining> \\
        --design-a-checkpoint-dir outputs/pretrain
"""

import argparse
import copy
import json
from pathlib import Path

DATA_LEVELS = ("100", "75", "50", "25")
ARMS = ("from_scratch", "generic", "design_a", "design_c")

DATASET_ROOT = "datasets/bottle_experiment/prepared"
AUDIO_BACKBONE = "MIT/ast-finetuned-audioset-10-10-0.4593"
IMAGENET_RESNET18 = "ResNet18_Weights.IMAGENET1K_V1"

# AST's own AudioSet normalization, from the checkpoint's preprocessor_config.json.
AUDIOSET_NORM_MEAN = -4.2677393
AUDIOSET_NORM_STD = 4.5689974

# Keys allowed to differ between arms. Anything else differing is a bug, not a decision.
INTENDED_ARM_DIFFERENCES = {
    "job_name",
    "output_dir",
    "dataset.repo_id",
    "dataset.root",
    "dataset.use_imagenet_stats",
    "policy.pretrained_backbone_weights",
    "policy.pretrained_audio_weights",
    "policy.audio_norm_mean",
    "policy.audio_norm_std",
    "policy.rgb_encoder_init_checkpoint",
    "policy.audio_encoder_init_checkpoint",
    "policy.output_features",  # documentation only; make_policy overwrites it from the dataset
}


def base_config(audio_norm_mean: float, audio_norm_std: float, time_dimension: int, steps: int) -> dict:
    return {
        "dataset": {
            "repo_id": None,
            "root": None,
            "episodes": None,
            # Subsets are separate prepared datasets, not `episodes` lists: EpisodeAwareSampler emits
            # absolute frame indices while DatasetReader.get_item expects relative ones, so a
            # non-prefix `episodes` list raises or silently reads the wrong frames.
            "image_transforms": {
                "enable": True,
                "max_num_transforms": 3,
                "random_order": True,
                "tfs": {
                    "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}},
                    "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}},
                    "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}},
                    "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}},
                    "sharpness": {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
                },
            },
            "revision": None,
            "use_imagenet_stats": False,
            "video_backend": "torchcodec",
        },
        "env": None,
        "job_name": None,
        "output_dir": None,
        "resume": False,
        "seed": 2025,
        "num_workers": 8,
        "batch_size": 64,
        "steps": steps,
        "eval_freq": 0,
        "log_freq": 200,
        # Final checkpoint only: selecting a checkpoint by rollout success would mean real-robot
        # rollouts per checkpoint per config, quietly multiplying the rollout budget.
        "save_checkpoint": True,
        "save_freq": steps,
        "use_policy_training_preset": True,
        "optimizer": {
            "type": "adam",
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "grad_clip_norm": 10.0,
            "betas": [0.95, 0.999],
            "eps": 1e-8,
        },
        "scheduler": {"type": "diffuser", "num_warmup_steps": 500, "name": "cosine"},
        "eval": {"n_episodes": 50, "batch_size": 50, "use_async_envs": False},
        "wandb": {
            "enable": True,
            "disable_artifact": True,
            "project": "instrumented_bottle",
            "entity": "rtalwar",
            "notes": None,
            "run_id": None,
        },
        "policy": {
            "type": "diffusion",
            # Written explicitly so make_policy does not auto-type them: auto-typing would make the
            # audio key STATE/MIN_MAX instead of AUDIO/IDENTITY, and would silently normalize the
            # spectrogram.
            "input_features": {
                "observation.images.wrist_image": {"type": "VISUAL", "shape": [3, 240, 320]},
                "observation.state": {"type": "STATE", "shape": [12]},
                "observation.audio.spectogram_values": {"type": "AUDIO", "shape": [3, time_dimension, 128]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [9]}},
            "n_obs_steps": 2,
            "horizon": 16,
            "n_action_steps": 8,
            "drop_n_last_frames": 7,
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MIN_MAX",
                "ACTION": "MIN_MAX",
                "AUDIO": "IDENTITY",
            },
            "vision_backbone": "resnet18",
            "resize_shape": None,
            "crop_ratio": None,
            "crop_shape": [216, 288],
            "crop_is_random": True,
            "pretrained_backbone_weights": None,
            # false in every arm: the encoder raises if GroupNorm is combined with torchvision
            # pretrained weights, and flipping it for the ImageNet arm alone would leave that arm the
            # only one using BatchNorm -- an architecture difference confounded with initialization.
            "use_group_norm": False,
            "spatial_softmax_num_keypoints": 32,
            "use_separate_rgb_encoder_per_camera": True,
            "rgb_binary_backbone": None,
            "audio_backbone": AUDIO_BACKBONE,
            "audio_norm_mean": audio_norm_mean,
            "audio_norm_std": audio_norm_std,
            "audio_feature_type": "embedding",
            "time_dimension": time_dimension,
            # false in every arm: the default freezes the AudioSet encoder while auto-unfreezing the
            # random-init one, which would make "all encoders finetuned" false for exactly one arm.
            "freeze_audio_encoder": False,
            "add_intermediate_audio_layer": False,
            "pretrained_audio_weights": False,
            "rgb_encoder_init_checkpoint": None,
            "audio_encoder_init_checkpoint": None,
            "down_dims": [512, 1024, 2048],
            "kernel_size": 5,
            "n_groups": 8,
            "diffusion_step_embed_dim": 128,
            "use_film_scale_modulation": True,
            "noise_scheduler_type": "DDIM",
            "num_train_timesteps": 100,
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            # epsilon-prediction is what makes design C's loss weighting free: every channel's target
            # is unit Gaussian noise, so the 3 instrumentation dims take 3/12 of the objective with no
            # weight to tune.
            "prediction_type": "epsilon",
            "clip_sample": True,
            "clip_sample_range": 1.0,
            "num_inference_steps": None,
            "do_mask_loss_for_padding": False,
            "optimizer_lr": 1e-4,
            "optimizer_betas": [0.95, 0.999],
            "optimizer_eps": 1e-8,
            "optimizer_weight_decay": 1e-6,
            "scheduler_name": "cosine",
            "scheduler_warmup_steps": 500,
        },
    }


def apply_arm(config: dict, arm: str, level: str, checkpoint_dir: Path | None) -> dict:
    config = copy.deepcopy(config)
    action_dims = 12 if arm == "design_c" else 9
    dataset_name = f"bottle_{action_dims}d_{level}"

    config["dataset"]["repo_id"] = dataset_name
    config["dataset"]["root"] = f"{DATASET_ROOT}/{dataset_name}/"
    config["policy"]["output_features"]["action"]["shape"] = [action_dims]
    config["job_name"] = f"bottle_{arm}_{level}"
    config["output_dir"] = f"outputs/train/bottle/{arm}_{level}"

    if arm == "generic":
        config["policy"]["pretrained_backbone_weights"] = IMAGENET_RESNET18
        config["policy"]["pretrained_audio_weights"] = True
        # Matched to the weights: these encoders expect their pretraining distribution.
        config["dataset"]["use_imagenet_stats"] = True
        config["policy"]["audio_norm_mean"] = AUDIOSET_NORM_MEAN
        config["policy"]["audio_norm_std"] = AUDIOSET_NORM_STD

    if arm == "design_a":
        if checkpoint_dir is None:
            raise ValueError("--design-a-checkpoint-dir is required to emit the design_a configs")
        # The checkpoint already holds ImageNet-derived, instrumentation-finetuned weights, so
        # pretrained_backbone_weights stays None -- torchvision's would just be overwritten.
        config["policy"]["rgb_encoder_init_checkpoint"] = str(checkpoint_dir / f"bottle_rgb_{level}.pt")
        config["policy"]["audio_encoder_init_checkpoint"] = str(checkpoint_dir / f"bottle_audio_{level}.pt")

    return config


def _flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= _flatten(value, f"{path}.")
        else:
            flat[path] = value
    return flat


def _assert_arms_differ_only_in(configs: dict[str, dict], allowed: set[str]) -> None:
    """The parity check. Anything differing outside `allowed` invalidates the comparison."""
    flattened = {arm: _flatten(config) for arm, config in configs.items()}
    reference_arm, reference = next(iter(flattened.items()))
    unexpected = {}
    for arm, flat in flattened.items():
        if arm == reference_arm:
            continue
        for key in set(reference) | set(flat):
            if reference.get(key) != flat.get(key) and not any(key.startswith(prefix) for prefix in allowed):
                unexpected.setdefault(key, {})[arm] = (reference.get(key), flat.get(key))
    if unexpected:
        raise AssertionError(f"arms differ in unintended keys: {json.dumps(unexpected, indent=2, default=str)}")
    print(f"parity check: arms differ only in {len(allowed)} intended keys")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-norm-mean",
        type=float,
        required=True,
        help="spectrogram mean over the training split, printed by train_ast_bottle.py. The config "
        "default is 0.0, which means NO normalization -- this must be set explicitly.",
    )
    parser.add_argument("--audio-norm-std", type=float, required=True)
    parser.add_argument("--time-dimension", type=int, default=298)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--design-a-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    base = base_config(args.audio_norm_mean, args.audio_norm_std, args.time_dimension, args.steps)

    # Check parity on one level; the level only changes the dataset path.
    _assert_arms_differ_only_in(
        {arm: apply_arm(base, arm, "100", args.design_a_checkpoint_dir) for arm in ARMS},
        INTENDED_ARM_DIFFERENCES,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for level in DATA_LEVELS:
            config = apply_arm(base, arm, level, args.design_a_checkpoint_dir)
            path = args.output_dir / f"bottle_{arm}_{level}.json"
            path.write_text(json.dumps(config, indent=2))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
