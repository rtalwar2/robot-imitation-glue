"""Generate the bottle-experiment training configs: two arms plus the init-verification cell.

Rewritten as of 2026-10-02 (branch `ral_simplified`): the four-arm matrix (from_scratch, generic,
design_a, design_c) was simplified to a single-factor comparison. Written as a generator rather
than hand-maintained JSON files because the entire experiment rests on the arms being identical
except where they are meant to differ; every shared value has exactly one definition, and
`_assert_arms_differ_only_in` fails the build if an unintended key ever diverges.

What each arm is:

    generic       CONTROL. ImageNet/AudioSet initialization -- the framework default everyone
                  trains from (upstream lerobot's DiffusionPolicy ships `pretrained_backbone_weights
                  = "ResNet18_Weights.IMAGENET1K_V1"`). Denoised vector = action(9).
    generic_c     TREATMENT. Identical to generic except the dataset's 12-dim action =
                  [action(9), instrumentation(3)]; lerobot derives the denoised width from the
                  dataset's `action` feature, so that single change IS the whole mechanism. The
                  instrumentation is never a policy input and is sliced off at inference -- the
                  deployed policy needs no sensor hardware. generic vs generic_c therefore
                  isolates sensor supervision exactly; no second factor is ever in play.
    from_scratch  VERIFICATION CELL, not a live arm. Random encoders, action(9). Trained and
                  evaluated once, at the 100% level only, to check the assumption that standard
                  pretrained init is never worse than random on this task family (it is an
                  assumption worth checking: ManiWAV's audio ablation preferred from-scratch ASTs
                  on contact audio, though that comparison confounds architecture with init).
                  Pre-registered one-sided rule: it is dropped from the design unless it ties or
                  beats `generic` at 100% ID -- a loss (or parity) is reported as the appendix
                  init-verification row, only a win promotes it to a live arm.

Why the matrix was simplified: the design-A (encoder pretraining) vs design-C (auxiliary
prediction) mechanism comparison is retired -- the mechanism itself comes from the colleague's
workshop paper, and running A would spend four trainings plus 160 rollouts on a comparison that is
explicitly not the headline. The workshop's C arm also differed from its baselines in two factors
at once (random init AND auxiliary channels); placing the mechanism on the standard pretrained
init makes this paper's comparison single-factor and de-confounds it.

Two rules this generator still enforces:

**Audio normalization for random-init arms is per LEVEL, never global.** `audio_norm_mean/std`
for `from_scratch` come from that level's own `audio_stats.json` (written by
prepare_datasets_bottle over exactly the episodes the level trains on). A single global value --
e.g. the 100% run's stats reused at 25% -- hands a run statistics from episodes it never sees.
The generic arms load AudioSet's own normalization constants, which are dataset-independent, so
they have no leak surface; the rule exists to protect the random-init cell and survives the
simplification unchanged.

**Nothing is hand-edited into a generated config.** Change the generator, re-run, keep the parity
assertion green.

Usage:
    # the eight curve configs (2 arms x 4 levels):
    python -m robot_imitation_glue.ur5station.bottle.generate_configs

    # the verification cell (100% only, per the pre-registered rule):
    python -m robot_imitation_glue.ur5station.bottle.generate_configs --arms from_scratch --levels 100
"""

import argparse
import copy
import json
from pathlib import Path

DATA_LEVELS = ("100", "75", "50", "25")
ARMS = ("generic", "generic_c", "from_scratch")

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
    "policy.repo_id",
    "dataset.repo_id",
    "dataset.root",
    "dataset.use_imagenet_stats",
    "policy.pretrained_backbone_weights",
    "policy.pretrained_audio_weights",
    "policy.audio_norm_mean",
    "policy.audio_norm_std",
    "policy.output_features",  # documentation only; make_policy overwrites it from the dataset
}


def load_level_audio_stats(level: str) -> dict:
    """That level's spectrogram mean/std/time_dimension, from prepare_datasets_bottle's stats pass.

    Computed over exactly the episodes the level trains on -- see the leakage note in the module
    docstring. The 9d and 12d datasets of a level carry identical copies; read the 9d one.
    """
    path = Path(DATASET_ROOT) / f"bottle_9d_{level}" / "audio_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m robot_imitation_glue.ur5station.bottle."
            f"prepare_datasets_bottle --stats-only` first (per-level audio stats are required; "
            "a global value would leak across data levels)"
        )
    return json.loads(path.read_text())


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
        # 64 OOM'd on a 24GB card: the unfrozen AST plus the resnet18 RGB encoder plus the diffusion
        # U-Net (down_dims up to 2048), all trained end-to-end, left no headroom at that batch size.
        "batch_size": 32,
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
            # DiffusionConfig types this as a plain `float`, not `float | None` (unlike
            # resize_shape/crop_shape) -- draccus rejects null for it. Unused anyway: it only
            # takes effect when resize_shape is set, and this config crops directly via
            # crop_shape with resize_shape left None, so 1.0 (the dataclass default, meaning
            # "no additional ratio-derived cropping") is the correct no-op value here.
            "crop_ratio": 1.0,
            "crop_shape": [216, 288],
            "crop_is_random": True,
            "pretrained_backbone_weights": None,
            # false in every arm: the encoder raises if GroupNorm is combined with torchvision
            # pretrained weights, and flipping it for the generic arms alone would leave them the
            # only ones using BatchNorm -- an architecture difference confounded with initialization.
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
            # epsilon-prediction is what makes the auxiliary channels' loss weighting free: every
            # channel's target is unit Gaussian noise, so the 3 instrumentation dims of the
            # treatment arm take 3/12 of the objective with no weight to tune.
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


def apply_arm(config: dict, arm: str, level: str) -> dict:
    config = copy.deepcopy(config)
    action_dims = 12 if arm == "generic_c" else 9
    dataset_name = f"bottle_{action_dims}d_{level}"

    config["dataset"]["repo_id"] = dataset_name
    config["dataset"]["root"] = f"{DATASET_ROOT}/{dataset_name}/"
    config["policy"]["output_features"]["action"]["shape"] = [action_dims]
    config["job_name"] = f"bottle_{arm}_{level}"
    config["output_dir"] = f"outputs/train/bottle/{arm}_{level}"
    # PreTrainedConfig.push_to_hub defaults to True and then requires repo_id -- same HF
    # namespace as the button experiment's checkpoints (ramen-noodels/red_round_button_*).
    config["policy"]["repo_id"] = f"ramen-noodels/bottle_{arm}_{level}"

    if arm in ("generic", "generic_c"):
        config["policy"]["pretrained_backbone_weights"] = IMAGENET_RESNET18
        config["policy"]["pretrained_audio_weights"] = True
        # Matched to the weights: these encoders expect their pretraining distribution. This is
        # also why the generic arms carry no per-level stats leak surface -- AudioSet's constants
        # know nothing about this task's episodes.
        config["dataset"]["use_imagenet_stats"] = True
        config["policy"]["audio_norm_mean"] = AUDIOSET_NORM_MEAN
        config["policy"]["audio_norm_std"] = AUDIOSET_NORM_STD

    return config


def _flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= _flatten(config=value, prefix=f"{path}.")
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
        "--arms",
        default="generic,generic_c",
        help="comma-separated subset of {generic, generic_c, from_scratch}; the curve defaults to "
        "control + treatment, from_scratch is the verification cell (run it with --levels 100)",
    )
    parser.add_argument("--levels", default=",".join(DATA_LEVELS), help="comma-separated subset of 100,75,50,25")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "configs")
    args = parser.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {ARMS}")
    levels = tuple(entry.strip() for entry in args.levels.split(",") if entry.strip())
    unknown_levels = set(levels) - set(DATA_LEVELS)
    if unknown_levels:
        raise SystemExit(f"unknown levels {sorted(unknown_levels)}; choose from {DATA_LEVELS}")

    # Build EVERYTHING before writing ANYTHING: a missing stats file at level 75 must not leave a
    # freshly written level-100 config behind, or a partial (and easily stale) set gets trained.
    to_write = {}
    for level in DATA_LEVELS:
        if level not in levels:
            continue
        # Per-LEVEL base: audio normalization for the random-init cell comes from that level's own
        # episodes. One base for all arms at a level, so the parity check below compares like with
        # like.
        stats = load_level_audio_stats(level)
        base = base_config(stats["audio_norm_mean"], stats["audio_norm_std"], stats["time_dimension"], args.steps)

        # Parity check at every level, always against from_scratch as the reference -- it needs
        # nothing but the stats file, so it is constructible even when only the generic arms are
        # requested.
        built = {arm: apply_arm(base, arm, level) for arm in dict.fromkeys(("from_scratch", *arms))}
        _assert_arms_differ_only_in(built, INTENDED_ARM_DIFFERENCES)
        for arm in arms:
            to_write[f"bottle_{arm}_{level}.json"] = built[arm]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in to_write.items():
        path = args.output_dir / name
        path.write_text(json.dumps(config, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
