"""Evaluate a trained bottle-opening policy with real rollouts.

One command per (checkpoint, appearance condition): --condition is just the label the operator
attaches to the run (the sticker set is a physical property of the bottle; there is no startup
confirmation).

    python -m robot_imitation_glue.ur5station.bottle.eval_bottle \\
        --checkpoint outputs/train/bottle/from_scratch_100/checkpoints/100000/pretrained_model \\
        --condition id --n-rollouts 10

Identical eval path for every arm, by construction:
  - `n_env_action_dims=9` is passed for all arms -- it slices design C's 3 auxiliary
    instrumentation channels off the denoised vector and is a no-op for the 9-dim arms.
  - The observation preprocessor mirrors `prepare_datasets_bottle` exactly: the policy must see at
    rollout time what it saw at training time. observation.state = [tcp_pose(6), ft - ft_bias(6)]
    with the FT bias captured at the home pose each episode, same as collection. (One inherent gap,
    protocol §4.8: training spectrograms passed through AV1 video quantization, live ones do not.)
  - Rollout poses come from a SPLIT's seeded distribution (splits.py, shared with collection).
    Protocol §1.8: OOD varies only the appearance axis -- the pose distribution never changes
    between ID and OOD.

Success: all three cap-sensor channels above their uncovered thresholds for
SUCCESS_SUSTAIN_STEPS consecutive control steps. Sustained rather than instantaneous because
run_0006 shows transient above-threshold excursions before the real transition; a single-step
criterion could score a failed rollout as success. Timeout: protocol §1.8 says 2x the median
demonstration duration -- measured 31.1 s over the 51 collected episodes, hence the 65 s default.

A rollout also ends when the operator presses Enter mid-rollout; the operator is then asked
whether the cap visually opened, and that answer becomes the episode's success label
(success_source "operator"). Reason: a mostly-open bottle whose third sensor channel never
clears is a real success the sensors miss. Sensor/timeout/force-abort endings keep the
sensor-derived label (success_source "sensors").

Every rollout is recorded (successes AND failures -- rollouts are evidence, unlike demonstrations)
and a JSON results file accumulates one row per rollout for the analysis stage.
"""

import argparse
import datetime
import json
import select
import sys
import termios
import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from airo_robots.grippers.hardware.schunk_process import SchunkGripperProcess

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference
from robot_imitation_glue.collect_data_bottle import (
    RETRACT_LIFT_METERS,
    log_observation_to_rerun,
)
from robot_imitation_glue.collect_data_delta import policy_action_to_tcp_pose
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.hardware.bottle_sensor import PER_CHANNEL_THRESHOLDS, is_uncovered
from robot_imitation_glue.ur5station.bottle.splits import (
    POSE_BLACKLIST,
    SPLIT_SEED_OFFSETS,
    split_for_dataset_name,  # noqa: F401  (re-exported convenience)
)
from robot_imitation_glue.ur5station.bottle_station_env import BottleStation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_PATHS = [
    _REPO_ROOT / "airo-mono" / "airo-typing",
    _REPO_ROOT / "airo-mono" / "airo-spatial-algebra",
    _REPO_ROOT / "airo-mono" / "airo-robots",
    _REPO_ROOT / "airo-mono" / "airo-camera-toolkit",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-agents",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-devices",
]
for _path in _LOCAL_PATHS:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# bottle_station_env already put bottle_experiment/ on sys.path.
from calibrate_and_hover_bottle import compute_hover_pose_above_bottle  # noqa: E402
from open_bottle_agent import (  # noqa: E402
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    RANDOM_SEED,
    RETREAT_SPEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    generate_reachable_bottle_poses,
)

N_ENV_ACTION_DIMS = 9  # [delta_xyz(3), rot6d(6)] -- what the robot accepts, for EVERY arm
CONTROL_FREQUENCY_HZ = 10

# All three channels must read uncovered for this many consecutive control steps (0.5 s at 10 Hz).
SUCCESS_SUSTAIN_STEPS = 5

# Abort guard: |drift-corrected force| above this on any axis means the policy is grinding into
# something -- stop the rollout (scored as failure) instead of letting it push harder.
MAX_ABS_FORCE_NEWTONS = 100.0


def drain_pending_input() -> None:
    """Discard keystrokes sitting unread in the terminal buffer (an Enter hammered through an
    earlier gate, extra stops from one rollout). Every gate and every rollout start calls this
    first, so one physical Enter advances exactly one prompt and a buffered newline can never
    silently skip a gate or instantly stop the next rollout."""
    if sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def wait_for_enter(prompt: str) -> None:
    drain_pending_input()
    input(prompt)


def poll_manual_stop() -> bool:
    """Non-blocking check for an operator Enter press (the mid-rollout stop key): consumes the
    line and returns True if one arrived since the last poll, otherwise returns immediately.
    Polled once per control step, so the stop lands within one 0.1 s cycle. No-op when stdin
    is not a terminal -- there is no operator on the other end of a pipe."""
    if not sys.stdin.isatty():
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable:
        sys.stdin.readline()
        return True
    return False


def make_observation_preprocessor():
    """Env observation dict -> the batched tensors the policy's own preprocessor expects.

    Must mirror prepare_datasets_bottle._frame_transform: same keys, same state composition, same
    FT drift correction. The policy-side NormalizerProcessorStep (loaded from the checkpoint)
    handles normalization -- nothing here should normalize beyond the /255 that the dataset's
    video decoding applied at training time.
    """

    def preprocessor(obs: dict) -> dict:
        wrist = torch.from_numpy(np.ascontiguousarray(obs["wrist_image"])).float() / 255.0
        wrist = wrist.permute(2, 0, 1).unsqueeze(0)

        state = np.concatenate(
            (
                np.asarray(obs["robot_pose"], dtype=np.float32),
                np.asarray(obs["ft"], dtype=np.float32) - np.asarray(obs["ft_bias"], dtype=np.float32),
            )
        )
        state = torch.from_numpy(state).unsqueeze(0)

        spectrogram = torch.from_numpy(np.ascontiguousarray(obs["spectogram_values"])).float()
        spectrogram = spectrogram.permute(2, 0, 1).unsqueeze(0)

        return {
            "observation.images.wrist_image": wrist,
            "observation.state": state,
            "observation.audio.spectogram_values": spectrogram,
        }

    return preprocessor


def corrected_force(obs: dict) -> np.ndarray:
    return np.asarray(obs["ft"][:3], dtype=np.float32) - np.asarray(obs["ft_bias"][:3], dtype=np.float32)


def run_rollout(env, agent, recorder, timeout_seconds: float, label: str) -> dict:
    """One policy rollout from the hover pose. Returns the result row for the results file."""
    control_period = 1.0 / CONTROL_FREQUENCY_HZ
    max_steps = int(timeout_seconds * CONTROL_FREQUENCY_HZ)
    all_channels = list(range(len(PER_CHANNEL_THRESHOLDS)))

    agent.reset()
    recorder.start_episode()
    sustained = 0
    outcome = "timeout"
    started = time.time()
    steps = 0

    drain_pending_input()

    for _ in range(max_steps):
        cycle_end = time.time() + control_period

        if poll_manual_stop():
            outcome = "manual_stop"
            break

        obs = env.get_observations()

        if np.abs(corrected_force(obs)).max() > MAX_ABS_FORCE_NEWTONS:
            outcome = "force_abort"
            print(f"[rollout] |force| exceeded {MAX_ABS_FORCE_NEWTONS} N -- aborting")
            break

        action, _used_images, _attn = agent.get_action(obs)
        current_pose = env.get_robot_pose_se3()
        next_pose = policy_action_to_tcp_pose(current_pose, action)
        recorder.record_step(obs, np.asarray(action, dtype=np.float64))
        env.act_tcp(next_pose, time.time() + control_period)
        steps += 1

        log_observation_to_rerun(obs, recording=True, n_episodes=recorder.n_recorded_episodes, label=label)

        # Sustained-uncovered success check, on the reading recorded THIS step.
        if is_uncovered(np.asarray(obs["bottle_sensor"]), all_channels):
            sustained += 1
            if sustained >= SUCCESS_SUSTAIN_STEPS:
                outcome = "success"
                break
        else:
            sustained = 0

        # 10 Hz pacing (best effort -- inference may exceed the period; the residual wait keeps
        # the loop from running faster than the training data's rate).
        remaining = cycle_end - time.time()
        if remaining > 0:
            time.sleep(remaining)

    duration = time.time() - started
    success = outcome == "success"
    success_source = "sensors"

    if outcome == "manual_stop":
        print(f"[rollout] stopped by operator after {steps} steps ({duration:.1f}s)")
        # The stop Enter is consumed by the poll; flush anything typed after it so the verdict
        # prompt can't be answered by a stray keystroke.
        drain_pending_input()
        verdict = input("[rollout] did the cap VISUALLY open? (y = success, anything else = failure): ")
        success = verdict.strip().lower() in ("y", "yes")
        success_source = "operator"

    recorder.set_episode_success(success)
    recorder.save_episode()  # failures too: rollouts are evidence

    final_reading = [float(v) for v in np.asarray(env.get_observations()["bottle_sensor"])]
    print(f"[rollout] {outcome} after {steps} steps ({duration:.1f}s), sensor={np.round(final_reading, 2)}")
    return {
        "outcome": outcome,
        "success": success,
        "success_source": success_source,
        "steps": steps,
        "duration_seconds": round(duration, 2),
        "final_sensor_reading": final_reading,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="the checkpoint's pretrained_model dir")
    parser.add_argument("--condition", required=True, choices=["id", "ood"], help="declared appearance condition")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument(
        "--split",
        default="test",
        choices=list(SPLIT_SEED_OFFSETS),
        help="which split's seeded pose stream to draw rollout poses from. 'test' by default: same "
        "distribution as training (protocol: only appearance distinguishes ID from OOD), but poses "
        "the policy has never executed.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=65.0,
        help="2x the median demonstration duration (measured 31.1s over the 51 collected episodes)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/eval/bottle_results.json"),
        help="JSON file accumulating one row per rollout, across all configs and conditions",
    )
    args = parser.parse_args()

    # e.g. outputs/train/bottle/from_scratch_100/checkpoints/100000/pretrained_model -> from_scratch_100
    config_name = next(
        (p.name for p in args.checkpoint.resolve().parents if p.parent.name == "bottle"),
        args.checkpoint.resolve().parent.name,
    )
    eval_dataset_name = f"eval_bottle_{config_name}_{args.condition}"

    print(f"[eval] config={config_name} condition={args.condition} split={args.split} n={args.n_rollouts}")

    schunk = SchunkGripperProcess(usb_interface="/dev/serial/by-path/pci-0000:00:14.0-usb-0:7:1.0-port0,11,115200,8E1")
    env = BottleStation(
        schunk,
        with_instrumentation=False,
        with_spectogram_model=False,
        with_spectogram=True,  # audio is a policy input; the spectrogram feed must be running
        use_internal_ft=True,
    )

    policy, lerobot_preprocessor, lerobot_postprocessor = make_lerobot_policy_for_inference(str(args.checkpoint))
    agent = LerobotAgent(
        policy,
        lerobot_preprocessor,
        lerobot_postprocessor,
        "cuda",
        make_observation_preprocessor(),
        n_env_action_dims=N_ENV_ACTION_DIMS,
    )

    recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((N_ENV_ACTION_DIMS,), dtype=np.float64),
        root_dataset_dir=Path(f"datasets/bottle_experiment/eval/{eval_dataset_name}"),
        dataset_name=eval_dataset_name,
        fps=CONTROL_FREQUENCY_HZ,
        use_videos=True,
    )
    already_done = recorder.n_recorded_episodes  # resumable: rerun the command to top up

    rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS[args.split])
    bottle_poses = generate_reachable_bottle_poses(
        args.n_rollouts, env.robot, env.robot_right, rng, blacklist=POSE_BLACKLIST[args.split]
    )
    if len(bottle_poses) < args.n_rollouts:
        raise RuntimeError(f"only {len(bottle_poses)}/{args.n_rollouts} poses generated -- widen the sampling box")

    rr.init("robot_imitation_glue_bottle_eval")
    rr.spawn(memory_limit="10GB")
    results = []

    try:
        for rollout_index in range(already_done, args.n_rollouts):
            print(f"\n===== rollout {rollout_index + 1}/{args.n_rollouts} ({config_name}, {args.condition}) =====")
            wait_for_enter("Press Enter to move the LEFT arm home (Ctrl+C to abort)...")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
            # Same per-episode FT drift zeroing as collection: the policy trained on
            # drift-corrected FT, so it must see drift-corrected FT here.
            env.capture_ft_bias()

            tcp_right_pose, _planned_cap_pose = bottle_poses[rollout_index]
            wait_for_enter("Press Enter to move the RIGHT arm to the rollout pose (Ctrl+C to abort)...")
            print("[move] ur_right via the neutral pose")
            env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)

            cap_pose = env.get_bottle_cap_pose()
            cap_normal = cap_pose[:3, 2]
            hover_pose = compute_hover_pose_above_bottle(cap_pose)
            print(f"[move] ur_left to hover above the cap at {np.round(hover_pose[:3, 3], 4)}")
            env.move_robot_to_tcp_pose(hover_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)

            wait_for_enter("Press Enter to hand control to the POLICY (Ctrl+C to abort)...")
            row = run_rollout(
                env,
                agent,
                recorder,
                args.timeout_seconds,
                label=f"ROLLOUT {rollout_index + 1}/{args.n_rollouts} {config_name} {args.condition}",
            )

            # Unrecorded retreat: lift off the cap along its normal, then home.
            lift_pose = env.get_robot_pose_se3().copy()
            lift_pose[:3, 3] = lift_pose[:3, 3] + RETRACT_LIFT_METERS * cap_normal
            env.move_robot_to_tcp_pose(lift_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

            row |= {
                "config": config_name,
                "checkpoint": str(args.checkpoint),
                "condition": args.condition,
                "split": args.split,
                "pose_index": rollout_index,
                "rollout_dataset": eval_dataset_name,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            results.append(row)

            args.results.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(args.results.read_text()) if args.results.exists() else []
            existing.append(row)
            args.results.write_text(json.dumps(existing, indent=2))

            if rollout_index + 1 < args.n_rollouts:
                wait_for_enter("Close the bottle by hand, then press Enter for the next rollout...")
    finally:
        recorder.finish_recording()
        env.close()

    n_success = sum(r["success"] for r in results)
    print(f"\n[eval] {config_name} / {args.condition}: {n_success}/{len(results)} successful this session")
    print(f"[eval] rows appended to {args.results}; rollouts recorded in datasets/bottle_experiment/eval/{eval_dataset_name}")


if __name__ == "__main__":
    main()
