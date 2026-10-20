"""Evaluate a trained bottle-opening policy with real rollouts.

One command per (checkpoint, appearance condition): --condition is just the label the operator
attaches to the run (the sticker set is a physical property of the bottle; there is no startup
confirmation).

    python -m robot_imitation_glue.ur5station.bottle.eval_bottle \\
        --checkpoint outputs/train/bottle/generic_100/checkpoints/100000/pretrained_model \\
        --condition id --n-rollouts 20

Identical eval path for every arm, by construction:
  - `n_env_action_dims=9` is passed for all arms -- it slices the treatment arm's (generic_c)
    3 auxiliary instrumentation channels off the denoised vector and is a no-op for the 9-dim arms.
  - The observation preprocessor mirrors `prepare_datasets_bottle` exactly: the policy must see at
    rollout time what it saw at training time. observation.state = [tcp_pose(6), ft - ft_bias(6)]
    with the FT bias captured at the home pose each episode, same as collection. (One inherent gap,
    protocol §4.8: training spectrograms passed through AV1 video quantization, live ones do not.)
  - Rollout poses come from a SPLIT's seeded distribution (splits.py, shared with collection).
    Protocol §1.8: OOD varies only the appearance axis -- the pose distribution never changes
    between ID and OOD.

Success: the deciding cap-sensor channels (OPEN_CHANNELS = S1, S2; S0 cannot separate closed from open,
see hardware/bottle_sensor.py) at or above their thresholds for SUCCESS_SUSTAIN_STEPS consecutive control
steps. Sustained rather than instantaneous because
run_0006 shows transient above-threshold excursions before the real transition; a single-step
criterion could score a failed rollout as success. Timeout: protocol §1.8 says 2x the median
demonstration duration rounded up to the next 5 s -- provisionally 2x31.1 s = 62.2 -> 65 s default
(the 31.1 s was measured over the old 51-episode set, which is being recollected; re-derive it per
the operator guide and pass --timeout-seconds explicitly).

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

from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference
from robot_imitation_glue.collect_data_bottle import (
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_OPEN_WIDTH_M,
    RETRACT_LIFT_METERS,
    _move_gripper_to,
    log_observation_to_rerun,
)
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.hardware.bottle_sensor import is_cap_open
from robot_imitation_glue.hardware.gripper_startup import GRIPPER_PORT_RIGHT, make_gripper
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

# The deciding channels (S1, S2) must read open for this many consecutive control steps (0.5 s at 10 Hz).
SUCCESS_SUSTAIN_STEPS = 5

# Abort guard: |drift-corrected force| above this on any axis means the policy is grinding into
# something -- stop the rollout (scored as failure) instead of letting it push harder.
# Twice the p95 per-episode peak of the 110 training episodes, expert + recovery (48.0 N; median 22.5,
# mean 26.8, max 140.8 N). Was 52 N (2x the 90 expert demos' mean peak) until 2026-10-16: that cut at
# demo-level pushing -- 4 of the 110 successful training episodes exceed it -- so it scored force rather
# than failure; at 96 N it is a safety stop and success is decided by the sensors / timeout. The peak force
# is in every eval dataset, so the 52 N would-have-aborted rate stays reportable as a secondary metric.
MAX_ABS_FORCE_NEWTONS = 96.0
# Largest drift-corrected force in the demos' first 12 steps (free air, descending from hover): 26.7 N.
FREE_AIR_FORCE_LIMIT_NEWTONS = 27.0
# Mean wrist-image brightness (0-1) at hover in the curve training data (bottle_9d_100, 70 episodes): p2.5-p97.5.
# The camera runs auto-exposure, which cannot compensate strong daylight: the 2026-10-16/19 curve evals ran in two
# lighting regimes (whole-rollout mean 0.53-0.57 with 15-25% saturated pixels vs 0.39-0.40 and the training data's
# 0.38), confounded with the arm. Logged per rollout so lighting can be checked as a covariate; control the room
# lighting instead of relying on this warning.
HOVER_BRIGHTNESS_RANGE = (0.326, 0.618)


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


def retreat_left_arm(env, cap_normal) -> None:
    """Lift off the cap along its normal, then go home; after a protective stop, let the operator clear it."""
    while True:
        try:
            if not env.robot.rtde_control.isProgramRunning():
                wait_for_enter("[retreat] the UR control script is not running: clear the protective stop on the "
                               "LEFT arm's pendant, then press Enter (Ctrl+C to abort)...")
                env.robot.rtde_control.reuploadScript()
                time.sleep(1.0)
            lift_pose = env.get_robot_pose_se3().copy()
            lift_pose[:3, 3] = lift_pose[:3, 3] + RETRACT_LIFT_METERS * cap_normal
            env.move_robot_to_tcp_pose(lift_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
            return
        except Exception as error:  # noqa: BLE001 -- any failure here is handed to the operator
            print(f"[retreat] failed: {error}")
            wait_for_enter("[retreat] fix the LEFT arm (pendant / freedrive it clear of the cap), then press Enter to retry...")


def run_rollout(env, agent, recorder, timeout_seconds: float, label: str) -> dict:
    """One policy rollout from the hover pose. Returns the result row for the results file."""
    control_period = 1.0 / CONTROL_FREQUENCY_HZ
    max_steps = int(timeout_seconds * CONTROL_FREQUENCY_HZ)

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

        # A protective stop kills the UR's control script; every servo command after that is ignored
        # ("RTDE control script is not running!") while the policy keeps "acting" -- end the rollout.
        if not env.robot.rtde_control.isProgramRunning():
            outcome = "robot_stopped"
            print("[rollout] the UR control script stopped (protective stop?) -- ending the rollout")
            break

        obs = env.get_observations()

        if np.abs(corrected_force(obs)).max() > MAX_ABS_FORCE_NEWTONS:
            outcome = "force_abort"
            print(f"[rollout] |force| exceeded {MAX_ABS_FORCE_NEWTONS} N -- aborting")
            break

        action, _used_images, _attn = agent.get_action(obs)
        current_pose = env.get_robot_pose_se3()
        next_pose = agent.action_to_tcp_pose(action, current_pose)  # delta or absolute target, per the policy
        recorder.record_step(obs, np.asarray(action, dtype=np.float64))
        env.act_tcp(next_pose, time.time() + control_period)
        steps += 1

        log_observation_to_rerun(obs, recording=True, n_episodes=recorder.n_recorded_episodes, label=label)

        # Sustained-open success check (the deciding channels, OPEN_CHANNELS), on the reading recorded THIS step.
        if is_cap_open(np.asarray(obs["bottle_sensor"])):
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

    if outcome in ("manual_stop", "robot_stopped"):
        print(f"[rollout] {outcome} after {steps} steps ({duration:.1f}s)")
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
        help="2x the median demonstration duration, rounded up to the next 5 s (65 s, fixed 2026-10-14 with "
        "N = 70; use the same value for every config and condition)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/eval/bottle_results.json"),
        help="JSON file accumulating one row per rollout, across all configs and conditions",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="override the checkpoint's DDIM denoising steps (default: its config, i.e. num_train_timesteps=100, "
        "~440 ms per inference on the 4090; 16 steps ~70 ms). Changes the eval condition: tagged into the "
        "config name, so it gets its own eval dataset and results rows",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="override how many actions of each predicted chunk are executed before re-planning (checkpoint: 8). "
        "1 = re-plan every control step; only fits the 10 Hz loop with a short --num-inference-steps",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="free-form suffix for a changed eval condition (e.g. 'caprot'), appended to the config name so "
        "those rollouts get their own eval dataset and results rows",
    )
    args = parser.parse_args()

    # e.g. outputs/train/bottle/from_scratch_100/checkpoints/100000/pretrained_model -> from_scratch_100
    config_name = next(
        (p.name for p in args.checkpoint.resolve().parents if p.parent.name == "bottle"),
        args.checkpoint.resolve().parent.name,
    )
    if args.num_inference_steps is not None:
        config_name += f"_ddim{args.num_inference_steps}"
    if args.n_action_steps is not None:
        config_name += f"_a{args.n_action_steps}"
    if args.tag:
        config_name += f"_{args.tag}"
    eval_dataset_name = f"eval_bottle_{config_name}_{args.condition}"

    print(f"[eval] config={config_name} condition={args.condition} split={args.split} n={args.n_rollouts}")

    schunk = make_gripper(GRIPPER_PORT_RIGHT)
    env = BottleStation(
        schunk,
        with_instrumentation=False,
        with_spectogram_model=False,
        with_spectogram=True,  # audio is a policy input; the spectrogram feed must be running
        use_internal_ft=True,
    )

    policy, lerobot_preprocessor, lerobot_postprocessor = make_lerobot_policy_for_inference(str(args.checkpoint))
    if args.num_inference_steps is not None:
        policy.config.num_inference_steps = args.num_inference_steps
        policy.diffusion.num_inference_steps = args.num_inference_steps  # what conditional_sample reads
    if args.n_action_steps is not None:
        horizon, n_obs = policy.config.horizon, policy.config.n_obs_steps
        if not 1 <= args.n_action_steps <= horizon - n_obs + 1:
            raise ValueError(f"--n-action-steps must be in [1, {horizon - n_obs + 1}] (horizon {horizon}, n_obs_steps {n_obs})")
        policy.config.n_action_steps = args.n_action_steps
        policy.reset()  # the action queue's maxlen is taken from the config in reset()
    print(f"[eval] inference: {policy.diffusion.num_inference_steps} DDIM steps, "
          f"{policy.config.n_action_steps} action(s) executed per prediction")
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
    # generate_reachable_bottle_poses numbers every constraint-passing pose and skips the blacklisted ones,
    # so position k in bottle_poses is the k-th index NOT in the blacklist.
    blacklisted = set(POSE_BLACKLIST[args.split])
    pose_indices = [i for i in range(args.n_rollouts + len(blacklisted)) if i not in blacklisted][: args.n_rollouts]
    if len(bottle_poses) < args.n_rollouts:
        raise RuntimeError(f"only {len(bottle_poses)}/{args.n_rollouts} poses generated -- widen the sampling box")

    rr.init("robot_imitation_glue_bottle_eval")
    rr.spawn(memory_limit="10GB")
    results = []

    try:
        for rollout_index in range(already_done, args.n_rollouts):
            print(f"\n===== rollout {rollout_index + 1}/{args.n_rollouts} ({config_name}, {args.condition}) -- "
                  f"pose index {pose_indices[rollout_index]} (the number to put in POSE_BLACKLIST) =====")
            wait_for_enter("Press Enter to move the LEFT arm home (Ctrl+C to abort)...")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
            # Re-zero the UR's FT sensor first: on 2026-10-12 its offset had grown to ~40 N (home bias
            # [32.9, -19.6, 7.1] vs ~[7, 3, 0] in the demos), and because that offset is fixed in the
            # SENSOR frame it changed by ~42 N between home and hover -- a force-abort in free air.
            env.robot.rtde_control.zeroFtSensor()
            # Same per-episode FT drift zeroing as collection: the policy trained on
            # drift-corrected FT, so it must see drift-corrected FT here.
            env.capture_ft_bias()

            tcp_right_pose, _planned_cap_pose = bottle_poses[rollout_index]
            # Same re-grasp as every collected episode (collect_data_bottle): at the neutral pose the fingers
            # open, the bottle is handed back, then ur_right moves to the pose and closes on it. Without this
            # the bottle would keep one grasp for all rollouts -- a presentation the policy never trained on.
            wait_for_enter("Press Enter to move the RIGHT arm to its neutral pose and open the gripper (Ctrl+C to abort)...")
            print("[move] ur_right to the neutral pose")
            env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            print(f"[gripper] ur_right opening to {GRIPPER_OPEN_WIDTH_M:.3f} m")
            # _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
            wait_for_enter("Close the cap, hold the bottle between the open fingers, then press Enter -- ur_right "
                           "moves to the rollout pose and closes (Ctrl+C to abort)...")
            env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
            print(f"[gripper] ur_right closing on the bottle at {GRIPPER_GRASP_WIDTH_M:.3f} m")
            # _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)

            cap_pose = env.get_bottle_cap_pose()
            cap_normal = cap_pose[:3, 2]
            hover_pose = compute_hover_pose_above_bottle(cap_pose)
            print(f"[move] ur_left to hover above the cap at {np.round(hover_pose[:3, 3], 4)}")
            env.move_robot_to_tcp_pose(hover_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)

            time.sleep(0.5)  # let the arm stop ringing
            hover_force = np.abs(corrected_force(env.get_observations())).max()
            print(f"[ft] free-air force at hover: {hover_force:.1f} N (demos: <= {FREE_AIR_FORCE_LIMIT_NEWTONS} N)")
            if hover_force > FREE_AIR_FORCE_LIMIT_NEWTONS:
                print("[ft] WARNING: the FT reading is off in free air -- the policy would see out-of-distribution force "
                      "and the rollout may force-abort. Check the payload on the pendant, then Ctrl+C and restart.")
            hover_brightness = float(np.asarray(env.get_observations()["wrist_image"]).mean() / 255.0)
            print(f"[camera] wrist-image brightness at hover: {hover_brightness:.3f} "
                  f"(training p2.5-p97.5: {HOVER_BRIGHTNESS_RANGE[0]}-{HOVER_BRIGHTNESS_RANGE[1]})")
            if not HOVER_BRIGHTNESS_RANGE[0] <= hover_brightness <= HOVER_BRIGHTNESS_RANGE[1]:
                print("[camera] WARNING: the scene is brighter/darker than any training episode -- check the room "
                      "lighting (blinds, lamps) before this rollout.")
            wait_for_enter("Press Enter to hand control to the POLICY (Ctrl+C to abort)...")
            row = run_rollout(
                env,
                agent,
                recorder,
                args.timeout_seconds,
                label=f"ROLLOUT {rollout_index + 1}/{args.n_rollouts} {config_name} {args.condition}",
            )

            row |= {
                "config": config_name,
                "checkpoint": str(args.checkpoint),
                "num_inference_steps": int(policy.diffusion.num_inference_steps),
                "n_action_steps": int(policy.config.n_action_steps),
                "condition": args.condition,
                "split": args.split,
                "pose_index": rollout_index,
                "hover_brightness": round(hover_brightness, 4),
                "rollout_dataset": eval_dataset_name,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            results.append(row)

            args.results.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(args.results.read_text()) if args.results.exists() else []
            existing.append(row)
            args.results.write_text(json.dumps(existing, indent=2))

            # Unrecorded retreat: lift off the cap along its normal, then home. AFTER the results row is
            # written, so a robot that cannot retreat (protective stop) never loses a saved rollout's row.
            retreat_left_arm(env, cap_normal)

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
