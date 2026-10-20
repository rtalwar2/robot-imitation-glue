"""Recovery demonstrations: put the gripper in the state the policy fails from, then demonstrate the way out.

    python -m robot_imitation_glue.ur5station.bottle.collect_recovery_bottle
    python -m robot_imitation_glue.ur5station.bottle.collect_recovery_bottle --n-episodes 20 --record-failure

Why (eval rollouts of 2026-10-12/13, 50 with first contact): the rollouts fail at the first push. A rollout
that reaches the grip point too deep -- median 129.8 mm below hover in the failures vs 125.4 mm in the
successes and exactly 126 mm (APPROACH_OFFSET) in every demo -- presses its fingers on the cap body
below the lid's lip: the push force climbs to 20-38 N, the lid sensors never move, and the policy keeps
pushing (40-50 mm) until the operator stops it. The successes catch the lip and the lid starts lifting
within 10-22 mm of push at ~19 N. The demos never show what to do from a jammed, too-deep push, so the
policy has nothing to imitate there.

Each episode:
  1. the same setup as collect_data_bottle_opening: home, FT bias, right arm to a train pose, re-grasp,
     hover, touch point (detector, click window when unsure), plan, Enter;
  2. the failure, NOT recorded: the demo's descent to the grip point, sinking DEPTH_EXCESS_M deeper as the\n     gripper finishes its yaw, then push
     PUSH_M along the planned push direction (stops early at JAM_FORCE_N). Both are drawn per episode;
     a push of 0 is a too-deep grip point caught before the push;
  3. the recovery, recorded: lift off along the cap normal to CLEARANCE_M above the planned grip depth,
     go back over the grip point, then the ordinary demo -- descend to the planned depth, push, legs,
     stop-on-open, checkpoint retries -- and the usual retreat to hover.
Episodes whose recovery does not open the cap are discarded, as in the expert collector. If the lid
opens during the induced failure it was not a failure: nothing was recorded, close the cap and go again.

--policy CHECKPOINT replaces step 2 with a rollout of a trained policy from the hover (default 10 DDIM steps,
1 action per prediction): the failure is then the policy's own. The operator presses Enter as soon as it
goes wrong (too deep, jammed); JAM_FORCE_N and POLICY_TIMEOUT_S stop it too. If the policy opens the cap
there is nothing to recover: the episode is discarded. The recovery (step 3) starts wherever it stopped.

Recording starts at the jam (step 3) by default: the frames of step 2 would teach the policy the very
descend-too-deep-and-keep-pushing it should unlearn. --record-failure records from the hover instead
(for masking those frames at dataset-prep time); the per-episode recovery_log.json says where the
recovery starts either way.

Writes to datasets/bottle_experiment/expert/bottle_opening_train_recovery (train split poses, so the
policy's pose distribution is unchanged), plus recovery_log.json there: per saved episode the pose index,
drawn depth excess and push, the push actually done, the force at the jam and the recovery's first frame.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation
from airo_camera_toolkit.utils.image_converter import ImageConverter

from robot_imitation_glue.collect_data_bottle import (
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_OPEN_WIDTH_M,
    _lift_off_cap,
    _move_gripper_to,
    _pose_reached,
    _retreat_to_hover,
    _tool_frame_step,
    log_idle_to_rerun,
    run_opening_motion_with_retry,
)
from robot_imitation_glue import collect_data_bottle
from robot_imitation_glue.collect_data_delta import policy_action_to_tcp_pose, step_action_to_policy_action_6d
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.hardware.bottle_sensor import channels_open, is_cap_open
from robot_imitation_glue.hardware.gripper_startup import GRIPPER_PORT_RIGHT, make_gripper
from robot_imitation_glue.ur5station.bottle.splits import POSE_BLACKLIST, SPLIT_N_POSES, SPLIT_SEED_OFFSETS
from robot_imitation_glue.ur5station.bottle_station_env import BottleStation
from robot_imitation_glue.utils import precise_wait

# bottle_station_env already put bottle_experiment/ on sys.path.
from calibrate_and_hover_bottle import compute_hover_pose_above_bottle, tcp_left_to_camera  # noqa: E402
from open_bottle_agent import (  # noqa: E402
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    RANDOM_SEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    generate_reachable_bottle_poses,
    log_plan_to_rerun,
    plan_opening_motion,
)
from open_bottle_demo import pixel_to_point_on_cap_plane, resolve_touch_point  # noqa: E402
from opening_motion import make_pose  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "bottle_opening_train_recovery"
DATASET_ROOT = _REPO_ROOT / "bottle_experiment/datasets/bottle_experiment/expert" / DATASET_NAME
FREQUENCY = 10

# Failure distribution, from the failed rollouts: first contact 127.5-133 mm below hover against 126 mm in
# the demos, i.e. 1.5-7 mm too deep; pushes of 5-53 mm before the stop. 2.5 mm is about where the
# successes stop (their deepest was 127.8 mm), so the low end still fails reliably.
DEPTH_EXCESS_M = (0.0025, 0.007)
PUSH_M = (0.0, 0.04)
# The induced push stops here: the failed rollouts peaked at 21-38 N (operator stop), the eval abort was 52 N at the time (96 N since 2026-10-16).
JAM_FORCE_N = 45.0
# Lift to this far above the PLANNED grip depth before going back over the grip point, so the fingers
# clear the lid's lip on the way back instead of dragging it the wrong way. 8 mm was too low on the rig
# (2026-10-13): the jammed push tilts the cap in the bottle gripper, and the way back caught it.
CLEARANCE_M = 0.03
# The opening motion the recovery ends with (approach to the grip point, push, legs) runs this much higher
# along the cap normal than planned: the jammed push tilts the cap in the bottle gripper, so the planned
# grip depth is too low afterwards (seen on the rig 2026-10-13).
RECOVERY_RAISE_M = 0.003
# Train pose INDICES (as printed per episode) skipped for the recovery collection only, on top of
# POSE_BLACKLIST["train"]. Kept out of splits.py: the expert dataset was collected against that list, and
# changing it would change which pose the code says each expert episode used.
# 11 (2026-10-13): stopped by the operator on the way to hover. 23 (2026-10-13): blacklisted by the operator.
RECOVERY_POSE_BLACKLIST = [11, 23]
# --policy: the rollout is stopped after this long even if nothing went wrong (the failures jammed 5-9 s in).
POLICY_TIMEOUT_S = 10.0


def install_observation_mode(slow, pause):
    """Slow every servoed move down `slow` times and/or wait for Enter before each, printing what comes
    next. Replaces collect_data_bottle.servo_to_waypoint, which the push, legs, retries, lift-offs and the
    retreat all go through, so the whole episode is affected, not only this script's own moves."""
    original = collect_data_bottle.servo_to_waypoint
    count = [0]

    def servo(env, recorder, target_pose, control_period, max_translation_step=0.01, max_rotation_step_rad=0.1,
              label="", **kwargs):
        count[0] += 1
        prompt = f"[move {count[0]}] {label} -> {np.round(target_pose[:3, 3], 4)}"
        if pause:
            input(prompt + "  (Enter to run, Ctrl+C to abort) ")
        else:
            print(prompt)
        return original(env, recorder, target_pose, control_period, max_translation_step=max_translation_step / slow,
                        max_rotation_step_rad=max_rotation_step_rad / slow, label=label, **kwargs)

    collect_data_bottle.servo_to_waypoint = servo


class _NotRecording:
    """Stands in for the recorder while the failure is staged: servo_to_waypoint records nothing."""

    def __init__(self, recorder):
        self._recorder = recorder

    def record_step(self, obs, action):
        pass

    @property
    def n_recorded_episodes(self):
        return self._recorder.n_recorded_episodes


class LidOpened(Exception):
    pass


class RecoveryStopped(Exception):
    pass


class _StopOnEnter:
    """The recorder, but every recorded step first checks for an operator Enter and raises RecoveryStopped.
    Every recorded move of the recovery (lift, way back, push, legs, retries, retreat) records one step per
    control cycle, so the stop lands within 0.1 s; the motion just stops getting new servo targets."""

    def __init__(self, recorder):
        self._recorder = recorder

    def record_step(self, obs, action):
        from robot_imitation_glue.ur5station.bottle.eval_bottle import poll_manual_stop

        if poll_manual_stop():
            raise RecoveryStopped
        self._recorder.record_step(obs, action)

    def __getattr__(self, name):
        return getattr(self._recorder, name)


def _frames_in_episode(recorder):
    buffer = recorder.lerobot_dataset.writer.episode_buffer
    return int(buffer["size"]) if buffer else 0


def _force(obs):
    return float(np.linalg.norm((np.asarray(obs["ft"]) - np.asarray(obs["ft_bias"]))[:3]))


def descend_deeper_while_turning(env, recorder, approach_pose, cap_normal, depth_excess, control_period,
                                 max_translation_step=0.01, max_rotation_step_rad=0.1, max_steps=2000):
    """The demo's hover -> grip point descent (yawing on the way), but the target sinks below the grip point
    in proportion to how much of the yaw is done: depth_excess deeper once the gripper has turned fully.

    A direct hover -> too-deep grip point servo cut through the cap on the rig (2026-10-13), and going to the
    grip point first and then straight down is not what a policy does; this keeps the demo's path while the
    fingers are still turning and only goes deeper as they reach the push orientation."""
    current = env.get_robot_pose_se3()
    def remaining_turn(pose):
        return np.linalg.norm(Rotation.from_matrix(pose[:3, :3].T @ approach_pose[:3, :3]).as_rotvec())
    initial_turn = max(remaining_turn(current), 1e-6)
    print(f"[move] descend to the grip point while turning {np.degrees(initial_turn):.0f} deg, going deeper as it turns")
    for _ in range(max_steps):
        progress = 1.0 - min(remaining_turn(current) / initial_turn, 1.0)
        target = approach_pose.copy()
        target[:3, 3] -= depth_excess * progress * cap_normal
        if progress > 0.99 and _pose_reached(current, target, 0.001, 0.02):
            break
        cycle_end = time.time() + control_period
        obs = env.get_observations()
        policy_action = step_action_to_policy_action_6d(
            _tool_frame_step(current, target, max_translation_step, max_rotation_step_rad))
        recorder.record_step(obs, policy_action)
        env.act_tcp(policy_action_to_tcp_pose(current, policy_action), time.time() + control_period)
        precise_wait(cycle_end)
        current = env.get_robot_pose_se3()
    return current


def load_policy_agent(checkpoint, num_inference_steps, n_action_steps):
    """The eval's agent (same observation preprocessing and inference overrides), for --policy."""
    from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy_for_inference
    from robot_imitation_glue.ur5station.bottle.eval_bottle import N_ENV_ACTION_DIMS, make_observation_preprocessor

    policy, preprocessor, postprocessor = make_lerobot_policy_for_inference(str(checkpoint))
    policy.config.num_inference_steps = num_inference_steps
    policy.diffusion.num_inference_steps = num_inference_steps
    policy.config.n_action_steps = n_action_steps
    policy.reset()
    print(f"[policy] {checkpoint}: {num_inference_steps} DDIM steps, {n_action_steps} action(s) per prediction")
    return LerobotAgent(policy, preprocessor, postprocessor, "cuda", make_observation_preprocessor(),
                        n_env_action_dims=N_ENV_ACTION_DIMS)


def run_policy_until_failure(env, agent, recorder, control_period):
    """Roll the policy out from the hover until the operator presses Enter, the force reaches JAM_FORCE_N,
    or POLICY_TIMEOUT_S passes. Returns (reason, force at the stop). Raises LidOpened if the policy opens the
    cap (sustained, as in the eval), and RuntimeError on a protective stop."""
    from robot_imitation_glue.ur5station.bottle.eval_bottle import (
        SUCCESS_SUSTAIN_STEPS,
        drain_pending_input,
        poll_manual_stop,
    )

    agent.reset()
    drain_pending_input()
    print("[policy] running -- press Enter as soon as it goes wrong (too deep / jammed)")
    sustained, force = 0, 0.0
    for _ in range(int(POLICY_TIMEOUT_S / control_period)):
        cycle_end = time.time() + control_period
        if poll_manual_stop():
            return "operator", force
        if not env.robot.rtde_control.isProgramRunning():
            raise RuntimeError("the LEFT arm's control script stopped (protective stop?) during the policy rollout")
        obs = env.get_observations()
        force = _force(obs)
        if force >= JAM_FORCE_N:
            print(f"[policy] force {force:.1f} N >= {JAM_FORCE_N:.0f} N -- stopping it")
            return "force", force
        sustained = sustained + 1 if is_cap_open(np.asarray(obs["bottle_sensor"])) else 0
        if sustained >= SUCCESS_SUSTAIN_STEPS:
            raise LidOpened
        action, _, _ = agent.get_action(obs)
        current = env.get_robot_pose_se3()
        recorder.record_step(obs, np.asarray(action, dtype=np.float64))
        env.act_tcp(agent.action_to_tcp_pose(action, current), time.time() + control_period)
        remaining = cycle_end - time.time()
        if remaining > 0:
            time.sleep(remaining)
    print(f"[policy] {POLICY_TIMEOUT_S:.0f} s passed -- stopping it")
    return "timeout", force


def push_until_jammed(env, recorder, target_pose, control_period, max_translation_step=0.01):
    """Servo toward target_pose (a straight push), recording into `recorder`, until it is reached or the
    force reaches JAM_FORCE_N. Raises LidOpened if the cap opens on the way -- then it was no failure."""
    current = env.get_robot_pose_se3()
    force = 0.0
    print(f"[move] push into the cap body -> {np.round(target_pose[:3, 3], 4)}")
    while not _pose_reached(current, target_pose, 0.001, 0.02):
        cycle_end = time.time() + control_period
        obs = env.get_observations()
        force = _force(obs)
        if is_cap_open(obs["bottle_sensor"]):
            raise LidOpened
        if force >= JAM_FORCE_N:
            print(f"[fail] jam force {force:.1f} N reached -- stopping the push")
            break
        policy_action = step_action_to_policy_action_6d(
            _tool_frame_step(current, target_pose, max_translation_step, 0.1))
        recorder.record_step(obs, policy_action)
        env.act_tcp(policy_action_to_tcp_pose(current, policy_action), time.time() + control_period)
        precise_wait(cycle_end)
        current = env.get_robot_pose_se3()
    return current, force


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-episodes", type=int, default=20, help="successful recovery episodes to collect")
    parser.add_argument("--record-failure", action="store_true",
                        help="record from the hover, the staged failure included (default: from the jam)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the depth/push draws")
    parser.add_argument("--policy", type=Path, default=None,
                        help="checkpoint (pretrained_model dir) whose rollout from the hover replaces the scripted failure")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="--policy: DDIM steps")
    parser.add_argument("--n-action-steps", type=int, default=1, help="--policy: actions executed per prediction")
    parser.add_argument("--slow", type=float, default=1.0,
                        help="run every move this many times slower, to watch it (nothing is saved)")
    parser.add_argument("--pause", action="store_true",
                        help="wait for Enter before every move and print which move it is (nothing is saved)")
    args = parser.parse_args()
    observing = args.slow != 1.0 or args.pause
    if observing:
        install_observation_mode(args.slow, args.pause)
        print(f"[observe] {args.slow:g}x slower{', pausing before every move' if args.pause else ''} -- "
              "episodes are NOT saved (slowed motion would distort the demonstrations)")

    schunk = make_gripper(GRIPPER_PORT_RIGHT)
    env = BottleStation(schunk, with_instrumentation=False, with_spectogram_model=False,
                        with_spectogram=True, use_internal_ft=True)
    # A run stopped before its first save leaves meta/info.json (0 episodes) and buffered images but no
    # meta/tasks.parquet; the recorder would then try to resume it and fail. Nothing in it is recoverable.
    if DATASET_ROOT.exists() and not (DATASET_ROOT / "meta" / "tasks.parquet").exists():
        print(f"[collect] {DATASET_ROOT} holds no saved episode (a run stopped before its first save) -- removing it")
        shutil.rmtree(DATASET_ROOT)
    recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((9,), dtype=np.float32),
        root_dataset_dir=DATASET_ROOT,
        dataset_name=DATASET_NAME,
        fps=FREQUENCY,
        use_videos=True,
    )
    not_recording = _NotRecording(recorder)
    agent = load_policy_agent(args.policy, args.num_inference_steps, args.n_action_steps) if args.policy else None
    log_path = DATASET_ROOT / "recovery_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    draws = np.random.default_rng(args.seed + recorder.n_recorded_episodes)  # a resumed run draws afresh
    control_period = 1 / FREQUENCY

    rr.init("robot_imitation_glue_bottle_recovery")
    rr.spawn(memory_limit="10GB")
    try:
        rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS["train"])
        blacklist = sorted(set(POSE_BLACKLIST["train"]) | set(RECOVERY_POSE_BLACKLIST))
        bottle_poses = generate_reachable_bottle_poses(
            SPLIT_N_POSES["train"], env.robot, env.robot_right, rng, blacklist=blacklist)
        # list position -> pose index, the number the blacklists hold (blacklisted indices still count)
        pose_indices = [i for i in range(len(bottle_poses) + len(blacklist)) if i not in blacklist][:len(bottle_poses)]
        intrinsics = env.get_camera_intrinsics()
        target = recorder.n_recorded_episodes + args.n_episodes

        while recorder.n_recorded_episodes < target:
            position = recorder.n_recorded_episodes % len(bottle_poses)
            pose_index = pose_indices[position]
            depth_excess, push = draws.uniform(*DEPTH_EXCESS_M), draws.uniform(*PUSH_M)
            failure = "the policy's rollout" if agent else f"{depth_excess * 1000:.1f} mm too deep, {push * 1000:.0f} mm push"
            print(f"\n===== {recorder.n_recorded_episodes}/{target} recovery episodes -- train pose index {pose_index} "
                  f"(the number to blacklist), failure: {failure} =====")

            # input("Press Enter to move the LEFT arm home (Ctrl+C to abort)...")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
            env.robot.rtde_control.zeroFtSensor()
            env.capture_ft_bias()

            # input("Press Enter to move the RIGHT arm to the bottle pose (Ctrl+C to abort)...")
            # _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
            env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            tcp_right_pose, _ = bottle_poses[position]
            env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
            # _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)
            cap_pose = env.get_bottle_cap_pose()
            cap_center, cap_normal = cap_pose[:3, 3], cap_pose[:3, 2]
            hover_pose = compute_hover_pose_above_bottle(cap_pose)
            env.move_robot_to_tcp_pose(hover_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)

            current_pose = env.get_robot_pose_se3()
            camera_pose_in_base = current_pose @ tcp_left_to_camera

            def grab_frame():
                return ImageConverter.from_numpy_int_format(
                    env.get_observations()["wrist_image_original"]).image_in_opencv_format

            while True:
                touch_pixel, _, image_bgr, _, _ = resolve_touch_point(grab_frame(), grab_frame)
                touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose_in_base, cap_pose)
                plan = plan_opening_motion(cap_pose, touch_point)
                image_rgb = ImageConverter.from_opencv_format(image_bgr).image_in_numpy_int_format
                log_plan_to_rerun(image_rgb, intrinsics, camera_pose_in_base, cap_pose, touch_point, touch_pixel, plan)
                answer = input("Check the plan in rerun. Enter = stage the failure and record the recovery, "
                               "t = retake the image and re-plan (Ctrl+C to abort): ").strip().lower()
                if answer != "t":
                    break

            reading = env.bottle_sensor.get_bottle_sensor()
            if channels_open(reading):
                print(f"[sensor] channel(s) {channels_open(reading)} already read open -- close the cap fully")

            if args.record_failure:
                recorder.start_episode()
            staging = recorder if args.record_failure else not_recording

            # --- the failure: too deep at the grip point, then push into the cap body ---
            approach_pose, push_pose = plan["waypoint_poses"][:2]
            stop_reason = "scripted"
            if agent is not None:
                if args.pause:
                    input("[move] policy rollout next (Enter to start it, Ctrl+C to abort) ")
                try:
                    stop_reason, jam_force = run_policy_until_failure(env, agent, staging, control_period)
                except LidOpened:
                    print("[policy] the policy opened the cap -- nothing to recover; discarding")
                    if args.record_failure:
                        recorder.delete_episode()
                    _lift_off_cap(env, not_recording, cap_normal, control_period, reason="policy opened the cap")
                    env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
                    input("Close the cap, then press Enter to go again...")
                    continue
            else:
                deep_offset = -depth_excess * cap_normal
                deep_approach = approach_pose.copy()
                deep_approach[:3, 3] += deep_offset
                print(f"[fail] descending to the grip point, {depth_excess * 1000:.1f} mm too deep by the end of the yaw")
                if args.pause:
                    input("[move] staging descent next (Enter to run, Ctrl+C to abort) ")
                descend_deeper_while_turning(env, staging, approach_pose, cap_normal, depth_excess, control_period,
                                             max_translation_step=0.01 / args.slow, max_rotation_step_rad=0.1 / args.slow)
                collect_data_bottle.servo_to_waypoint(env, staging, deep_approach, control_period,
                                                      label="staging: settle at the too-deep grip point")
                push_direction = push_pose[:3, 3] - approach_pose[:3, 3]
                push_direction /= np.linalg.norm(push_direction)
                jam_force = _force(env.get_observations())
                try:
                    if push > 0.001:
                        push_target = deep_approach.copy()
                        push_target[:3, 3] += push * push_direction
                        print(f"[fail] pushing {push * 1000:.0f} mm into the cap body (stops at {JAM_FORCE_N:.0f} N)")
                        _, jam_force = push_until_jammed(env, staging, push_target, control_period,
                                                         max_translation_step=0.01 / args.slow)
                except LidOpened:
                    print("[fail] the lid opened during the staged failure -- not a failure; discarding")
                    if args.record_failure:
                        recorder.delete_episode()
                    _lift_off_cap(env, not_recording, cap_normal, control_period, reason="lid opened while staging")
                    env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
                    # input("Close the cap, then press Enter to redo this pose with a new draw...")
                    continue
            jam_pose = env.get_robot_pose_se3()
            # where it stopped relative to the PLANNED grip point: depth below it, distance along the cap plane
            offset = jam_pose[:3, 3] - approach_pose[:3, 3]
            depth_below_grip = -float(offset @ cap_normal)
            pushed = float(np.linalg.norm(offset - (offset @ cap_normal) * cap_normal))
            print(f"[fail] stopped ({stop_reason}) {depth_below_grip * 1000:.1f} mm below the planned grip depth, "
                  f"{pushed * 1000:.0f} mm from the grip point along the cap, force {jam_force:.1f} N")

            # --- the recovery, recorded ---
            if not args.record_failure:
                recorder.start_episode()
            recovery_frame = _frames_in_episode(recorder)
            # Enter during the recovery stops it (RecoveryStopped from the recorder) and asks whether to keep it
            from robot_imitation_glue.ur5station.bottle.eval_bottle import drain_pending_input, retreat_left_arm
            drain_pending_input()
            watched = _StopOnEnter(recorder)
            stopped_by_operator = False
            print("[recover] recording -- press Enter to stop the recovery and decide whether to keep it")
            try:
                print(f"[recover] lifting to {CLEARANCE_M * 1000:.0f} mm above the planned grip depth")
                lifted = jam_pose.copy()
                lifted[:3, 3] += max(depth_below_grip + CLEARANCE_M, 0.0) * cap_normal
                collect_data_bottle.servo_to_waypoint(env, watched, lifted, control_period, label="recovery: lift off the cap body")
                if pushed > 0.002:
                    print("[recover] back over the grip point")
                    over_grip = make_pose(approach_pose[:3, :3], approach_pose[:3, 3] + CLEARANCE_M * cap_normal)
                    collect_data_bottle.servo_to_waypoint(env, watched, over_grip, control_period, label="recovery: back over the grip point")
                # the ordinary demo from here: descend to the planned grip depth, push, legs, stop-on-open, retries
                raised_plan = dict(plan)
                for key in ("waypoint_poses", "leg_turn_poses"):
                    raised_plan[key] = [make_pose(pose[:3, :3], pose[:3, 3] + RECOVERY_RAISE_M * cap_normal)
                                        for pose in plan[key]]
                print(f"[recover] the opening motion runs {RECOVERY_RAISE_M * 1000:.0f} mm higher than planned")
                _, success = run_opening_motion_with_retry(env, watched, raised_plan, cap_normal, control_period)

                if cap_normal @ (env.get_robot_pose_se3()[:3, 3] - cap_center) < 0.0:
                    _lift_off_cap(env, watched, cap_normal, control_period, reason="end of the motion")
                _retreat_to_hover(env, watched, plan, cap_normal, hover_pose, control_period)
                left_home = env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED)
                right_neutral = env.robot_right.move_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
                left_home.wait()
                right_neutral.wait()
            except RecoveryStopped:
                stopped_by_operator = True
                print(f"[recover] stopped by the operator after {_frames_in_episode(recorder)} recorded frame(s)")
                # off the cap first (not recorded), so the arm is not left pressing on it while you answer
                retreat_left_arm(env, cap_normal)
                env.robot_right.move_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED).wait()
                drain_pending_input()
                keep = input("[recover] save what was recorded so far? (y = save, anything else = discard): ")
                success = False
                if keep.strip().lower() in ("y", "yes"):
                    label = input("[recover] did the cap open (label it a success)? (y = success, anything else = "
                                  "failure -- prepare_datasets only uses successes): ")
                    success = label.strip().lower() in ("y", "yes")
                else:
                    recorder.delete_episode()
                    stopped_by_operator = None  # discarded: nothing left to save

            if stopped_by_operator is None:
                print("[episode] discarded")
                continue
            recorder.set_episode_success(success)
            if observing:
                recorder.delete_episode()
                print(f"[observe] recovery {'opened' if success else 'did NOT open'} the cap -- not saved")
                # input("Close the cap, then press Enter for the next episode...")
            elif success or stopped_by_operator:
                recorder.save_episode()
                log.append({"episode": recorder.n_recorded_episodes - 1, "pose_index": pose_index,
                            "failure": f"policy {args.policy}" if agent else "scripted",
                            "stop_reason": stop_reason, "depth_below_grip_mm": round(depth_below_grip * 1000, 2),
                            **({} if agent else {"depth_excess_mm": round(depth_excess * 1000, 2),
                                                 "push_drawn_mm": round(push * 1000, 1)}),
                            "push_done_mm": round(pushed * 1000, 1), "jam_force_n": round(jam_force, 1),
                            "success": bool(success), "stopped_by_operator": stopped_by_operator,
                            "recorded_failure": args.record_failure, "recovery_first_frame": recovery_frame,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
                log_path.write_text(json.dumps(log, indent=1))
                print(f"[episode] saved (episode {recorder.n_recorded_episodes - 1}, pushed {pushed * 1000:.0f} mm, "
                      f"jam force {jam_force:.1f} N)")
                log_idle_to_rerun(env, recorder, "episode saved -- close the bottle by hand")
                # input("Close the bottle by hand, then press Enter to continue...")
            else:
                recorder.delete_episode()
                print("[episode] the recovery did not open the cap -- discarded")
                # input("Close the cap again, then press Enter to retry this pose with a new draw...")
    finally:
        recorder.finish_recording()
        print(f"[collect] {recorder.n_recorded_episodes} recovery episodes in {DATASET_ROOT}")
        env.close()


if __name__ == "__main__":
    main()
