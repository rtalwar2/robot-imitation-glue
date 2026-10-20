"""Automated data collection for the two-arm bottle-opening task.

Modelled on `collect_data_xyz` in `collect_data_delta.py`: a fixed-frequency loop that
records `(obs, policy_action)` frames in the same 9-dim `[delta_xyz(3), rotation_6d(6)]`
format. Unlike `collect_data_xyz` (which servoes toward a live target computed from sensor
feedback every cycle), the bottle-opening motion is a chain of *precomputed* waypoints
(`bottle_experiment/open_bottle_agent.plan_opening_motion`), previously executed with
blocking `move_linear_to_tcp_pose(...).wait()` calls. `servo_to_waypoint` below is the
bridge: it replaces each blocking waypoint move with a fixed-rate sub-loop that records a
step every cycle while servoing toward that same waypoint, so the recorded action is
exactly the commanded delta pose.
"""

import sys
import time
from pathlib import Path
from queue import Empty

import cv2
import loguru
import numpy as np
import rerun as rr
from airo_camera_toolkit.utils.image_converter import ImageConverter
from scipy.spatial.transform import Rotation

from robot_imitation_glue.base import BaseDatasetRecorder, BaseEnv
from robot_imitation_glue.collect_data_delta import (
    Event,
    State,
    init_keyboard_listener,
    policy_action_to_tcp_pose,
    step_action_to_policy_action_6d,
)
from robot_imitation_glue.hardware.bottle_sensor import (
    SENSOR_CHECKPOINTS,
    channels_open,
    is_cap_open,
    is_uncovered,
)
from robot_imitation_glue.utils import precise_wait

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOTTLE_EXPERIMENT_PATH = str(_REPO_ROOT / "bottle_experiment")
if _BOTTLE_EXPERIMENT_PATH not in sys.path:
    sys.path.insert(0, _BOTTLE_EXPERIMENT_PATH)

from calibrate_and_hover_bottle import (  # noqa: E402
    compute_hover_pose_above_bottle,
    tcp_left_to_camera,
)
from motion_constants import (  # noqa: E402
    CHECKPOINT_GRACE_S,
    DEPTH_NUDGE_M,
    LEGS,
    MAX_MOTION_RETRIES,
    OPEN_DWELL_S,
    OPEN_POLL_S,
    RETRACT_LIFT_METERS,
    STOP_LEGS_WHEN_OPEN,
    TURN_DURING_LEGS,
)
from open_bottle_agent import (  # noqa: E402
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    RETREAT_SPEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    log_plan_to_rerun,
    plan_opening_motion,
)
from opening_motion import in_place_turns, make_pose, retreat_rotations, turn_angle_of  # noqa: E402
from open_bottle_demo import (  # noqa: E402
    pixel_to_point_on_cap_plane,
    resolve_touch_point,
    save_touch_point_sample,
)

logger = loguru.logger

# Right-arm (bottle) gripper widths, metres between fingers -- set at the rig during the 2026-10
# recollection prep: 0.040 clears the bottle body for release/placement, 0.023 is the taught
# clamp on the body (never fully closed, so the grasp force limits the squeeze).
GRIPPER_OPEN_WIDTH_M = 0.028
GRIPPER_GRASP_WIDTH_M = 0.023


def log_observation_to_rerun(obs, recording, n_episodes, label=""):
    """Live wrist feed, spectrogram and cap-sensor traces in rerun, with a recording banner.

    Takes an already-fetched observation rather than reading the env itself: the DDS subscribers
    behind `bottle_sensor` and the spectrogram use `reader.take()`, which *consumes* samples, so a
    second reader (a background viz thread, or a viz call that re-fetches) would steal readings
    from the recording loop and silently corrupt what lands in the dataset.
    """
    vis_img = np.ascontiguousarray(obs["wrist_image"].copy())
    if recording:
        cv2.putText(vis_img, "RECORDING", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        cv2.putText(vis_img, "NOT RECORDING", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    cv2.putText(vis_img, f"episodes: {n_episodes}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if label:
        cv2.putText(vis_img, label, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    rr.log("wrist_image", rr.Image(vis_img, rr.ColorModel.RGB))
    rr.log("status", rr.TextLog(f"{'RECORDING' if recording else 'not recording'} | {label}"))
    if "spectogram_image" in obs:
        rr.log("spectogram", rr.Image(obs["spectogram_image"], rr.ColorModel.RGB))
    for channel_index, voltage in enumerate(obs["bottle_sensor"]):
        rr.log(f"bottle_sensor/S{channel_index}", rr.Scalars(float(voltage)))
    rr.log("ft", rr.TextLog(str(np.round(np.asarray(obs["ft"]) - np.asarray(obs["ft_bias"]), 2))))


def log_idle_to_rerun(env, dataset_recorder, label):
    """Refresh the rerun view while NOT recording (moving to home, verifying, waiting on input).

    Safe to fetch here because the recording loop is not running concurrently. Blocking `input()`
    prompts freeze the view on the last frame, which is honest -- nothing is being recorded then.
    """
    log_observation_to_rerun(
        env.get_observations(), recording=False, n_episodes=dataset_recorder.n_recorded_episodes, label=label
    )


def _tool_frame_step(current_pose, target_pose, max_translation_step, max_rotation_step_rad):
    """Clipped delta from current_pose to target_pose in the TOOL frame: [delta_xyz(3),
    delta_rotvec(3)], the format step_action_to_policy_action_6d expects."""
    delta_translation_base = target_pose[:3, 3] - current_pose[:3, 3]
    delta_translation_tool = current_pose[:3, :3].T @ delta_translation_base
    distance = np.linalg.norm(delta_translation_tool)
    if distance > max_translation_step:
        delta_translation_tool = delta_translation_tool / distance * max_translation_step

    delta_rotation = current_pose[:3, :3].T @ target_pose[:3, :3]
    rotvec = Rotation.from_matrix(delta_rotation).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle > max_rotation_step_rad:
        rotvec = rotvec / angle * max_rotation_step_rad

    return np.concatenate((delta_translation_tool, rotvec))


def _pose_reached(current_pose, target_pose, pos_tol, rot_tol_rad):
    # print("check pose reached")
    pos_error = np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])
    delta_rotation = current_pose[:3, :3].T @ target_pose[:3, :3]
    rot_error = np.linalg.norm(Rotation.from_matrix(delta_rotation).as_rotvec())
    return pos_error <= pos_tol and rot_error <= rot_tol_rad


def servo_to_waypoint(
    env: BaseEnv,
    dataset_recorder: BaseDatasetRecorder,
    target_pose,
    control_period,
    max_translation_step=0.01,
    max_rotation_step_rad=0.1,
    pos_tol=0.001,
    rot_tol_rad=0.02,
    max_steps=2000,
    label="",
):
    """Fixed-rate replacement for `move_linear_to_tcp_pose(target_pose).wait()`: servoes the
    left arm to target_pose, recording one (obs, policy_action) frame per control cycle.
    Returns the pose actually reached (== target_pose, modulo tolerance).

    Frames are recorded with the dataset default `next.success=False` -- an episode's
    outcome is only knowable in hindsight (the checkpoint for a segment is only checked
    AFTER that segment's frames are already recorded), so the caller must retroactively
    label the whole episode via dataset_recorder.set_episode_success() once the outcome is
    known, rather than trying to pass the right value in here per step.
    """
    current_pose = env.get_robot_pose_se3()

    for _ in range(max_steps):
        if _pose_reached(current_pose, target_pose, pos_tol, rot_tol_rad):
            break
        cycle_end_time = time.time() + control_period
        obs = env.get_observations()
        step_action = _tool_frame_step(current_pose, target_pose, max_translation_step, max_rotation_step_rad)
        policy_action = step_action_to_policy_action_6d(step_action)
        next_pose = policy_action_to_tcp_pose(current_pose, policy_action)
        dataset_recorder.record_step(obs, policy_action)
        env.act_tcp(next_pose, time.time() + control_period)
        log_observation_to_rerun(
            obs, recording=True, n_episodes=dataset_recorder.n_recorded_episodes, label=label
        )
        precise_wait(cycle_end_time)
        current_pose = env.get_robot_pose_se3()
    else:
        logger.warning(f"servo_to_waypoint: did not reach target pose within {max_steps} steps")
    return current_pose


def _check_checkpoint(env, event_name):
    """Evaluate the sensor-verified checkpoint for event_name, if SENSOR_CHECKPOINTS gates it.

    Returns None if event_name isn't gated, True if gated and satisfied within CHECKPOINT_GRACE_S
    (the lid can finish flipping just after the arm stops), False if gated and still not satisfied
    after it. Raises if the sensor feed has never delivered a real sample (see
    BottleSensorSubscriber.has_received_sample) -- otherwise a dead/disconnected
    bottle_ble_reader.py would silently read as "still covered" forever.
    """
    required_channels = SENSOR_CHECKPOINTS.get(event_name)
    if not required_channels:
        return None

    reading = env.get_observations()["bottle_sensor"]
    if not env.bottle_sensor.has_received_sample():
        raise RuntimeError(
            f"bottle_sensor has never received a DDS sample (still reading {reading} at "
            f"{event_name}) -- is bottle_ble_reader.py running and connected? Every "
            "checkpoint will otherwise read as 'covered' forever and the motion will just "
            "keep retrying against a dead sensor feed."
        )

    deadline = time.monotonic() + CHECKPOINT_GRACE_S
    while not is_uncovered(reading, required_channels) and time.monotonic() < deadline:
        precise_wait(OPEN_POLL_S)
        reading = env.get_observations()["bottle_sensor"]
    if is_uncovered(reading, required_channels):
        logger.info(f"[verify] {event_name}: OK (reading={np.round(reading, 2)})")
        return True
    logger.info(f"[verify] {event_name}: not uncovered (reading={np.round(reading, 2)}) -- will redo the motion")
    return False


def _cap_is_open_sustained(env, dwell_s=OPEN_DWELL_S):
    """True only if the deciding channels (OPEN_CHANNELS: S1, S2) hold at/above threshold for `dwell_s` straight.

    Polled because the DDS subscriber keeps a single sample (History.KeepLast(1)) -- there is no
    history to look back through, so "sustained" has to be measured forward. The poll does not move
    the arm, which leaves a dwell_s gap in the recorded frames; that is deliberate, since a gap is
    cheaper than skipping the check and labelling run_0017's transient a success.
    """
    if not env.bottle_sensor.has_received_sample():
        raise RuntimeError(
            f"bottle_sensor has never received a DDS sample -- is bottle_ble_reader.py running and "
            f"connected? Stop-on-open would otherwise never fire (or fire off garbage)."
        )
    deadline = time.monotonic() + dwell_s
    while True:
        reading = env.bottle_sensor.get_bottle_sensor()
        if not is_cap_open(reading):
            return False
        if time.monotonic() >= deadline:
            return True
        precise_wait(OPEN_POLL_S)


def _lift_off_cap(env, dataset_recorder, cap_normal, control_period, reason):
    """Lift RETRACT_LIFT_METERS along the cap normal to disengage the gripper, recorded.

    Shared by the retry path and the stop-on-open exit: the gripper is pressed 2+ cm below the cap
    plane, so ending the motion means coming off the cap the same way in both cases. Going all the
    way back to the hover pose would work but wastes most of the travel -- the episode's own retreat
    handles that, and every centimetre here is recorded as demonstration steps.
    """
    print(f"[move] lifting {RETRACT_LIFT_METERS * 100:.0f}cm off the cap ({reason})")
    lift_pose = env.get_robot_pose_se3().copy()
    lift_pose[:3, 3] = lift_pose[:3, 3] + RETRACT_LIFT_METERS * cap_normal
    return servo_to_waypoint(
        env,
        dataset_recorder,
        lift_pose,
        control_period,
        label=f"lifting {RETRACT_LIFT_METERS * 100:.0f}cm ({reason})",
    )


def _unwind_in_place(env, dataset_recorder, plan, cap_normal, to_angle, control_period, reason):
    """Turn the gripper in place, recorded, to `to_angle` about the cap normal (turn_angle_of's scale: 0 =
    the push's yaw), in steps that each take the intended way round.

    Needed because servo_to_waypoint follows the SHORTEST rotation: the legs turn the gripper by up to
    ~240 deg, so servoing straight back to the push's yaw or the hover pose would keep turning instead
    of unwinding and wind the wrist (and the camera cable) a full turn.
    """
    current = env.get_robot_pose_se3()
    from_angle = turn_angle_of(plan, cap_normal, current)
    for rotation in in_place_turns(plan, cap_normal, from_angle, to_angle):
        current = servo_to_waypoint(env, dataset_recorder, make_pose(rotation, current[:3, 3]), control_period,
                                    label=f"unwinding in place ({reason})")
    return current


def _retreat_to_hover(env, dataset_recorder, plan, cap_normal, hover_pose, control_period):
    """Rise to the hover pose WHILE unwinding, recorded.

    The rotation back to the hover orientation can exceed 180 deg, and servo_to_waypoint follows the
    shortest rotation, so one servo to hover_pose would keep turning and wind the wrist. retreat_rotations
    undoes the wrist's net turn since hover in one direction, in steps of <= 120 deg (undoing the leg turns
    and then the initial yaw separately reversed direction when those had opposite signs), and the position
    rises toward the hover position across those steps -- the gripper climbs away from the lid while it turns.
    """
    current = env.get_robot_pose_se3()
    rotations = retreat_rotations(plan, cap_normal, current, hover_pose[:3, :3])
    start, end = current[:3, 3], hover_pose[:3, 3]
    for k, rotation in enumerate(rotations, start=1):
        current = servo_to_waypoint(
            env, dataset_recorder, make_pose(rotation, start + (end - start) * k / len(rotations)), control_period,
            max_translation_step=RETREAT_SPEED * control_period,
            label=f"returning to hover while unwinding ({k}/{len(rotations)})",
        )
    return current


def _move_gripper_to(env, width, tol_m=0.002, timeout_s=10.0):
    """Command the bottle gripper to `width` and block until the fingers are done.

    Polls get_current_width(), not move()'s AwaitableAction: that sleeps 0.11 s and then checks
    velocity, so it can report done while the fingers are still at the old width. Completion is
    arrival OR stall-after-moving -- a grasp legitimately stalls on the bottle body (the force
    limit working), so the position need not reach the target there.

    Raises on failure instead of warning-and-continuing: if the fingers are not holding the
    bottle, the taught grasp transform behind get_bottle_cap_pose() is fiction and every frame
    recorded afterwards would be a poisoned episode.
    """
    try:
        env.gripper.move(width)
        deadline = time.monotonic() + timeout_s
        started = False
        while time.monotonic() < deadline:
            if abs(env.gripper.get_current_width() - width) <= tol_m:
                return
            if env.gripper.is_moving():
                started = True
            elif started:
                return  # moved, then stalled: contact with the bottle body or a hard stop
            time.sleep(0.02)
        stuck_at = env.gripper.get_current_width()
    except Empty as exc:
        raise RuntimeError(
            "[gripper] the gripper child process stopped answering commands (result-queue timeout) "
            "-- it died after startup or lost the port; restart the collector"
        ) from exc
    raise RuntimeError(
        f"[gripper] could not reach {width:.3f} m (fingers stuck at {stuck_at:.3f} m) -- the bottle "
        "is not grasped; aborting the episode rather than recording garbage"
    )


def _build_motion_segments(plan):
    """The motion as an ordered list of (label, base_target_pose, checkpoint_event_or_None).

    Segment order: approach (yaw+descend, ungated) -> push (gated only if push_end is in
    SENSOR_CHECKPOINTS) -> leg2 (leg_2_end) -> leg3 (leg_3_end) -> leg4..leg6 (ungated). Every leg pose
    carries its turn (opening_motion: the gripper turns with the path); with TURN_DURING_LEGS off, an
    ungated in-place turn segment precedes each leg.
    """
    approach_pose, push_pose, *leg_poses = plan["waypoint_poses"]
    # Only name a checkpoint that the map actually gates: a segment labelled with an ungated event
    # still counts towards _previous_checkpoint_segment_index, so the cascade would read
    # "the earlier checkpoint also failed" every time and force a full restart instead of resuming
    # from the last leg that really did uncover its channel.
    push_ckpt = "push_end" if "push_end" in SENSOR_CHECKPOINTS else None
    segments = [("approach", approach_pose, None), ("push", push_pose, push_ckpt)]
    for leg_index, leg_pose in enumerate(leg_poses):
        angle_deg, offset = LEGS[leg_index]
        event_name = f"leg_{leg_index + 2}_end"
        checkpoint = event_name if event_name in SENSOR_CHECKPOINTS else None
        label = f"leg{leg_index + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm)"
        if not TURN_DURING_LEGS:
            segments.append((f"turn before leg{leg_index + 2}", plan["leg_turn_poses"][leg_index], None))
        segments.append((label, leg_pose, checkpoint))
    return segments


def _previous_checkpoint_segment_index(segments, segment_index):
    """The segment index of the checkpoint immediately before segments[segment_index]'s own
    checkpoint, or None if segments[segment_index] is the first checkpoint in the motion."""
    checkpoint_segment_indices = [i for i, (_, _, cp) in enumerate(segments) if cp is not None]
    position = checkpoint_segment_indices.index(segment_index)
    return checkpoint_segment_indices[position - 1] if position > 0 else None


def run_opening_motion_with_retry(env, dataset_recorder, plan, cap_normal, control_period):
    """Execute the descend + push + LEGS motion, gated by sensor-verified checkpoints
    (leg_2_end, leg_3_end, leg_5_end -- see SENSOR_CHECKPOINTS), pressing DEPTH_NUDGE_M
    deeper into the cap on every retry (an identical replay would very likely just fail
    identically) -- up to MAX_MOTION_RETRIES times.

    Recovery is cascading rather than always a full restart: on a failed checkpoint, retract
    to the hover pose and RE-CHECK the PRECEDING checkpoint's sensor.
      - If that earlier checkpoint still holds, only that slipped -- resume the motion
        directly from the earlier checkpoint's own waypoint (deeper), skipping the segments
        before it (they're still fine, no need to redo them).
      - If the earlier checkpoint has ALSO lost its grip, the problem runs deeper than just
        this leg -- restart the whole motion from the grip point.
      - The first checkpoint (leg_2_end) has no earlier checkpoint to fall back on, so its
        failure always triggers a full restart.
    The retraction and every (partial or full) redo are recorded as ordinary steps, so a
    slipped grip and its recovery become part of the demonstration.

    Returns (final_pose, success): success is True only once the LAST gated checkpoint of
    SENSOR_CHECKPOINTS -- the stage where the final channel uncovers -- has passed. It is located
    by position rather than by a hardcoded event name: after the first 2026-10-07 remap a literal
    "leg_6_end" comparison could never fire and every episode would have been recorded as a
    failure. The map was re-remapped the SAME DAY (S2 leg_3_end -> leg_5_end, after the first
    collected pose showed a leg-3 rotation shortfall) -- which is precisely why the gate resolves
    by position and survived both remaps without a code change.
    """
    segments = _build_motion_segments(plan)
    gated_indices = [i for i, (_, _, cp) in enumerate(segments) if cp is not None]
    last_gated_index = gated_indices[-1] if gated_indices else None

    attempt = 0
    resume_index = 0
    reached_pose = segments[0][1]
    episode_success = False
    # Stop-on-open (and its success label) needs every channel to have read covered before the push; a
    # channel already open at the grip point makes "all channels open" true as soon as the others open.
    stop_on_open_armed = True

    while True:
        depth_offset = -attempt * DEPTH_NUDGE_M * cap_normal

        if attempt > 0:
            logger.info(
                f"[verify] attempt {attempt}/{MAX_MOTION_RETRIES}: resuming from "
                f"'{segments[resume_index][0]}', {attempt * DEPTH_NUDGE_M * 100:.1f}cm deeper"
            )

        episode_success = False
        checkpoint_failed_index = None
        stopped_open = False
        for segment_index in range(resume_index, len(segments)):
            label, base_pose, checkpoint_name = segments[segment_index]
            target_pose = base_pose.copy()
            target_pose[:3, 3] = target_pose[:3, 3] + depth_offset
            print(f"[move] {label} -> {np.round(target_pose[:3, 3], 4)}")
            reached_pose = servo_to_waypoint(
                env, dataset_recorder, target_pose, control_period,
                label=f"attempt {attempt}: {label}",
            )
            if segment_index == 0 and attempt == 0:  # at the grip point, before the push
                reading = env.bottle_sensor.get_bottle_sensor()
                already_open = channels_open(reading)
                if already_open:
                    stop_on_open_armed = False
                    logger.warning(
                        "[sensor] " + ", ".join(f"S{c} ({reading[c]:.2f} V)" for c in already_open)
                        + " already read(s) OPEN at the grip point, before the push -- is the cap fully closed? "
                        "Stop-on-open is OFF for this episode and its checkpoints on those channels prove nothing."
                    )

            # Checked after every segment, including ungated ones: once the whole cap is open the
            # remaining legs have nothing left to do. Cost is nil before the pop -- the poll returns
            # on its first read as soon as any channel is still covered.
            if STOP_LEGS_WHEN_OPEN and stop_on_open_armed and _cap_is_open_sustained(env):
                episode_success = True
                stopped_open = True
                logger.info(
                    f"[verify] cap fully open (S1 and S2 >= threshold for "
                    f"{OPEN_DWELL_S}s) after '{label}' -- skipping the remaining "
                    f"{len(segments) - segment_index - 1} segment(s) and lifting off"
                )
                break

            if checkpoint_name is None:
                continue
            checkpoint_ok = _check_checkpoint(env, checkpoint_name)
            if checkpoint_ok is False:
                checkpoint_failed_index = segment_index
                break
            if checkpoint_ok is True and segment_index == last_gated_index:
                episode_success = True

        if stopped_open:
            reached_pose = _lift_off_cap(
                env, dataset_recorder, cap_normal, control_period, reason="cap already open"
            )
            return reached_pose, True

        if checkpoint_failed_index is None:
            return reached_pose, episode_success

        if attempt == MAX_MOTION_RETRIES:
            logger.warning(f"[verify] giving up after {MAX_MOTION_RETRIES} retries -- continuing anyway")
            return reached_pose, False

        # Lift just far enough to disengage the gripper from the cap, so the sensor reads the cap's
        # own state rather than whatever the gripper is holding it in.
        reached_pose = _lift_off_cap(
            env, dataset_recorder, cap_normal, control_period, reason="to check recovery state"
        )

        previous_segment_index = _previous_checkpoint_segment_index(segments, checkpoint_failed_index)
        if previous_segment_index is None:
            resume_index = 0
            logger.info("[verify] no earlier checkpoint to fall back on -- restarting from the grip point")
        else:
            previous_checkpoint_name = segments[previous_segment_index][2]
            if _check_checkpoint(env, previous_checkpoint_name):
                resume_index = previous_segment_index
                logger.info(
                    f"[verify] '{previous_checkpoint_name}' still holds -- resuming from "
                    f"'{segments[previous_segment_index][0]}'"
                )
            else:
                resume_index = 0
                logger.info(f"[verify] '{previous_checkpoint_name}' has also failed -- restarting from the grip point")

        # turn back (in place, at the lifted pose) to the orientation the resumed segment starts from
        reached_pose = _unwind_in_place(env, dataset_recorder, plan, cap_normal,
                                        turn_angle_of(plan, cap_normal, segments[resume_index][1]),
                                        control_period, reason="before the retry")
        attempt += 1

    return reached_pose, False


def collect_data_bottle_opening(env, dataset_recorder, frequency=10, bottle_poses=None, n_episodes=None, pose_indices=None):
    """Collect `n_episodes` successful demonstrations, one per generated bottle pose by default.

    Defaults to len(bottle_poses), so each pose is used exactly once when nothing fails. Failed
    episodes are discarded without advancing the counter, so a pose that fails is retried rather
    than skipped -- which also means a pose that can never succeed will loop indefinitely; watch
    the printed progress.

    Counting successful episodes in the dataset (rather than loop iterations) makes this resumable:
    the recorder picks up n_recorded_episodes from an existing dataset, so a re-run tops up to the
    target instead of starting over.

    pose_indices: optional explicit list of indices into bottle_poses to run, in order, instead of
    the normal `n_recorded_episodes % len(bottle_poses)` cycling -- one successful episode per
    entry. For targeted recovery of specific known-missing episodes (e.g. re-collecting poses that
    were lost to a crash), where the poses to run don't correspond to the resumed dataset's
    existing episode count. n_episodes is ignored when this is set: the target is always
    `n_recorded_episodes at start + len(pose_indices)`.
    """
    if not bottle_poses:
        raise ValueError("bottle_poses is empty -- generate reachable poses first")
    if pose_indices is not None:
        target_episodes = dataset_recorder.n_recorded_episodes + len(pose_indices)
    else:
        target_episodes = len(bottle_poses) if n_episodes is None else n_episodes
    recovery_cursor = 0

    rr.init("robot_imitation_glue_bottle")
    rr.spawn(memory_limit="10GB")
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)
    control_period = 1 / frequency

    intrinsics = env.get_camera_intrinsics()

    if dataset_recorder.n_recorded_episodes >= target_episodes:
        print(
            f"[collect] dataset already holds {dataset_recorder.n_recorded_episodes}/{target_episodes} "
            "episodes -- nothing to do"
        )
        return

    try:
        while not event.quit and dataset_recorder.n_recorded_episodes < target_episodes:
            print(f"\n===== {dataset_recorder.n_recorded_episodes}/{target_episodes} episodes collected =====")
            input("Press Enter to move the LEFT arm to the retracted home pose (Ctrl+C to abort)...")
            print("[move] retreating ur_left home")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

            # Zero the FT drift at the home pose, which is the one configuration that is identical
            # every episode -- the payload's gravity contribution is therefore constant here, so any
            # episode-to-episode change in the reading is drift. Must happen before the right arm
            # moves the bottle, so nothing is in contact. The raw reading is still what gets
            # recorded; this only publishes the offset alongside it.
            # Re-zero the UR's FT sensor first, as eval_bottle and collect_recovery_bottle do: its offset is
            # fixed in the SENSOR frame, so a home-pose bias alone leaves (R_t - R_home) * offset at every
            # other orientation. The 90 expert demos of 2026-10-09/10 were collected without this and read
            # 6.7 N median (up to 17 N) in free air, pose-dependent, vs ~2 N in the eval rollouts.
            env.robot.rtde_control.zeroFtSensor()
            env.capture_ft_bias()
            log_idle_to_rerun(env, dataset_recorder, "at home -- FT bias captured")

            pose_index = pose_indices[recovery_cursor] if pose_indices is not None else (
                dataset_recorder.n_recorded_episodes % len(bottle_poses)
            )
            rr.set_time("pose", sequence=dataset_recorder.n_recorded_episodes)
            tcp_right_pose, _planned_cap_pose = bottle_poses[pose_index]

            print(f"\n=== pose {pose_index + 1}/{len(bottle_poses)} (episode {dataset_recorder.n_recorded_episodes}) ===")
            input("Press Enter to move the RIGHT arm to the next bottle pose (Ctrl+C to abort)...")
            # Via the neutral pose first, so the right arm always reaches a bottle pose from the same
            # configuration rather than swinging directly between two arbitrary sampled poses.
            # is_tcp_pose_reachable only checks per-waypoint IK, never the path between waypoints, so
            # a direct pose-to-pose move can sweep through the other arm or the table even though both
            # endpoints are individually fine.
            print("[move] ur_right to the neutral pose")
            # Fingers open BEFORE travelling: this is also the release point for the bottle held
            # from the previous episode -- the operator re-stands it at the new pose while the
            # arm transits, and the close below re-establishes the taught grasp transform
            # (DEFAULT_TCP_RIGHT_TO_BOTTLE) that get_bottle_cap_pose() infers the cap pose from.
            print(f"[gripper] ur_right opening to {GRIPPER_OPEN_WIDTH_M:.3f} m")
            _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
            env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
            print(f"[gripper] ur_right closing on the bottle at {GRIPPER_GRASP_WIDTH_M:.3f} m")
            _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)
            cap_pose = env.get_bottle_cap_pose()
            cap_center = cap_pose[:3, 3]
            cap_normal = cap_pose[:3, 2]

            hover_pose = compute_hover_pose_above_bottle(cap_pose)
            print(f"[move] ur_left to hover pose above the cap at {np.round(hover_pose[:3, 3], 4)}")
            env.move_robot_to_tcp_pose(hover_pose,joint_speed=LEFT_TRANSIT_JOINT_SPEED)

            current_pose = env.get_robot_pose_se3()
            camera_pose_in_base = current_pose @ tcp_left_to_camera
            hover_height = float(cap_normal.dot(current_pose[:3, 3] - cap_center))
            log_idle_to_rerun(env, dataset_recorder, "at hover -- verify the touch point")

            def grab_frame():
                """A fresh wrist frame. The arm is stationary at the hover pose throughout
                verification, so re-grabbing does not invalidate hover_height, camera_pose_in_base
                or cap_pose."""
                image_bgr = ImageConverter.from_numpy_int_format(
                    env.get_observations()["wrist_image_original"]
                ).image_in_opencv_format
                if image_bgr.shape[:2] != (720, 1280):
                    logger.warning(
                        f"wrist camera resolution is {image_bgr.shape[:2]}, but the touch-point detector was "
                        "trained on 720p (1280x720) frames -- detection may be inaccurate. "
                        "(CameraFactory.create_wrist_camera requests 720p and no longer falls back, so this "
                        "means the frame is being resized somewhere downstream.)"
                    )
                return image_bgr

            # The learned detector's point, with the verify/click window only when it is unsure; grab_frame
            # lets 't' in that window replace a blurred/occluded frame. The plan prompt can start over from a
            # fresh frame too: 't' there re-grabs, re-detects and re-plans.
            while True:
                touch_pixel, verify_overlay, image_bgr, detected_pixel, human_checked = resolve_touch_point(
                    grab_frame(), grab_frame
                )
                image_rgb = ImageConverter.from_opencv_format(image_bgr).image_in_numpy_int_format
                if human_checked:  # only frames a person looked at become training labels
                    save_touch_point_sample(image_bgr, verify_overlay, hover_height, touch_pixel, detected_pixel)
                if tuple(touch_pixel) != tuple(detected_pixel):
                    print(f"[verify] using corrected touch point {touch_pixel} (detection was {detected_pixel})")

                touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose_in_base, cap_pose)
                plan = plan_opening_motion(cap_pose, touch_point)
                log_plan_to_rerun(image_rgb, intrinsics, camera_pose_in_base, cap_pose, touch_point, touch_pixel, plan)

                print(f"[plan] touch point={np.round(touch_point, 4)}  grip point={np.round(plan['grip_point'], 4)}")
                answer = input("Check the plan in rerun. Enter = run the opening motion, t = retake the image and "
                               "re-plan (Ctrl+C to abort): ").strip().lower()
                if answer != "t":
                    break
                print("[detect] retaking the image")

            dataset_recorder.start_episode()

            _final_pose, episode_success = run_opening_motion_with_retry(
                env, dataset_recorder, plan, cap_normal, control_period
            )

            # retreat: return to the hover pose above the cap centre (recorded), then transit back
            # home (not recorded -- a reset, not demonstration behaviour). Ending at the hover pose
            # rather than rising straight up from wherever leg 6 finished gives every demonstration
            # the same terminal state relative to the bottle, instead of one that varies with which
            # leg the motion ended on and how far the retries pushed it.
            # A motion that ran to its end is still pressed below the cap plane, so it comes off the cap
            # first (straight up, no rotation); then it rises to hover while unwinding (_retreat_to_hover).
            if cap_normal @ (env.get_robot_pose_se3()[:3, 3] - cap_center) < 0.0:
                _lift_off_cap(env, dataset_recorder, cap_normal, control_period, reason="end of the motion")
            print("[move] returning to the hover pose above the cap, unwinding the leg turns on the way")
            _retreat_to_hover(env, dataset_recorder, plan, cap_normal, hover_pose, control_period)

            # ur_left goes home while ur_right takes the bottle back to its neutral pose, at the same time to
            # save the transit. Both move away from the hover/bottle region; their paths are not checked
            # against each other, so keep a hand near the e-stop. The next episode's neutral move is then a
            # no-op, and the fingers still open there first, as before.
            print("[move] ur_left home and ur_right (with the bottle) to its neutral pose, together")
            left_home = env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED)
            right_neutral = env.robot_right.move_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            left_home.wait()
            right_neutral.wait()

            # every frame so far was recorded with the dataset default next.success=False --
            # the true outcome is only known now, so label the whole episode retroactively.
            dataset_recorder.set_episode_success(episode_success)

            if episode_success:
                dataset_recorder.save_episode()
                print(f"[episode] saved (episode {dataset_recorder.n_recorded_episodes - 1})")
                if pose_indices is not None:
                    recovery_cursor += 1
                if event.quit or dataset_recorder.n_recorded_episodes >= target_episodes:
                    break  # done -- no point asking the operator to reset for an episode that won't run
                log_idle_to_rerun(env, dataset_recorder, "episode saved -- close the bottle by hand")
                input("Close the bottle by hand, then press Enter to continue to the next pose...")
            else:
                dataset_recorder.delete_episode()
                print(
                    f"[episode] FAILED to open the bottle after {MAX_MOTION_RETRIES} retries -- "
                    "discarding this recording."
                )
                if event.quit:
                    break
                # recovery_cursor/n_recorded_episodes didn't change, so the next loop iteration
                # retries this same pose -- every pose ends up with exactly one successful demo.
                input("Close the cap again and rotate it a bit, then press Enter to retry this pose...")
    finally:
        listener.stop()
        dataset_recorder.finish_recording()
        collected = dataset_recorder.n_recorded_episodes
        if collected >= target_episodes:
            print(f"[collect] done: {collected}/{target_episodes} episodes collected")
        else:
            print(f"[collect] stopped early with {collected}/{target_episodes} episodes -- re-run to top up")
