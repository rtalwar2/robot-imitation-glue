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
from robot_imitation_glue.hardware.bottle_sensor import SENSOR_CHECKPOINTS, is_uncovered
from robot_imitation_glue.utils import precise_wait

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOTTLE_EXPERIMENT_PATH = str(_REPO_ROOT / "bottle_experiment")
if _BOTTLE_EXPERIMENT_PATH not in sys.path:
    sys.path.insert(0, _BOTTLE_EXPERIMENT_PATH)

from calibrate_and_hover_bottle import (  # noqa: E402
    compute_hover_pose_above_bottle,
    tcp_left_to_camera,
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
from open_bottle_demo import (  # noqa: E402
    LEGS,
    pixel_to_point_on_cap_plane,
    save_touch_point_sample,
    verify_or_correct_touch_point,
)
from touch_point_detector import detect_touch_point  # noqa: E402

logger = loguru.logger

DEPTH_NUDGE_M = 0.003  # metres to press deeper per retry attempt, along -cap_normal
MAX_MOTION_RETRIES = 3  # how many times to retract and redo the whole push+legs motion
RETRACT_LIFT_METERS = 0.05  # metres to lift off the cap along its normal before retrying


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

    Returns None if event_name isn't gated, True if gated and satisfied, False if gated and
    not satisfied. Raises if the sensor feed has never delivered a real sample (see
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

    if is_uncovered(reading, required_channels):
        logger.info(f"[verify] {event_name}: OK (reading={np.round(reading, 2)})")
        return True
    logger.info(f"[verify] {event_name}: not uncovered (reading={np.round(reading, 2)}) -- will redo the motion")
    return False


def _build_motion_segments(plan):
    """The motion as an ordered list of (label, base_target_pose, checkpoint_event_or_None).

    Segment order: approach (yaw+descend, ungated) -> push (push_end) -> leg2 (ungated) ->
    leg3 (leg_3_end) -> leg4 (ungated) -> leg5 (ungated) -> leg6 (leg_6_end).
    """
    approach_pose, push_pose, *leg_poses = plan["waypoint_poses"]
    segments = [("approach", approach_pose, None), ("push", push_pose, "push_end")]
    for leg_index, leg_pose in enumerate(leg_poses):
        angle_deg, offset = LEGS[leg_index]
        event_name = f"leg_{leg_index + 2}_end"
        checkpoint = event_name if event_name in SENSOR_CHECKPOINTS else None
        label = f"leg{leg_index + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm)"
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
    (push_end, leg_3_end, leg_6_end -- see SENSOR_CHECKPOINTS), pressing DEPTH_NUDGE_M
    deeper into the cap on every retry (an identical replay would very likely just fail
    identically) -- up to MAX_MOTION_RETRIES times.

    Recovery is cascading rather than always a full restart: on a failed checkpoint, retract
    to the hover pose and RE-CHECK the PRECEDING checkpoint's sensor.
      - If that earlier checkpoint still holds, only that slipped -- resume the motion
        directly from the earlier checkpoint's own waypoint (deeper), skipping the segments
        before it (they're still fine, no need to redo them).
      - If the earlier checkpoint has ALSO lost its grip, the problem runs deeper than just
        this leg -- restart the whole motion from the grip point.
      - The first checkpoint (push_end) has no earlier checkpoint to fall back on, so its
        failure always triggers a full restart.
    The retraction and every (partial or full) redo are recorded as ordinary steps, so a
    slipped grip and its recovery become part of the demonstration.

    Returns (final_pose, success): success is True only once leg_6_end -- the checkpoint
    for the last channel to uncover -- has passed.
    """
    segments = _build_motion_segments(plan)

    attempt = 0
    resume_index = 0
    reached_pose = segments[0][1]
    episode_success = False

    while True:
        depth_offset = -attempt * DEPTH_NUDGE_M * cap_normal

        if attempt > 0:
            logger.info(
                f"[verify] attempt {attempt}/{MAX_MOTION_RETRIES}: resuming from "
                f"'{segments[resume_index][0]}', {attempt * DEPTH_NUDGE_M * 100:.1f}cm deeper"
            )

        episode_success = False
        checkpoint_failed_index = None
        for segment_index in range(resume_index, len(segments)):
            label, base_pose, checkpoint_name = segments[segment_index]
            target_pose = base_pose.copy()
            target_pose[:3, 3] = target_pose[:3, 3] + depth_offset
            print(f"[move] {label} -> {np.round(target_pose[:3, 3], 4)}")
            reached_pose = servo_to_waypoint(
                env, dataset_recorder, target_pose, control_period,
                label=f"attempt {attempt}: {label}",
            )

            if checkpoint_name is None:
                continue
            checkpoint_ok = _check_checkpoint(env, checkpoint_name)
            if checkpoint_ok is False:
                checkpoint_failed_index = segment_index
                break
            if checkpoint_ok is True and checkpoint_name == "leg_6_end":
                episode_success = True

        if checkpoint_failed_index is None:
            return reached_pose, episode_success

        if attempt == MAX_MOTION_RETRIES:
            logger.warning(f"[verify] giving up after {MAX_MOTION_RETRIES} retries -- continuing anyway")
            return reached_pose, False

        # Lift just far enough to disengage the gripper from the cap, so the sensor reads the cap's
        # own state rather than whatever the gripper is holding it in. Going all the way back to the
        # hover pose would work too but wastes most of the travel: the re-check only needs the cap
        # released, and every centimetre of it is recorded as demonstration steps.
        print(f"[verify] lifting {RETRACT_LIFT_METERS * 100:.0f}cm off the cap to check recovery state")
        lift_pose = env.get_robot_pose_se3().copy()
        lift_pose[:3, 3] = lift_pose[:3, 3] + RETRACT_LIFT_METERS * cap_normal
        servo_to_waypoint(
            env, dataset_recorder, lift_pose, control_period,
            label=f"lifting {RETRACT_LIFT_METERS * 100:.0f}cm to retry",
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

        attempt += 1

    return reached_pose, False


def collect_data_bottle_opening(env, dataset_recorder, frequency=10, bottle_poses=None, n_episodes=None):
    """Collect `n_episodes` successful demonstrations, one per generated bottle pose by default.

    Defaults to len(bottle_poses), so each pose is used exactly once when nothing fails. Failed
    episodes are discarded without advancing the counter, so a pose that fails is retried rather
    than skipped -- which also means a pose that can never succeed will loop indefinitely; watch
    the printed progress.

    Counting successful episodes in the dataset (rather than loop iterations) makes this resumable:
    the recorder picks up n_recorded_episodes from an existing dataset, so a re-run tops up to the
    target instead of starting over.
    """
    if not bottle_poses:
        raise ValueError("bottle_poses is empty -- generate reachable poses first")
    target_episodes = len(bottle_poses) if n_episodes is None else n_episodes

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
            env.capture_ft_bias()
            log_idle_to_rerun(env, dataset_recorder, "at home -- FT bias captured")

            pose_index = dataset_recorder.n_recorded_episodes % len(bottle_poses)
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
            env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)

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

            def grab_and_detect():
                """A fresh wrist frame with the touch-point detector run on it.

                The arm is stationary at the hover pose throughout verification, so re-grabbing
                does not invalidate hover_height, camera_pose_in_base or cap_pose.
                """
                image_bgr = ImageConverter.from_numpy_int_format(
                    env.get_observations()["wrist_image_original"]
                ).image_in_opencv_format
                if image_bgr.shape[:2] != (720, 1280):
                    logger.warning(
                        f"wrist camera resolution is {image_bgr.shape[:2]}, but touch_point_detector's radii "
                        "were calibrated on 720p (1280x720) frames -- detection may be inaccurate. "
                        "(CameraFactory.create_wrist_camera requests 720p and no longer falls back, so this "
                        "means the frame is being resized somewhere downstream.)"
                    )
                return image_bgr, detect_touch_point(image_bgr, hover_height)

            image_bgr, detected_pixel = grab_and_detect()
            # regrab lets 't' replace a blurred/occluded frame instead of forcing a choice
            # between accepting a bad grab and aborting the episode.
            touch_pixel, corrected_pixel, verify_overlay, image_bgr, detected_pixel = verify_or_correct_touch_point(
                image_bgr, detected_pixel, regrab=grab_and_detect
            )
            image_rgb = ImageConverter.from_opencv_format(image_bgr).image_in_numpy_int_format
            save_touch_point_sample(image_bgr, verify_overlay, hover_height, touch_pixel, detected_pixel)
            if corrected_pixel is not None:
                print(f"[verify] using corrected touch point {touch_pixel} (detection was {detected_pixel})")

            touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose_in_base, cap_pose)
            plan = plan_opening_motion(cap_pose, touch_point)
            log_plan_to_rerun(image_rgb, intrinsics, camera_pose_in_base, cap_pose, touch_point, touch_pixel, plan)

            print(f"[plan] touch point={np.round(touch_point, 4)}  grip point={np.round(plan['grip_point'], 4)}")
            input("Check the plan in rerun. Press Enter to run the opening motion (Ctrl+C to abort)...")

            dataset_recorder.start_episode()

            _final_pose, episode_success = run_opening_motion_with_retry(
                env, dataset_recorder, plan, cap_normal, control_period
            )

            # retreat: return to the hover pose above the cap centre (recorded), then transit back
            # home (not recorded -- a reset, not demonstration behaviour). Ending at the hover pose
            # rather than rising straight up from wherever leg 6 finished gives every demonstration
            # the same terminal state relative to the bottle, instead of one that varies with which
            # leg the motion ended on and how far the retries pushed it.
            print("[move] returning to the hover pose above the cap")
            servo_to_waypoint(
                env, dataset_recorder, hover_pose.copy(), control_period,
                max_translation_step=RETREAT_SPEED * control_period,
                label="returning to hover above the cap",
            )

            print("[move] retreating ur_left home")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

            # every frame so far was recorded with the dataset default next.success=False --
            # the true outcome is only known now, so label the whole episode retroactively.
            dataset_recorder.set_episode_success(episode_success)

            if episode_success:
                dataset_recorder.save_episode()
                print(f"[episode] saved (episode {dataset_recorder.n_recorded_episodes - 1})")
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
                # n_recorded_episodes didn't change, so the next loop iteration retries this same
                # pose -- every generated pose ends up with exactly one successful recorded demo.
                input("Close the cap again and rotate it a bit, then press Enter to retry this pose...")
    finally:
        listener.stop()
        dataset_recorder.finish_recording()
        collected = dataset_recorder.n_recorded_episodes
        if collected >= target_episodes:
            print(f"[collect] done: {collected}/{target_episodes} episodes collected")
        else:
            print(f"[collect] stopped early with {collected}/{target_episodes} episodes -- re-run to top up")
