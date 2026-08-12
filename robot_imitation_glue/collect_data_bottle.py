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
    HOVER_HEIGHT_METERS,
    compute_hover_pose_above_bottle,
    tcp_left_to_camera,
)
from open_bottle_agent import (  # noqa: E402
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    RETREAT_SPEED,
    RIGHT_JOINT_SPEED,
    log_plan_to_rerun,
    plan_opening_motion,
)
from open_bottle_demo import (  # noqa: E402
    APPROACH_OFFSET,
    LEGS,
    pixel_to_point_on_cap_plane,
    save_touch_point_sample,
    verify_or_correct_touch_point,
)
from touch_point_detector import detect_touch_point  # noqa: E402

logger = loguru.logger

DEPTH_NUDGE_M = 0.003  # metres to press deeper per retry attempt, along -cap_normal
MAX_MOTION_RETRIES = 3  # how many times to retract and redo the whole push+legs motion


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
    success=False,
):
    """Fixed-rate replacement for `move_linear_to_tcp_pose(target_pose).wait()`: servoes the
    left arm to target_pose, recording one (obs, policy_action) frame per control cycle.
    Returns the pose actually reached (== target_pose, modulo tolerance)."""
    current_pose = env.get_robot_pose_se3()
    print("in servo function")
    print(f"current_pose: {current_pose}")

    for _ in range(max_steps):
        if _pose_reached(current_pose, target_pose, pos_tol, rot_tol_rad):
            break
        # print("finish pose reached")
        cycle_end_time = time.time() + control_period
        obs = env.get_observations()
        step_action = _tool_frame_step(current_pose, target_pose, max_translation_step, max_rotation_step_rad)
        policy_action = step_action_to_policy_action_6d(step_action)
        next_pose = policy_action_to_tcp_pose(current_pose, policy_action)
        dataset_recorder.record_step(obs, policy_action, success=success)
        # print(f"nextpose: {next_pose}")
        env.act_tcp(next_pose, time.time() + control_period)
        # print("finish next pose")
        for channel_index, voltage in enumerate(obs["bottle_sensor"]):
            rr.log(f"bottle_sensor/S{channel_index}", rr.Scalars(float(voltage)))
        precise_wait(cycle_end_time)
        current_pose = env.get_robot_pose_se3()
    else:
        logger.warning(f"servo_to_waypoint: did not reach target pose within {max_steps} steps")
    return current_pose


def run_opening_motion_with_retry(env, dataset_recorder, plan, cap_normal, hover_pose, control_period):
    """Execute the descend + push + LEGS motion, gated by sensor-verified checkpoints
    (leg_3_end, leg_6_end -- see SENSOR_CHECKPOINTS). If a checkpoint isn't satisfied, this
    retracts to the hover pose and redoes the WHOLE motion from the grip point again,
    pressing DEPTH_NUDGE_M deeper into the cap each attempt (an identical replay would very
    likely just fail identically) -- up to MAX_MOTION_RETRIES times. The retraction and
    every redo are recorded as ordinary steps, so a slipped grip and its recovery become
    part of the demonstration.

    Returns (final_pose, success): success is True only once leg_6_end -- the checkpoint
    that requires all 3 channels uncovered -- has passed.
    """
    approach_pose, push_pose, *leg_poses = plan["waypoint_poses"]

    reached_pose = approach_pose
    for attempt in range(MAX_MOTION_RETRIES + 1):
        depth_offset = -attempt * DEPTH_NUDGE_M * cap_normal
        attempt_approach = approach_pose.copy()
        attempt_approach[:3, 3] = attempt_approach[:3, 3] + depth_offset
        attempt_push = push_pose.copy()
        attempt_push[:3, 3] = attempt_push[:3, 3] + depth_offset

        if attempt > 0:
            logger.info(
                f"[verify] retry {attempt}/{MAX_MOTION_RETRIES}: redoing the opening motion "
                f"{attempt * DEPTH_NUDGE_M * 100:.1f}cm deeper"
            )

        print(f"[move] yaw + descend to grip point {np.round(attempt_approach[:3, 3], 4)}")
        servo_to_waypoint(env, dataset_recorder, attempt_approach, control_period)

        print(f"[push] straight line to {np.round(attempt_push[:3, 3], 4)}")
        reached_pose = servo_to_waypoint(env, dataset_recorder, attempt_push, control_period)

        checkpoint_failed = False
        episode_success = False
        for leg_index, leg_pose in enumerate(leg_poses):
            angle_deg, offset = LEGS[leg_index]
            attempt_leg_pose = leg_pose.copy()
            attempt_leg_pose[:3, 3] = attempt_leg_pose[:3, 3] + depth_offset
            print(f"[push] leg {leg_index + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm)")
            reached_pose = servo_to_waypoint(
                env, dataset_recorder, attempt_leg_pose, control_period, success=episode_success
            )

            event_name = f"leg_{leg_index + 2}_end"
            required_channels = SENSOR_CHECKPOINTS.get(event_name)
            if not required_channels:
                continue

            reading = env.get_observations()["bottle_sensor"]
            if is_uncovered(reading, required_channels):
                logger.info(f"[verify] {event_name}: OK (reading={np.round(reading, 2)})")
                if event_name == "leg_6_end":
                    episode_success = True
            else:
                logger.info(f"[verify] {event_name}: not uncovered (reading={np.round(reading, 2)}) -- will redo the motion")
                checkpoint_failed = True
                break

        if not checkpoint_failed:
            return reached_pose, episode_success

        if attempt == MAX_MOTION_RETRIES:
            logger.warning(f"[verify] giving up after {MAX_MOTION_RETRIES} retries -- continuing anyway")
            return reached_pose, False

        print("[verify] retracting to the hover pose to redo the opening motion")
        servo_to_waypoint(env, dataset_recorder, hover_pose.copy(), control_period)

    return reached_pose, False


def collect_data_bottle_opening(env, dataset_recorder, frequency=10, bottle_poses=None):
    if not bottle_poses:
        raise ValueError("bottle_poses is empty -- generate reachable poses first")

    rr.init("robot_imitation_glue_bottle")
    rr.spawn(memory_limit="10GB")
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)
    control_period = 1 / frequency

    intrinsics = env.get_camera_intrinsics()

    try:
        while not event.quit:
            input("Press Enter to move the LEFT arm to the retracted home pose (Ctrl+C to abort)...")
            print("[move] retreating ur_left home")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

            pose_index = dataset_recorder.n_recorded_episodes % len(bottle_poses)
            rr.set_time("pose", sequence=dataset_recorder.n_recorded_episodes)
            tcp_right_pose, _planned_cap_pose = bottle_poses[pose_index]

            print(f"\n=== pose {pose_index + 1}/{len(bottle_poses)} (episode {dataset_recorder.n_recorded_episodes}) ===")
            input("Press Enter to move the RIGHT arm to the next bottle pose (Ctrl+C to abort)...")
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

            obs = env.get_observations()
            image_rgb = obs["wrist_image_original"]
            image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
            if image_bgr.shape[:2] != (720, 1280):
                logger.warning(
                    f"wrist camera resolution is {image_bgr.shape[:2]}, but touch_point_detector's radii "
                    "were calibrated on 720p (1280x720) frames -- detection may be inaccurate. "
                    "(CameraFactory.create_wrist_camera requests 720p and no longer falls back, so this "
                    "means the frame is being resized somewhere downstream.)"
                )

            detected_pixel = detect_touch_point(image_bgr, hover_height)
            touch_pixel, corrected_pixel, verify_overlay = verify_or_correct_touch_point(image_bgr, detected_pixel)
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
                env, dataset_recorder, plan, cap_normal, hover_pose, control_period
            )

            # retreat: lift off the cap along its normal (recorded), then transit back home (not recorded -- a reset, not demonstration behaviour)
            lift_pose = env.get_robot_pose_se3().copy()
            lift_pose[:3, 3] = lift_pose[:3, 3] + (HOVER_HEIGHT_METERS - APPROACH_OFFSET) * cap_normal
            print("[move] lifting off the cap")
            servo_to_waypoint(
                env, dataset_recorder, lift_pose, control_period,
                max_translation_step=RETREAT_SPEED * control_period, success=episode_success,
            )

            dataset_recorder.save_episode()
            print(f"[episode] saved, success={episode_success}")

            print("[move] retreating ur_left home")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

            if event.quit:
                break
            input("Close the bottle by hand, then press Enter to continue to the next pose...")
    finally:
        listener.stop()
        dataset_recorder.finish_recording()
