"""
Automated data-collection agent for the bottle-opening task, based on open_bottle_demo.py.

Seeded pose generation: samples N_POSES random poses for ur_right (holding the bottle)
subject to constraints --
  - the bottle is never upside down: the cap normal stays within MAX_TILT_DEG of vertical
    (tilt is allowed up to that);
  - the cap center lies inside the CAP_POSITION_MIN/MAX box (ur_left base frame);
  - the right gripper points generally toward ur_left, never backwards: its TCP z-axis
    stays within GRIPPER_AIM_MAX_DEG of the right-to-left base direction;
  - ur_right can reach the pose (is_tcp_pose_reachable);
  - ur_left can "reach the bottle": the hover pose AND every waypoint of the full opening
    motion (yawed approach at the grip point, push target, all LEGS) are reachable. Since
    the real touch point is only known after detection, the motion is checked for
    N_TOUCH_ANGLE_SAMPLES hypothetical touch points spread around the rim
    (BOTTLE_RIM_RADIUS), so the pose stays feasible wherever the seam points.

Then loops over the accepted poses. For each pose:
  1. ur_left retreats to LEFT_HOME_JOINTS, ur_right moves the bottle to the pose;
  2. ur_left moves to the hover pose above the cap;
  3. frame capture + the learned touch-point detector; when it is unsure, the same
     click-to-correct verification window as open_bottle_demo, and that human-checked frame +
     point is saved to the lid_touch_point_debug dataset;
  4. the planned motion is shown in rerun (per-pose timeline), Enter confirms;
  5. ur_left descends to the grip point WITH the GRIPPER_YAW_DEG yaw applied in the same
     move (rotation and translation interpolated together in one moveL);
  6. push + LEGS, exactly as in open_bottle_demo (constants imported from there, so
     tuning the demo tunes the agent too);
  7. ur_left lifts off and retreats home, then waits: close the bottle by hand and press
     Enter to continue to the next pose.

Reachability caveat: is_tcp_pose_reachable checks per-waypoint IK/safety limits only --
it does not check the linear paths between waypoints or arm-arm collisions, so keep a
hand near the e-stop as usual.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path

import motion_constants
import numpy as np
import opening_motion
import rerun as rr
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from calibrate_and_hover_bottle import (
    BASE_LEFT_TO_BASE_RIGHT,
    DEFAULT_TCP_RIGHT_TO_BOTTLE,
    HOVER_HEIGHT_METERS,
    LEFT_ROBOT_IP,
    RIGHT_ROBOT_IP,
    compute_hover_pose_above_bottle,
    get_bottle_cap_pose_in_base_left,
    orthonormalize_rotation,
    tcp_left_to_camera,
)
from camera_utils import freeze_auto_exposure
from motion_constants import APPROACH_OFFSET, LEGS, MOVE_SPEED
from open_bottle_demo import (
    pixel_to_point_on_cap_plane,
    project_point_to_pixel,
    resolve_touch_point,
    save_touch_point_sample,
)
# plan_opening_motion / make_pose are re-exported from here for older importers (collect_data_bottle,
# check_hover_center_bias); the geometry itself lives in opening_motion.py.
from opening_motion import make_pose, plan_opening_motion, rotation_about_axis  # noqa: F401 (re-exported)

RANDOM_SEED = 0
N_POSES = 5  # how many valid poses to generate (and then execute)
MAX_SAMPLE_ATTEMPTS = 2000  # give up on generation after this many rejected samples

# constraint parameters for the sampled bottle poses
MAX_TILT_DEG = 30.0  # max angle between the cap normal and vertical (90+ would be sideways/upside down)
CAP_POSITION_MIN = np.array([0.30, -0.15, 0.30])  # cap-center sampling box, ur_left base frame -- TUNE to your workspace
CAP_POSITION_MAX = np.array([0.50, 0.15, 0.45])

# the right gripper must point generally toward ur_left, never backwards: max angle
# between ur_right's TCP z-axis (the gripper's pointing direction) and the right-to-left
# base direction. Generous by default -- tilting is fine, only the general direction is
# enforced, not a direct aim at ur_left's base.
GRIPPER_AIM_MAX_DEG = 60.0

# hypothetical touch points used for the left-arm reachability check (the real touch
# point is only known after detection, so check the whole rim)
BOTTLE_RIM_RADIUS = 0.04  # metres, roughly where on the cap the touch point sits
N_TOUCH_ANGLE_SAMPLES = 8

LEFT_HOME_JOINTS = [ 0.06980903 ,-0.46889468, -1.61281288 ,-1.67641511 , 1.54615736 , 0]

# Neutral via-pose for ur_right, passed through before every move to a sampled bottle pose so the
# arm always approaches from the same configuration instead of swinging directly between two
# arbitrary poses. Stored as joints rather than a TCP pose on purpose: IK could otherwise pick a
# different elbow/wrist branch and land in the very configuration the via-point exists to avoid.
# The corresponding TCP pose in ur_right's base frame, for reference (taught with the old 219.5 mm TCP
# offset; ur_right's TCP is the bare flange since 2026-10-08, so the flange sits ~22 cm behind this):
#   [[-0.158  0.023 -0.987 -0.591]
#    [-0.987  0.01   0.158 -0.045]
#    [ 0.014  1.     0.021  0.305]
#    [ 0.     0.     0.     1.   ]]
RIGHT_NEUTRAL_JOINTS = [-0.177, -1.799, 2.27, -0.494, 1.555, 0.017]
RIGHT_JOINT_SPEED = 0.2  # rad/s for ur_right's pose-to-pose moves
LEFT_TRANSIT_JOINT_SPEED = 0.2  # rad/s for ur_left's home <-> hover transits
RETREAT_SPEED = 0.03  # m/s for lifting off the cap after the motion


def sample_cap_pose(rng):
    """Random cap pose in ur_left's base frame: position uniform in the box, tilt from
    vertical uniform in [0, MAX_TILT_DEG] about a random horizontal axis (so never upside
    down), plus a random yaw about the cap's own axis."""
    position = rng.uniform(CAP_POSITION_MIN, CAP_POSITION_MAX)
    tilt = np.radians(rng.uniform(0.0, MAX_TILT_DEG))
    azimuth = rng.uniform(0.0, 2 * np.pi)
    yaw = rng.uniform(0.0, 2 * np.pi)

    tilt_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    rotation = rotation_about_axis(tilt_axis, tilt) @ rotation_about_axis(np.array([0.0, 0.0, 1.0]), yaw)
    return make_pose(rotation, position)


def is_opening_motion_reachable(cap_pose, ur_left):
    """Can ur_left do the whole job for this cap pose: hover pose + all motion waypoints,
    for touch points anywhere around the rim?"""
    hover_pose = compute_hover_pose_above_bottle(cap_pose)
    if not ur_left.is_tcp_pose_reachable(hover_pose):
        return False
    cap_center = cap_pose[:3, 3]
    cap_x, cap_y = cap_pose[:3, 0], cap_pose[:3, 1]
    for k in range(N_TOUCH_ANGLE_SAMPLES):
        angle = 2 * np.pi * k / N_TOUCH_ANGLE_SAMPLES
        touch_point = cap_center + BOTTLE_RIM_RADIUS * (np.cos(angle) * cap_x + np.sin(angle) * cap_y)
        plan = plan_opening_motion(cap_pose, touch_point)
        for waypoint_pose in plan["waypoint_poses"]:
            if not ur_left.is_tcp_pose_reachable(waypoint_pose):
                return False
    return True


# Generated pose lists are cached here, keyed by everything that decides them (see _pose_cache_key), so a
# script restart loads the list in a second instead of re-running hundreds of IK checks. Set
# BOTTLE_POSES_NO_CACHE=1 to force a fresh generation (e.g. after changing a robot's safety limits, which
# the key cannot see).
POSE_CACHE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "pose_cache"


def _pose_cache_key(n_poses, ur_left, ur_right, rng, blacklist):
    """Fingerprint of every input of generate_reachable_bottle_poses: the request (count, blacklist, the
    rng's exact state), the geometry it uses (grasp transform, base offset, hover height, sampling limits,
    every motion constant -- the reachability check plans the whole motion), both controllers' TCP offsets
    (IK depends on them), and the SOURCE of the functions that sample and check, so a code change also
    invalidates the cache."""
    functions = (generate_reachable_bottle_poses, sample_cap_pose, is_opening_motion_reachable,
                 opening_motion.plan_opening_motion, compute_hover_pose_above_bottle)
    payload = {
        "n_poses": n_poses, "blacklist": sorted(blacklist), "rng": rng.bit_generator.state,
        "grasp": DEFAULT_TCP_RIGHT_TO_BOTTLE.tolist(), "bases": BASE_LEFT_TO_BASE_RIGHT.tolist(),
        "hover": HOVER_HEIGHT_METERS,
        "sampling": [MAX_SAMPLE_ATTEMPTS, MAX_TILT_DEG, CAP_POSITION_MIN.tolist(), CAP_POSITION_MAX.tolist(),
                     GRIPPER_AIM_MAX_DEG, BOTTLE_RIM_RADIUS, N_TOUCH_ANGLE_SAMPLES],
        "motion": {k: v for k, v in vars(motion_constants).items() if k.isupper()},
        "tcp_offsets": [list(ur_left.rtde_control.getTCPOffset()), list(ur_right.rtde_control.getTCPOffset())],
        "code": [inspect.getsource(f) for f in functions],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def generate_reachable_bottle_poses(n_poses, ur_left, ur_right, rng, blacklist=()):
    """generate_reachable_bottle_poses_uncached, through an on-disk cache (POSE_CACHE_DIR).

    A cache hit returns exactly what generation would: the same poses in the same order, the blacklist
    applied the same way (it is part of the key, so changing it regenerates), and `rng` left in the same
    state generation would have left it in.
    """
    key = _pose_cache_key(n_poses, ur_left, ur_right, rng, blacklist)
    cache_file = POSE_CACHE_DIR / f"poses_{key}.npz"
    if cache_file.exists() and os.environ.get("BOTTLE_POSES_NO_CACHE") != "1":
        cached = np.load(cache_file)
        rng.bit_generator.state = json.loads(str(cached["rng_state_after"]))
        poses = list(zip(cached["tcp_right"], cached["cap"]))
        print(f"[generate] loaded {len(poses)}/{n_poses} poses from cache {cache_file.name} "
              f"(blacklist {sorted(blacklist)}; BOTTLE_POSES_NO_CACHE=1 regenerates)")
        return poses
    poses = generate_reachable_bottle_poses_uncached(n_poses, ur_left, ur_right, rng, blacklist)
    POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, tcp_right=np.array([t for t, _ in poses]).reshape(-1, 4, 4),
             cap=np.array([c for _, c in poses]).reshape(-1, 4, 4),
             rng_state_after=json.dumps(rng.bit_generator.state, default=int))
    print(f"[generate] cached {len(poses)} poses as {cache_file.name}")
    return poses


def generate_reachable_bottle_poses_uncached(n_poses, ur_left, ur_right, rng, blacklist=()):
    """Sample cap poses until n_poses pass all constraints. Returns a list of
    (tcp_right_pose, cap_pose) tuples.

    `blacklist` holds 0-based *pose indices* to skip. The index counts every pose that passes
    the constraints, blacklisted ones included, so the numbering does not shift when something
    is excluded: blacklisting index 0 and asking for 5 poses returns what would have been
    indices 1-5, and keeps sampling until it has 5. That stability holds because a pose is only
    rejected after `sample_cap_pose` has drawn it, and rejecting it consumes no further
    randomness -- so every other index still refers to exactly the same pose it did before.

    Indices are only meaningful for a given rng seed. Since each split is seeded differently,
    a blacklist belongs to one split and means nothing in another.
    """
    blacklist = set(blacklist)
    accepted = []
    pose_index = 0  # counts constraint-passing poses, including blacklisted ones
    for attempt in range(1, MAX_SAMPLE_ATTEMPTS + 1):
        if len(accepted) == n_poses:
            break
        cap_pose = sample_cap_pose(rng)

        cap_pose_in_base_right = np.linalg.inv(BASE_LEFT_TO_BASE_RIGHT) @ cap_pose
        tcp_right = cap_pose_in_base_right @ np.linalg.inv(DEFAULT_TCP_RIGHT_TO_BOTTLE)
        tcp_right[:3, :3] = orthonormalize_rotation(tcp_right[:3, :3])

        # gripper must point generally toward ur_left: ur_left's base sits along -x of
        # ur_right's base frame (the bases share their orientation), so the gripper's
        # z-axis must stay within GRIPPER_AIM_MAX_DEG of that direction
        direction_to_left = np.array([-1.0, 0.0, 0.0])
        if tcp_right[:3, 2].dot(direction_to_left) < np.cos(np.radians(GRIPPER_AIM_MAX_DEG)):
            continue

        if not ur_right.is_tcp_pose_reachable(tcp_right):
            continue
        if not is_opening_motion_reachable(cap_pose, ur_left):
            continue

        index = pose_index
        pose_index += 1
        if index in blacklist:
            print(f"[generate] pose index {index} is blacklisted -- skipping "
                  f"(cap center={np.round(cap_pose[:3, 3], 3)})")
            continue

        accepted.append((tcp_right, cap_pose))
        print(f"[generate] pose {len(accepted)}/{n_poses} (index {index}) accepted after {attempt} attempts: "
              f"cap center={np.round(cap_pose[:3, 3], 3)} tilt from vertical="
              f"{np.degrees(np.arccos(np.clip(cap_pose[2, 2], -1, 1))):.0f} deg")
    else:
        print(f"[generate] WARNING: only {len(accepted)}/{n_poses} poses found in {MAX_SAMPLE_ATTEMPTS} attempts "
              "-- widen the box / tilt limits, or check the reachability constraints")

    unused = blacklist - set(range(pose_index))
    if unused:
        # A blacklist entry beyond the last index generated never fired, so it is either a typo
        # or left over from a different seed / n_poses -- either way it is silently doing nothing.
        print(f"[generate] WARNING: blacklist indices {sorted(unused)} were never reached "
              f"(only {pose_index} poses generated) -- wrong split's blacklist, or a typo?")
    return accepted


def log_plan_to_rerun(image_rgb, intrinsics, camera_pose_in_base, cap_pose, touch_point, touch_pixel, plan):
    cap_center = cap_pose[:3, 3]
    grip_pixel = project_point_to_pixel(plan["grip_point"], intrinsics, camera_pose_in_base)
    push_end_pixel = project_point_to_pixel(plan["push_end"], intrinsics, camera_pose_in_base)
    cap_center_pixel = project_point_to_pixel(cap_center, intrinsics, camera_pose_in_base)
    leg_end_pixels = [project_point_to_pixel(p, intrinsics, camera_pose_in_base) for p in plan["leg_ends"]]

    rr.log("camera/image", rr.Image(image_rgb, rr.ColorModel.RGB))
    rr.log("camera/image/touch_point", rr.Points2D([touch_pixel], colors=[(0, 255, 0)], radii=6.0, labels=["touch point"]))
    rr.log("camera/image/grip_point", rr.Points2D([grip_pixel], colors=[(255, 0, 0)], radii=6.0, labels=["grip point"]))
    rr.log("camera/image/bottle_center", rr.Points2D([cap_center_pixel], colors=[(255, 255, 0)], radii=6.0, labels=["bottle center"]))
    rr.log("camera/image/planned_push", rr.LineStrips2D([[grip_pixel, push_end_pixel] + leg_end_pixels], colors=[(255, 0, 255)]))

    rr.log("world/touch_point", rr.Points3D([touch_point], colors=[(0, 255, 0)], radii=0.003, labels=["touch point"]))
    rr.log("world/grip_point", rr.Points3D([plan["grip_point"]], colors=[(255, 0, 0)], radii=0.003, labels=["grip point"]))
    rr.log("world/bottle_center", rr.Points3D([cap_center], colors=[(255, 255, 0)], radii=0.003, labels=["bottle center"]))
    rr.log("world/planned_push", rr.LineStrips3D([[plan["approach_position"], plan["push_target"]] + plan["leg_targets"]], colors=[(255, 0, 255)]))


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    rr.init("open_bottle_agent", spawn=True)

    camera = Realsense(resolution=Realsense.RESOLUTION_720, fps=15, enable_depth=False, enable_pointcloud=False)
    freeze_auto_exposure(camera)
    intrinsics = camera.intrinsics_matrix()

    ur_left = URrtde(ip_address=LEFT_ROBOT_IP)
    ur_right = URrtde(ip_address=RIGHT_ROBOT_IP)

    print(f"[agent] seed={RANDOM_SEED} -- generating {N_POSES} reachable bottle poses...")
    poses = generate_reachable_bottle_poses(N_POSES, ur_left, ur_right, rng)
    if not poses:
        raise SystemExit("No reachable poses found -- adjust the sampling box / constraints.")

    print(f"[agent] moving ur_left to the home configuration")
    ur_left.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

    for pose_index, (tcp_right, planned_cap_pose) in enumerate(poses):
        rr.set_time("pose", sequence=pose_index)
        print(f"\n=== pose {pose_index + 1}/{len(poses)} ===")
        input("Press Enter to move the RIGHT arm to the next bottle pose (Ctrl+C to abort)...")
        ur_right.move_to_tcp_pose(tcp_right, joint_speed=RIGHT_JOINT_SPEED).wait()

        # live cap pose from the right arm's actual TCP (should match the planned one)
        cap_pose = get_bottle_cap_pose_in_base_left(DEFAULT_TCP_RIGHT_TO_BOTTLE, ur_right)
        cap_center = cap_pose[:3, 3]
        cap_normal = cap_pose[:3, 2]

        hover_pose = compute_hover_pose_above_bottle(cap_pose)
        print(f"[move] ur_left to hover pose above the cap at {np.round(hover_pose[:3, 3], 4)}")
        ur_left.move_to_tcp_pose(hover_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

        start_pose = ur_left.get_tcp_pose()
        camera_pose_in_base = start_pose @ tcp_left_to_camera
        hover_height = float(cap_normal.dot(start_pose[:3, 3] - cap_center))
        print(f"[start] TCP is {hover_height * 100:.1f}cm above the cap plane (expected ~{HOVER_HEIGHT_METERS * 100:.0f}cm)")

        image_rgb = camera.get_rgb_image_as_int()
        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        def grab_frame():
            return ImageConverter.from_numpy_int_format(camera.get_rgb_image_as_int()).image_in_opencv_format

        # the learned detector's point, with the verify/click window only when it is unsure
        touch_pixel, verify_overlay, image_bgr, detected_pixel, human_checked = resolve_touch_point(image_bgr, grab_frame)
        image_rgb = ImageConverter.from_opencv_format(image_bgr).image_in_numpy_int_format  # 't' may have retaken it
        if human_checked:  # only frames a person looked at become training labels
            save_touch_point_sample(image_bgr, verify_overlay, hover_height, touch_pixel, detected_pixel)
        if tuple(touch_pixel) != tuple(detected_pixel):
            print(f"[verify] using corrected touch point {touch_pixel} (detection was {detected_pixel})")

        touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose_in_base, cap_pose)
        plan = plan_opening_motion(cap_pose, touch_point)
        log_plan_to_rerun(image_rgb, intrinsics, camera_pose_in_base, cap_pose, touch_point, touch_pixel, plan)

        print(f"[plan] touch point={np.round(touch_point, 4)}  grip point={np.round(plan['grip_point'], 4)}")
        input("Check the plan in rerun. Press Enter to run the opening motion (Ctrl+C to abort)...")

        # descend to the grip point with the yaw applied in the same move (one moveL
        # interpolates the rotation and translation together)
        approach_pose, push_pose, *leg_poses = plan["waypoint_poses"]
        print(f"[move] yaw + descend to {np.round(plan['approach_position'], 4)}")
        ur_left.move_linear_to_tcp_pose(approach_pose, linear_speed=MOVE_SPEED).wait()

        print(f"[push] straight line to {np.round(plan['push_target'], 4)}")
        ur_left.move_linear_to_tcp_pose(push_pose, linear_speed=MOVE_SPEED).wait()
        for leg_index, leg_pose in enumerate(leg_poses):
            angle_deg, offset = LEGS[leg_index]
            print(f"[push] leg {leg_index + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm)")
            ur_left.move_linear_to_tcp_pose(leg_pose, linear_speed=MOVE_SPEED).wait()

        # retreat: lift off the cap along its normal, then transit back home
        lift_pose = ur_left.get_tcp_pose()
        lift_pose[:3, 3] = lift_pose[:3, 3] + (HOVER_HEIGHT_METERS - APPROACH_OFFSET) * cap_normal
        print("[move] lifting off the cap and retreating home")
        ur_left.move_linear_to_tcp_pose(lift_pose, linear_speed=RETREAT_SPEED).wait()
        ur_left.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

        input("Close the bottle by hand, then press Enter to continue to the next pose...")

    print(f"\n[done] all {len(poses)} poses executed")
