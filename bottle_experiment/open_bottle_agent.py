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
  3. frame capture + touch-point detection + the same click-to-correct verification
     window as open_bottle_demo -- every frame + verified point is saved to the
     lid_touch_point_debug dataset (this loop is the dataset builder);
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

import numpy as np
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
from open_bottle_demo import (
    APPROACH_OFFSET,
    GRIPPER_YAW_DEG,
    LEGS,
    MOVE_SPEED,
    PUSH_OVERSHOOT,
    PUSH_TARGET_TANGENTIAL_OFFSET,
    RADIAL_OUTWARD_OFFSET,
    TANGENTIAL_OFFSET,
    compute_yawed_gripper_orientation,
    pixel_to_point_on_cap_plane,
    project_point_to_pixel,
    rotation_about_axis,
    save_touch_point_sample,
    verify_or_correct_touch_point,
)
from touch_point_detector import detect_touch_point

RANDOM_SEED = 0
N_POSES = 5  # how many valid poses to generate (and then execute)
MAX_SAMPLE_ATTEMPTS = 2000  # give up on generation after this many rejected samples

# constraint parameters for the sampled bottle poses
MAX_TILT_DEG = 30.0  # max angle between the cap normal and vertical (90+ would be sideways/upside down)
CAP_POSITION_MIN = np.array([0.35, -0.15, 0.30])  # cap-center sampling box, ur_left base frame -- TUNE to your workspace
CAP_POSITION_MAX = np.array([0.50, 0.05, 0.45])

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
RIGHT_JOINT_SPEED = 0.2  # rad/s for ur_right's pose-to-pose moves
LEFT_TRANSIT_JOINT_SPEED = 0.2  # rad/s for ur_left's home <-> hover transits
RETREAT_SPEED = 0.03  # m/s for lifting off the cap after the motion


def make_pose(rotation, position):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def plan_opening_motion(cap_pose, touch_point):
    """The same geometry as open_bottle_demo's main, as a reusable function: from the cap
    pose and a (real or hypothetical) touch point, compute the grip/push/leg points and
    the yawed gripper orientation. Returns a dict; 'waypoint_poses' holds the 4x4 poses
    the gripper actually visits, in order (approach, push target, leg targets)."""
    cap_center = cap_pose[:3, 3]
    cap_normal = cap_pose[:3, 2]

    outward = touch_point - cap_center
    outward = outward / np.linalg.norm(outward)
    tangent = np.cross(cap_normal, outward)
    tangent = tangent / np.linalg.norm(tangent)

    grip_point = touch_point + RADIAL_OUTWARD_OFFSET * outward + TANGENTIAL_OFFSET * tangent
    push_end = cap_center + PUSH_TARGET_TANGENTIAL_OFFSET * tangent
    push_direction = push_end - grip_point
    push_direction = push_direction / np.linalg.norm(push_direction)
    approach_position = grip_point + APPROACH_OFFSET * cap_normal
    push_target = push_end + APPROACH_OFFSET * cap_normal + PUSH_OVERSHOOT * push_direction

    leg_ends, leg_targets = [], []
    leg_direction = push_direction
    leg_end, leg_target = push_end, push_target
    for angle_deg, offset in LEGS:
        leg_direction = rotation_about_axis(cap_normal, np.radians(angle_deg)) @ leg_direction
        leg_direction -= cap_normal * cap_normal.dot(leg_direction)
        leg_direction /= np.linalg.norm(leg_direction)
        leg_end = leg_end + offset * leg_direction
        leg_target = leg_target + offset * leg_direction
        leg_ends.append(leg_end)
        leg_targets.append(leg_target)

    yawed_rotation = compute_yawed_gripper_orientation(cap_normal, -outward, GRIPPER_YAW_DEG)
    waypoint_poses = [make_pose(yawed_rotation, p) for p in [approach_position, push_target, *leg_targets]]

    return {
        "grip_point": grip_point,
        "push_end": push_end,
        "push_target": push_target,
        "leg_ends": leg_ends,
        "leg_targets": leg_targets,
        "approach_position": approach_position,
        "yawed_rotation": yawed_rotation,
        "waypoint_poses": waypoint_poses,
    }


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


def generate_reachable_bottle_poses(n_poses, ur_left, ur_right, rng):
    """Sample cap poses until n_poses pass all constraints. Returns a list of
    (tcp_right_pose, cap_pose) tuples."""
    accepted = []
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

        accepted.append((tcp_right, cap_pose))
        print(f"[generate] pose {len(accepted)}/{n_poses} accepted after {attempt} attempts: "
              f"cap center={np.round(cap_pose[:3, 3], 3)} tilt from vertical="
              f"{np.degrees(np.arccos(np.clip(cap_pose[2, 2], -1, 1))):.0f} deg")
    else:
        print(f"[generate] WARNING: only {len(accepted)}/{n_poses} poses found in {MAX_SAMPLE_ATTEMPTS} attempts "
              "-- widen the box / tilt limits, or check the reachability constraints")
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

        detected_pixel = detect_touch_point(image_bgr, hover_height)
        touch_pixel, corrected_pixel, verify_overlay, image_bgr, detected_pixel = verify_or_correct_touch_point(
            image_bgr, detected_pixel
        )
        save_touch_point_sample(image_bgr, verify_overlay, hover_height, touch_pixel, detected_pixel)
        if corrected_pixel is not None:
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
