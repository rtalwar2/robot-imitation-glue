"""Hand-tune the opening motion one stage at a time: the push first, then each leg -- no sensors, you judge.

ur_right holds the bottle at its neutral pose (RIGHT_NEUTRAL_JOINTS) for the whole session; ur_left goes
home only at the start and at the end, and waits at the hover pose in between. Each run:
  1. at the hover pose the learned detector finds the touch point (click window only if it is unsure)
     and the plan -- push + the first N legs -- is drawn in rerun;
  2. Enter: ur_left yaws and descends to the grip point; anything else: stay at hover;
  3. Enter: push + N legs, then lift RETRACT_LIFT_METERS off the cap; anything else: lift without
     pushing;
  4. back to the hover pose (slide back, then un-yaw in place).
Close the cap by hand and edit motion_constants.py between runs: the constants are RELOADED every run,
so no restart is needed.

    python tune_opening_motion.py            # start with the push only (N = 0)
    python tune_opening_motion.py --legs 2   # push + the first two LEGS entries

At the run prompt: Enter runs with the current N, a number sets N (e.g. 1 = push + leg 1), q quits.
Each leg turns the gripper (LEG_YAW_MODE) -- in place before the leg, or during it (TURN_DURING_LEGS).
At the grip-point gate, a = run the same push + legs AIR_LIFT_M above the cap (no contact), to check
the motion and the printed leg speeds first.
"""

import argparse
import importlib
import time

import numpy as np
import rerun as rr
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from calibrate_and_hover_bottle import (
    DEFAULT_TCP_RIGHT_TO_BOTTLE,
    LEFT_ROBOT_IP,
    RIGHT_ROBOT_IP,
    compute_hover_pose_above_bottle,
    get_bottle_cap_pose_in_base_left,
    tcp_left_to_camera,
)
from camera_utils import freeze_auto_exposure
import motion_constants
import opening_motion
from open_bottle_agent import LEFT_HOME_JOINTS, LEFT_TRANSIT_JOINT_SPEED, RIGHT_JOINT_SPEED, RIGHT_NEUTRAL_JOINTS, log_plan_to_rerun
from open_bottle_demo import pixel_to_point_on_cap_plane, resolve_touch_point


AIR_LIFT_M = 0.04  # how far above the real path the 'a' (in-air) run flies


def rotation_angle(a, b):
    return float(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))


def run_motion(ur_left, mc, plan, motion, lift):
    """Push + legs at MOVE_SPEED, each pose shifted by `lift` (zero for the real run); every leg is timed."""
    shift = lambda pose: opening_motion.make_pose(pose[:3, :3], pose[:3, 3] + lift)  # noqa: E731
    push, *legs = motion
    print(f"[move] push -> {np.round(push[:3, 3], 4)}")
    ur_left.move_linear_to_tcp_pose(shift(push), linear_speed=mc.MOVE_SPEED).wait()
    previous = push
    for i, leg in enumerate(legs):
        turn, length = rotation_angle(previous, leg), float(np.linalg.norm(leg[:3, 3] - previous[:3, 3]))
        if not mc.TURN_DURING_LEGS:
            ur_left.move_linear_to_tcp_pose(shift(plan["leg_turn_poses"][i]), linear_speed=mc.YAW_SPEED).wait()
        start = time.monotonic()
        ur_left.move_linear_to_tcp_pose(shift(leg), linear_speed=mc.MOVE_SPEED).wait()
        took = time.monotonic() - start
        during = f"turning {np.degrees(turn):.0f} deg during it" if mc.TURN_DURING_LEGS else "turned before it"
        print(f"[move] leg {i + 1}: {length * 1000:.0f} mm, {during}, took {took:.2f} s "
              f"= {length / took * 1000:.0f} mm/s average (MOVE_SPEED {mc.MOVE_SPEED * 1000:.0f} mm/s, incl. accel)")
        previous = leg


def describe(mc, n_legs):
    legs = ", ".join(f"({a:g} deg, {length * 100:g} cm)" for a, length in mc.LEGS[:n_legs]) or "none"
    return (f"push {mc.PUSH_LENGTH_M * 100:.2f} cm at {mc.PUSH_ANGLE_DEG:g} deg | approach {mc.APPROACH_OFFSET * 100:g} cm, "
            f"grip offsets {mc.RADIAL_OUTWARD_OFFSET * 100:g} cm out / {mc.TANGENTIAL_OFFSET * 100:g} cm tangential, "
            f"yaw {mc.GRIPPER_YAW_DEG:g} deg | legs {legs} (depth step {mc.LEG_DEPTH_STEP_M * 1000:g} mm)")


# ur_left's joints at the hover pose, recorded the first time it gets there (from home, so unwound) and
# returned to EXACTLY afterwards. A Cartesian move to the hover pose takes the shortest rotation, and once
# the wrist has turned more than 180 deg away (yaw + leg turns) that shortest way continues the turn: on
# 2026-10-09 it left joint 6 at -337 deg -- the right orientation, wound a full turn, camera cable included.
# A joint move back to recorded joints can only unwind the way it wound.
_hover_joints = None
# Joint speed for going back to _hover_joints. That move is mostly the wrist unwinding the yaw + leg turns
# (200+ deg), which at LEFT_TRANSIT_JOINT_SPEED (0.2 rad/s) took ~20 s; it starts 5 cm above the cap.
RETURN_JOINT_SPEED = 0.6


def go_to_hover(ur_left, hover_target):
    global _hover_joints
    if _hover_joints is None:
        ur_left.move_to_tcp_pose(hover_target, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        _hover_joints = ur_left.get_joint_configuration()
    else:
        ur_left.move_to_joint_configuration(_hover_joints, joint_speed=RETURN_JOINT_SPEED).wait()


def back_to_hover(ur_left, mc, pose_now, cap_normal, hover):
    """Lift off along the cap normal, slide back over the hover position keeping the yaw, then un-yaw by
    returning to the recorded hover JOINTS (see _hover_joints) rather than with a Cartesian rotation."""
    lift = pose_now.copy()
    lift[:3, 3] += mc.RETRACT_LIFT_METERS * cap_normal
    ur_left.move_linear_to_tcp_pose(lift, linear_speed=mc.MOVE_SPEED).wait()
    ur_left.move_linear_to_tcp_pose(opening_motion.make_pose(lift[:3, :3], hover[:3, 3]), linear_speed=mc.MOVE_SPEED).wait()
    go_to_hover(ur_left, hover)


def plan_at_hover(ur_left, ur_right, camera, intrinsics, n_legs):
    """Reload the constants, go to hover, find the touch point and plan; draws the first n legs in rerun."""
    mc = importlib.reload(motion_constants)
    plan_opening_motion = importlib.reload(opening_motion).plan_opening_motion
    if n_legs > len(mc.LEGS):
        print(f"[tune] only {len(mc.LEGS)} legs defined -- running all of them")
        n_legs = len(mc.LEGS)

    cap_pose = get_bottle_cap_pose_in_base_left(DEFAULT_TCP_RIGHT_TO_BOTTLE, ur_right)
    go_to_hover(ur_left, compute_hover_pose_above_bottle(cap_pose))
    hover = ur_left.get_tcp_pose()
    camera_pose = hover @ tcp_left_to_camera

    def grab_frame():
        return ImageConverter.from_numpy_int_format(camera.get_rgb_image_as_int()).image_in_opencv_format

    touch_pixel, _, image_bgr, _, _ = resolve_touch_point(grab_frame(), grab_frame)
    touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose, cap_pose)
    plan = plan_opening_motion(cap_pose, touch_point)
    image_rgb = ImageConverter.from_opencv_format(image_bgr).image_in_numpy_int_format
    shown = {**plan, "leg_ends": plan["leg_ends"][:n_legs], "leg_targets": plan["leg_targets"][:n_legs]}  # only what runs
    log_plan_to_rerun(image_rgb, intrinsics, camera_pose, cap_pose, touch_point, touch_pixel, shown)

    print(f"[tune] {describe(mc, n_legs)}")
    return mc, plan, cap_pose, hover, n_legs


def run_once(ur_left, ur_right, camera, intrinsics, n_legs):
    mc, plan, cap_pose, hover, n_legs = plan_at_hover(ur_left, ur_right, camera, intrinsics, n_legs)
    if input("Check the plan in rerun. Enter = descend to the grip point, anything else = stay at hover: ").strip():
        return

    approach, *motion = plan["waypoint_poses"][: 2 + n_legs]  # approach, push target, then the first n legs
    ur_left.move_linear_to_tcp_pose(opening_motion.make_pose(approach[:3, :3], hover[:3, 3]), linear_speed=mc.YAW_SPEED).wait()
    print(f"[move] approach -> {np.round(approach[:3, 3], 4)}")
    ur_left.move_linear_to_tcp_pose(approach, linear_speed=mc.MOVE_SPEED).wait()

    stages = "push" + (f" + {n_legs} leg(s)" if n_legs else "")
    answer = input(f"At the grip point. Enter = {stages}, a = the same {AIR_LIFT_M * 100:.0f} cm above the cap "
                   "(no contact), anything else = back to hover without pushing: ").strip()
    if answer not in ("", "a"):
        back_to_hover(ur_left, mc, ur_left.get_tcp_pose(), cap_pose[:3, 2], hover)
        return
    lift = AIR_LIFT_M * cap_pose[:3, 2] if answer == "a" else np.zeros(3)
    if answer == "a":
        ur_left.move_linear_to_tcp_pose(opening_motion.make_pose(approach[:3, :3], approach[:3, 3] + lift),
                                        linear_speed=mc.MOVE_SPEED).wait()
    run_motion(ur_left, mc, plan, motion, lift)
    back_to_hover(ur_left, mc, ur_left.get_tcp_pose(), cap_pose[:3, 2], hover)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legs", type=int, default=0, help="legs to run after the push (0 = push only)")
    args = parser.parse_args()

    rr.init("tune_opening_motion", spawn=True)
    camera = Realsense(resolution=Realsense.RESOLUTION_720, fps=15, enable_depth=False, enable_pointcloud=False)
    freeze_auto_exposure(camera)
    intrinsics = camera.intrinsics_matrix()
    ur_left = URrtde(ip_address=LEFT_ROBOT_IP)
    ur_right = URrtde(ip_address=RIGHT_ROBOT_IP)
    ur_left.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
    ur_right.move_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED).wait()

    n_legs, run = args.legs, 1
    while True:
        answer = input(f"\n[run {run}] close the cap. Enter = push + {n_legs} leg(s), a number = change that, q = quit: ").strip()
        if answer == "q":
            break
        if answer.isdigit():
            n_legs = int(answer)
            continue
        run_once(ur_left, ur_right, camera, intrinsics, n_legs)
        run += 1
    ur_left.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()


if __name__ == "__main__":
    main()
