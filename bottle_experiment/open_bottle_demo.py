"""
Bottle-opening demo: detect the lid touch point from the hover pose, move the gripper onto
it, then push in a straight line toward the cap center to open the lid.

Expected starting state (this script does NOT move to the start pose itself):
  - ur_right holds the bottle (grasp matching DEFAULT_TCP_RIGHT_TO_BOTTLE),
  - ur_left is already at the hover pose ~5cm above the bottle cap, camera facing the cap
    (e.g. via calibrate_and_hover_bottle.py with HOVER_HEIGHT_METERS = 0.05).

Flow:
  1. grab a frame and run the touch-point detection;
  2. show the detection in an OpenCV window for verification -- click to correct it if
     it's wrong (Enter/y/space accepts, r undoes the correction, q/Esc aborts). The frame
     + verified point are saved to the lid_touch_point_debug dataset either way, growing
     the labeled set (a failed detection can be labeled by clicking manually);
  3. back-project the verified pixel onto the cap plane (whose pose we know through
     DEFAULT_TCP_RIGHT_TO_BOTTLE) to get the 3D touch point;
  4. compute the grip point: the touch point shifted RADIAL_OUTWARD_OFFSET radially
     outward (away from the center, past the cap edge) AND TANGENTIAL_OFFSET to the
     "left" -- along the cap-plane tangent, in the direction the old circular motion went
     (counter-clockwise about the cap's outward normal);
  5. show the image with all points + the planned push line in rerun, and wait for Enter;
  6. yaw the gripper in place about its own TCP z-axis to GRIPPER_YAW_DEG relative to the
     touch-point -> bottle-center axis (tune the constant), then move ur_left so the
     gripper is at the grip point, APPROACH_OFFSET along the cap normal;
  7. wait for Enter again, then push the gripper in a straight line from the grip point
     to the push end point -- the cap center shifted PUSH_TARGET_TANGENTIAL_OFFSET along
     the same tangential axis (plus PUSH_OVERSHOOT) -- keeping the gripper orientation and
     height fixed;
  8. finally execute the LEGS list in order: each leg travels its offset at its angle
     counter-clockwise from the PREVIOUS leg's direction (the first is relative to the
     push direction; 90 = perpendicular/left), all at the same height.
"""

import json
import os

import cv2
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
    orthonormalize_rotation,
    tcp_left_to_camera,
)
from camera_utils import freeze_auto_exposure
from lid_touch_point_annotator import OUTPUT_DIR, ClickState, draw_overlay
from touch_point_detector import detect_touch_point
import time

APPROACH_OFFSET = -0.022  # metres above the grip point where the gripper stops (negative = press below the cap plane)
RADIAL_OUTWARD_OFFSET = 0.01  # metres from the touch point radially outward (away from the cap center)
# metres to the "left" of the touch point: along the cap-plane tangent at the touch point,
# in the counter-clockwise direction about the cap's outward normal (the direction the old
# circular motion went). Negate if "left" turns out to be the other way on the real bottle.
TANGENTIAL_OFFSET = -0.04
# metres to the "left" of the cap center (same tangential axis as TANGENTIAL_OFFSET):
# the push line ends here instead of at the center itself
PUSH_TARGET_TANGENTIAL_OFFSET = 0.30
# metres to keep pushing past the push end point along the same line (0.0 = stop exactly
# there; negative = stop short of it)
PUSH_OVERSHOOT = -0.27
# legs of the opening motion, executed in order after the push, each as
# (angle_deg, offset_m). The angle is measured counter-clockwise about the cap's outward
# normal RELATIVE TO THE PREVIOUS leg's direction (the first entry is relative to the
# push direction; 90 = exactly perpendicular/"to the left"). The offset is the distance
# travelled. All legs run in the cap plane at the pressed height. Add/remove/tune entries
# freely -- the planned path preview, prints, and execution all follow this list.
LEGS = [
    (45.0, 0.03),  # second leg
    (20.0, 0.06),  # third leg
    (30.0, 0.04),  # fourth leg
    (30.0, 0.03),  # fifth leg
    (30.0, 0.01),  # fifth leg
]
# yaw of the gripper about its own TCP z-axis (which faces the cap), applied by rotating
# in place BEFORE descending to the grip point. Defined relative to the touch-point ->
# bottle-center axis: at 0 the gripper's x-axis points from the touch point toward the
# center; positive rotates counter-clockwise about the cap's outward normal.
GRIPPER_YAW_DEG = 90
MOVE_SPEED = 0.01  # m/s, same slow speed as the annotation tool


def pixel_to_point_on_cap_plane(pixel_xy, intrinsics, camera_pose_in_base, cap_pose_in_base):
    """Back-project an image pixel onto the bottle-cap plane, in ur_left base frame.

    The detection gives only a 2D pixel; its depth is recovered by intersecting the
    camera ray with the cap plane (origin = cap center, normal = cap z-axis), both known
    in the base frame through ur_right's kinematics + DEFAULT_TCP_RIGHT_TO_BOTTLE.
    """
    ray_camera = np.linalg.inv(intrinsics) @ np.array([pixel_xy[0], pixel_xy[1], 1.0])
    ray_base = camera_pose_in_base[:3, :3] @ ray_camera
    origin = camera_pose_in_base[:3, 3]

    cap_center = cap_pose_in_base[:3, 3]
    cap_normal = cap_pose_in_base[:3, 2]
    t = cap_normal.dot(cap_center - origin) / cap_normal.dot(ray_base)
    return origin + t * ray_base


def project_point_to_pixel(point_in_base, intrinsics, camera_pose_in_base):
    """Project a 3D point (base frame) back into image pixel coordinates, for rerun display."""
    point_in_camera = np.linalg.inv(camera_pose_in_base) @ np.append(point_in_base, 1.0)
    pixel = intrinsics @ point_in_camera[:3]
    return pixel[:2] / pixel[2]


def rotation_about_axis(axis, angle_rad):
    """Rodrigues rotation matrix about a unit axis."""
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def compute_yawed_gripper_orientation(cap_normal, reference_direction, yaw_deg):
    """Gripper orientation facing the cap (TCP z-axis = -cap normal) with a prescribed yaw
    about that z-axis: at yaw 0 the gripper's x-axis points along `reference_direction`
    (touch point -> bottle center); positive yaw rotates it counter-clockwise about the
    cap's outward normal."""
    z_axis = -cap_normal
    x_axis = rotation_about_axis(cap_normal, np.radians(yaw_deg)) @ reference_direction
    x_axis = x_axis - z_axis * z_axis.dot(x_axis)  # keep exactly perpendicular to z
    x_axis *= -1
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return orthonormalize_rotation(np.column_stack([x_axis, y_axis, z_axis]))


def verify_or_correct_touch_point(image_bgr, detected_pixel):
    """Show the frame in an OpenCV window so the detection can be verified or corrected.

    Click to set a corrected touch point (green cross; click again to move it), press
    Enter/y/space to accept the current point, r to undo the correction, q/Esc to abort.
    When the detection failed (detected_pixel is None), a click is required before
    accepting. Returns (final_pixel, corrected_click_or_None, overlay_image).
    """
    window_name = "Verify touch point (click to correct)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    click_state = ClickState()
    cv2.setMouseCallback(window_name, click_state.on_mouse_event)

    if detected_pixel is None:
        print("[verify] detection FAILED -- click the touch point manually, then press Enter")
    print("[verify] keys: click=correct, Enter/y/space=accept, r=undo correction, q/Esc=abort")

    while True:
        overlay = draw_overlay(image_bgr, click_state.point, detected_pixel)
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord("y"), ord(" ")):
            final_pixel = click_state.point if click_state.point is not None else detected_pixel
            if final_pixel is None:
                print("[verify] no touch point set yet -- click one first")
                continue
            cv2.destroyWindow(window_name)
            return final_pixel, click_state.point, overlay
        elif key == ord("r"):
            click_state.point = None
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted at touch-point verification -- not moving.")


def save_touch_point_sample(image_bgr, overlay, hover_height, click_xy, detected_xy):
    """Append this frame + human-verified touch point to the lid_touch_point_debug dataset,
    in the same sample_NNNN format the annotation tool writes (so all evaluation tooling
    keeps working on the combined set)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_index = len([n for n in os.listdir(OUTPUT_DIR) if n.endswith("_raw.jpg")])
    prefix = os.path.join(OUTPUT_DIR, f"sample_{sample_index:04d}")
    cv2.imwrite(f"{prefix}_raw.jpg", image_bgr)
    cv2.imwrite(f"{prefix}_overlay.jpg", overlay)
    with open(f"{prefix}.json", "w") as f:
        json.dump({"height": hover_height, "click_xy": list(click_xy), "detected_xy": list(detected_xy) if detected_xy is not None else None}, f)
    print(f"[dataset] saved {prefix}  height={hover_height:.3f}  click={click_xy}  detected={detected_xy}")


if __name__ == "__main__":
    rr.init("open_bottle_demo", spawn=True)
    # 720p to match touch_point_detector's radius prior (calibrated on 720p frames);
    # intrinsics_matrix() returns the intrinsics for this stream resolution.
    camera = Realsense(resolution=Realsense.RESOLUTION_720, fps=15, enable_depth=False, enable_pointcloud=False)
    freeze_auto_exposure(camera)
    intrinsics = camera.intrinsics_matrix()
    ur_left = URrtde(ip_address=LEFT_ROBOT_IP)
    # ur_left.move_to_joint_configuration([ 0.06967844 ,-1.40953115 ,-1.61241627, -1.67278638 , 1.54791272 , 3.03650355],joint_speed=0.02).wait()
    ur_left.move_to_joint_configuration([ 0.06980903 ,-0.46889468, -1.61281288 ,-1.67641511 , 1.54615736 , 0], joint_speed=0.2).wait()
    # time.sleep(2)
    ur_right = URrtde(ip_address=RIGHT_ROBOT_IP)
    ur_right.rtde_control.teachMode()
    input("Move the right arm where ever you want")
    ur_right.rtde_control.endTeachMode()

    cap_pose = get_bottle_cap_pose_in_base_left(DEFAULT_TCP_RIGHT_TO_BOTTLE, ur_right)
    hover_pose = compute_hover_pose_above_bottle(cap_pose, 0.1)
    print(f"hover pose: {hover_pose}")
    print(f"Moving ur_left above the bottle cap at {hover_pose[:3, 3]}")

    ur_left.move_to_tcp_pose(hover_pose, joint_speed=0.2).wait()
    cap_center = cap_pose[:3, 3]
    cap_normal = cap_pose[:3, 2]  # points outward from the cap, toward the gripper

    start_pose = ur_left.get_tcp_pose()
    camera_pose_in_base = start_pose @ tcp_left_to_camera

    # actual height of the TCP above the cap plane -- the radius prior needs this, and it
    # should be ~0.05 if the robot is at the expected start pose.
    hover_height = float(cap_normal.dot(start_pose[:3, 3] - cap_center))
    print(f"[start] TCP is {hover_height * 100:.1f}cm above the cap plane (expected ~5cm)")

    image_rgb = camera.get_rgb_image_as_int()
    image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

    detected_pixel = detect_touch_point(image_bgr, hover_height)

    # verify/correct interactively, and grow the labeled dataset with this frame:
    # click_xy is the human-verified ground truth (the accepted detection, or the correction)
    touch_pixel, corrected_pixel, verify_overlay = verify_or_correct_touch_point(image_bgr, detected_pixel)
    save_touch_point_sample(image_bgr, verify_overlay, hover_height, touch_pixel, detected_pixel)
    if corrected_pixel is not None:
        print(f"[verify] using corrected touch point {touch_pixel} (detection was {detected_pixel})")

    touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose_in_base, cap_pose)

    # radially-outward direction in the cap plane: from the cap center through the touch point
    outward = touch_point - cap_center
    outward /= np.linalg.norm(outward)

    # tangent at the touch point, counter-clockwise about the cap's outward normal --
    # "left" of the touch point as seen along the old circular motion's direction
    tangent = np.cross(cap_normal, outward)
    tangent /= np.linalg.norm(tangent)

    # the gripper targets the touch point shifted radially outward (past the cap edge)
    # AND tangentially to the left
    grip_point = touch_point + RADIAL_OUTWARD_OFFSET * outward + TANGENTIAL_OFFSET * tangent

    # opening motion: straight-line push from the grip point toward the push end point --
    # the cap center shifted along the same tangential axis -- at the gripper's height
    push_end = cap_center + PUSH_TARGET_TANGENTIAL_OFFSET * tangent
    push_direction = push_end - grip_point
    push_direction /= np.linalg.norm(push_direction)
    approach_position = grip_point + APPROACH_OFFSET * cap_normal
    push_target = push_end + APPROACH_OFFSET * cap_normal + PUSH_OVERSHOOT * push_direction

    # legs after the push: each direction is the previous leg's direction rotated by the
    # leg's angle counter-clockwise about the cap normal, all in the cap plane at the
    # pressed height. leg_ends are the cap-plane points (for the image preview),
    # leg_targets the pressed-height points the robot actually moves to.
    leg_ends, leg_targets = [], []
    leg_direction = push_direction
    leg_end, leg_target = push_end, push_target
    for angle_deg, offset in LEGS:
        leg_direction = rotation_about_axis(cap_normal, np.radians(angle_deg)) @ leg_direction
        leg_direction -= cap_normal * cap_normal.dot(leg_direction)  # keep exactly in-plane
        leg_direction /= np.linalg.norm(leg_direction)
        leg_end = leg_end + offset * leg_direction
        leg_target = leg_target + offset * leg_direction
        leg_ends.append(leg_end)
        leg_targets.append(leg_target)

    # --- rerun: image with detected touch point, grip point, push target + 3D view ---
    grip_pixel = project_point_to_pixel(grip_point, intrinsics, camera_pose_in_base)
    cap_center_pixel = project_point_to_pixel(cap_center, intrinsics, camera_pose_in_base)
    rr.log("camera/image", rr.Image(image_rgb, rr.ColorModel.RGB))
    rr.log("camera/image/touch_point", rr.Points2D([touch_pixel], colors=[(0, 255, 0)], radii=6.0, labels=["touch point"]))
    if corrected_pixel is not None and detected_pixel is not None:
        rr.log("camera/image/detection_uncorrected", rr.Points2D([detected_pixel], colors=[(128, 128, 128)], radii=6.0, labels=["detection (overruled)"]))
    rr.log("camera/image/grip_point", rr.Points2D([grip_pixel], colors=[(255, 0, 0)], radii=6.0, labels=["grip point"]))
    push_end_pixel = project_point_to_pixel(push_end, intrinsics, camera_pose_in_base)
    rr.log("camera/image/bottle_center", rr.Points2D([cap_center_pixel], colors=[(255, 255, 0)], radii=6.0, labels=["bottle center"]))
    rr.log("camera/image/push_end", rr.Points2D([push_end_pixel], colors=[(0, 128, 255)], radii=6.0, labels=["push end"]))
    leg_end_pixels = [project_point_to_pixel(p, intrinsics, camera_pose_in_base) for p in leg_ends]
    rr.log("camera/image/planned_push", rr.LineStrips2D([[grip_pixel, push_end_pixel] + leg_end_pixels], colors=[(255, 0, 255)]))

    rr.log("world/touch_point", rr.Points3D([touch_point], colors=[(0, 255, 0)], radii=0.003, labels=["touch point"]))
    rr.log("world/grip_point", rr.Points3D([grip_point], colors=[(255, 0, 0)], radii=0.003, labels=["grip point"]))
    rr.log("world/bottle_center", rr.Points3D([cap_center], colors=[(255, 255, 0)], radii=0.003, labels=["bottle center"]))
    rr.log("world/push_end", rr.Points3D([push_end], colors=[(0, 128, 255)], radii=0.003, labels=["push end"]))
    rr.log("world/planned_push", rr.LineStrips3D([[approach_position, push_target] + leg_targets], colors=[(255, 0, 255)]))

    print(f"[detect] touch pixel={touch_pixel}  touch point (base frame)={np.round(touch_point, 4)}")
    print(f"[detect] grip point ({RADIAL_OUTWARD_OFFSET * 100:.1f}cm outward + {TANGENTIAL_OFFSET * 100:.1f}cm left of touch point, base frame)={np.round(grip_point, 4)}")
    print(f"[detect] push end ({PUSH_TARGET_TANGENTIAL_OFFSET * 100:.1f}cm left of cap center)  push target (base frame)={np.round(push_target, 4)}")
    input("Check the detection in rerun. Press Enter to move the gripper to the grip point (Ctrl+C to abort)...")

    # yaw the gripper about its own TCP z-axis (rotate in place at the hover position)
    # to GRIPPER_YAW_DEG relative to the touch-point -> bottle-center axis
    yawed_rotation = compute_yawed_gripper_orientation(cap_normal, -outward, GRIPPER_YAW_DEG)
    yaw_pose = np.eye(4)
    yaw_pose[:3, :3] = yawed_rotation
    yaw_pose[:3, 3] = start_pose[:3, 3]
    print(f"[move] yawing gripper to {GRIPPER_YAW_DEG:.0f} deg relative to the touch->center axis")
    ur_left.move_linear_to_tcp_pose(yaw_pose, linear_speed=0.2).wait()

    # descend to the grip point (TANGENTIAL_OFFSET left of the touch point, APPROACH_OFFSET
    # along the cap normal), keeping the yawed orientation
    approach_pose = np.eye(4)
    approach_pose[:3, :3] = yawed_rotation
    approach_pose[:3, 3] = approach_position
    print(f"[move] descending to {np.round(approach_position, 4)}")
    ur_left.move_linear_to_tcp_pose(approach_pose, linear_speed=MOVE_SPEED).wait()

    input("Press Enter to start the opening motion (Ctrl+C to abort)...")
    # a single moveL is already a straight Cartesian line -- no waypoints needed
    push_pose = approach_pose.copy()
    push_pose[:3, 3] = push_target
    print(f"[push] straight line to {np.round(push_target, 4)}")
    ur_left.move_linear_to_tcp_pose(push_pose, linear_speed=MOVE_SPEED).wait()

    # chained legs, each at its angle relative to the previous leg's direction
    leg_pose = push_pose.copy()
    for i, ((angle_deg, offset), leg_target) in enumerate(zip(LEGS, leg_targets)):
        leg_pose = leg_pose.copy()
        leg_pose[:3, 3] = leg_target
        print(f"[push] leg {i + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm) to {np.round(leg_target, 4)}")
        ur_left.move_linear_to_tcp_pose(leg_pose, linear_speed=MOVE_SPEED).wait()
    ur_left.move_to_joint_configuration([ 0.06967844 ,-1.40953115 ,-1.61241627, -1.67278638 , 1.54791272 , 3.03650355])

    print("[done] opening motion finished")
