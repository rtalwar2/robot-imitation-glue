"""
Bottle-opening demo: detect the lid touch point from the hover pose, move the gripper onto
it, then push in a straight line toward the cap center to open the lid.

Expected starting state (this script does NOT move to the start pose itself):
  - ur_right holds the bottle (grasp matching DEFAULT_TCP_RIGHT_TO_BOTTLE),
  - ur_left is already at the hover pose ~5cm above the bottle cap, camera facing the cap
    (e.g. via calibrate_and_hover_bottle.py with HOVER_HEIGHT_METERS = 0.05).

Flow:
  1. grab a frame and run the learned touch-point detector (touch_point_keypoints/);
  2. if it is confident (score >= MIN_SCORE) its point is used as is; otherwise the frame opens in
     an OpenCV window to verify or correct it (Enter/y/space accepts, r undoes the correction,
     q/Esc aborts), and that human-checked frame + point is saved to the lid_touch_point_debug
     dataset -- model-only points are never saved, so the training labels stay human;
  3. back-project the verified pixel onto the cap plane (whose pose we know through
     DEFAULT_TCP_RIGHT_TO_BOTTLE) to get the 3D touch point;
  4. plan the motion (opening_motion.plan_opening_motion): the grip point is the touch point
     shifted RADIAL_OUTWARD_OFFSET radially outward AND TANGENTIAL_OFFSET to the "left" -- along
     the cap-plane tangent, counter-clockwise about the cap's outward normal;
  5. show the image with all points + the planned push line in rerun, and wait for Enter;
  6. yaw the gripper in place about its own TCP z-axis to GRIPPER_YAW_DEG relative to the
     touch-point -> bottle-center axis (tune the constant), then move ur_left so the
     gripper is at the grip point, APPROACH_OFFSET along the cap normal;
  7. wait for Enter again, then push the gripper in a straight line PUSH_LENGTH_M long, in the
     tangent direction turned PUSH_ANGLE_DEG toward the cap centre, keeping the gripper
     orientation and height fixed;
  8. finally execute the LEGS list in order: each leg travels its offset at its angle
     counter-clockwise from the PREVIOUS leg's direction (the first is relative to the
     push direction; 90 = perpendicular/left), each descending LEG_DEPTH_STEP_M deeper below
     the cap plane than the leg before it (0.0 = all at the same height).
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
    tcp_left_to_camera,
)
from camera_utils import freeze_auto_exposure
from lid_touch_point_annotator import OUTPUT_DIR, ClickState, draw_overlay
from motion_constants import (
    GRIPPER_YAW_DEG,
    LEGS,
    LEG_DEPTH_STEP_M,
    MOVE_SPEED,
    PUSH_ANGLE_DEG,
    PUSH_LENGTH_M,
    RADIAL_OUTWARD_OFFSET,
    TANGENTIAL_OFFSET,
    TURN_DURING_LEGS,
    YAW_SPEED,
)
from opening_motion import plan_opening_motion

# True opens the verify window for every touch point, even when the learned detector is confident.
VERIFY_EVERY_TOUCH_POINT = False


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


def verify_or_correct_touch_point(image_bgr, detected_pixel, regrab=None):
    """Show the frame in an OpenCV window so the detection can be verified or corrected.

    Click to set a corrected touch point (green cross; click again to move it), press
    Enter/y/space to accept the current point, r to undo the correction, q/Esc to abort.
    When the detection failed (detected_pixel is None), a click is required before
    accepting.

    `regrab`, if given, is a zero-argument callable returning a fresh
    (image_bgr, detected_pixel) pair -- typically a new camera frame with the detector re-run
    on it. Pressing t then replaces the frame being annotated, which is the fix for a blurred,
    badly exposed or occluded grab: without it the only options are to accept a bad frame or
    abort the episode. Any correction clicked so far is discarded, since its coordinates refer
    to the old frame.

    Returns (final_pixel, corrected_click_or_None, overlay_image, image_bgr, detected_pixel).
    The last two are returned rather than reused from the caller's arguments because `regrab`
    may have replaced both: the caller must save and project the frame that was actually
    annotated, and record the detection that belongs to it -- pairing a fresh image with the
    previous frame's detection would quietly corrupt the touch-point dataset.
    """
    window_name = "Verify touch point (click to correct)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    click_state = ClickState()
    cv2.setMouseCallback(window_name, click_state.on_mouse_event)

    if detected_pixel is None:
        print("[verify] detection FAILED -- click the touch point manually, then press Enter")
    retake_key = ", t=retake image" if regrab is not None else ""
    print(f"[verify] keys: click=correct, Enter/y/space=accept, r=undo correction{retake_key}, q/Esc=abort")

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
            return final_pixel, click_state.point, overlay, image_bgr, detected_pixel
        elif key == ord("r"):
            click_state.point = None
        elif key == ord("t") and regrab is not None:
            image_bgr, detected_pixel = regrab()
            # The old click was in the old frame's pixel coordinates, so it no longer means
            # anything -- drop it rather than silently carrying it onto a different image.
            click_state.point = None
            print(f"[verify] retook the image -- detection is now {detected_pixel}")
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted at touch-point verification -- not moving.")


_learned_detector = None


def learned_touch_point(image_bgr):
    """The learned detector's touch pixel and its confidence score (loaded on first use)."""
    global _learned_detector
    if _learned_detector is None:
        from touch_point_keypoints.detector import TouchPointKeypointDetector

        _learned_detector = TouchPointKeypointDetector()
    x, y, score = _learned_detector.predict(image_bgr)
    return (int(round(x)), int(round(y))), score


def resolve_touch_point(image_bgr, grab_frame=None):
    """The touch pixel to use: the learned detector's, with the click window only when it is unsure.

    Below touch_point_keypoints.detector.MIN_SCORE (or always, with VERIFY_EVERY_TOUCH_POINT) the
    model's point opens in verify_or_correct_touch_point as the proposal; `grab_frame`, a callable
    returning a fresh BGR frame, enables its 't' retake.

    Returns (touch_pixel, overlay, image_bgr, detected_pixel, human_checked). Only human-checked
    results belong in the lid_touch_point_debug dataset -- saving the model's own guesses as labels
    would train it on itself.
    """
    from touch_point_keypoints.detector import MIN_SCORE

    detected, score = learned_touch_point(image_bgr)
    if score >= MIN_SCORE and not VERIFY_EVERY_TOUCH_POINT:
        print(f"[detect] learned touch point {detected} (score {score:.2f}) -- used without asking")
        return detected, draw_overlay(image_bgr, None, detected), image_bgr, detected, False
    reason = f"has score {score:.2f} (< {MIN_SCORE})" if score < MIN_SCORE else f"(score {score:.2f}), check asked for"
    print(f"[detect] learned touch point {detected} {reason} -- verify or click it")

    def regrab():
        frame = grab_frame()
        return frame, learned_touch_point(frame)[0]

    touch_pixel, _, overlay, image_bgr, detected = verify_or_correct_touch_point(
        image_bgr, detected, regrab=regrab if grab_frame is not None else None)
    return touch_pixel, overlay, image_bgr, detected, True


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
    # Deferred on purpose: open_bottle_agent imports this module's motion constants, so a
    # module-level import of it back here is an unresolvable circular import -- both this
    # script and the analyze fork fail at import time. Flat module name, as every other
    # bottle_experiment import in this file.
    from open_bottle_agent import LEFT_HOME_JOINTS, LEFT_TRANSIT_JOINT_SPEED

    rr.init("open_bottle_demo", spawn=True)
    # 720p to match touch_point_detector's radius prior (calibrated on 720p frames);
    # intrinsics_matrix() returns the intrinsics for this stream resolution.
    camera = Realsense(resolution=Realsense.RESOLUTION_720, fps=15, enable_depth=False, enable_pointcloud=False)
    freeze_auto_exposure(camera)
    intrinsics = camera.intrinsics_matrix()
    ur_left = URrtde(ip_address=LEFT_ROBOT_IP)
    # ur_left.move_to_joint_configuration([ 0.06967844 ,-1.40953115 ,-1.61241627, -1.67278638 , 1.54791272 , 3.03650355],joint_speed=0.02).wait()
    # ur_left.move_to_joint_configuration([ 0.06980903 ,-0.46889468, -1.61281288 ,-1.67641511 , 1.54615736 , 0], joint_speed=0.2).wait()
    ur_left.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

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

    def grab_frame():
        return ImageConverter.from_numpy_int_format(camera.get_rgb_image_as_int()).image_in_opencv_format

    touch_pixel, verify_overlay, image_bgr, detected_pixel, human_checked = resolve_touch_point(image_bgr, grab_frame)
    if human_checked:  # grow the labeled dataset with frames a person looked at
        save_touch_point_sample(image_bgr, verify_overlay, hover_height, touch_pixel, detected_pixel)
    corrected_pixel = touch_pixel if tuple(touch_pixel) != tuple(detected_pixel) else None
    if corrected_pixel is not None:
        print(f"[verify] using corrected touch point {touch_pixel} (detection was {detected_pixel})")

    touch_point = pixel_to_point_on_cap_plane(touch_pixel, intrinsics, camera_pose_in_base, cap_pose)

    plan = plan_opening_motion(cap_pose, touch_point)
    grip_point, push_end, push_target = plan["grip_point"], plan["push_end"], plan["push_target"]
    approach_position, leg_ends, leg_targets = plan["approach_position"], plan["leg_ends"], plan["leg_targets"]

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
    print(f"[detect] push {PUSH_LENGTH_M * 100:.1f}cm at {PUSH_ANGLE_DEG:.1f} deg toward the cap centre  push target (base frame)={np.round(push_target, 4)}")
    input("Check the detection in rerun. Press Enter to move the gripper to the grip point (Ctrl+C to abort)...")

    # yaw the gripper about its own TCP z-axis (rotate in place at the hover position)
    # to GRIPPER_YAW_DEG relative to the touch-point -> bottle-center axis
    yawed_rotation = plan["yawed_rotation"]
    yaw_pose = np.eye(4)
    yaw_pose[:3, :3] = yawed_rotation
    yaw_pose[:3, 3] = start_pose[:3, 3]
    print(f"[move] yawing gripper to {GRIPPER_YAW_DEG:.0f} deg relative to the touch->center axis")
    ur_left.move_linear_to_tcp_pose(yaw_pose, linear_speed=YAW_SPEED).wait()

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

    # chained legs, each at its angle relative to the previous leg's direction; the gripper turns with the
    # path (opening_motion) -- during the leg, or in place first (YAW_SPEED) with TURN_DURING_LEGS off
    for i, ((angle_deg, offset), turn_pose, leg_pose) in enumerate(zip(LEGS, plan["leg_turn_poses"], plan["waypoint_poses"][2:])):
        print(f"[push] leg {i + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm, {(i + 1) * LEG_DEPTH_STEP_M * 100:.1f}cm deeper than the push) to {np.round(leg_pose[:3, 3], 4)}")
        if not TURN_DURING_LEGS:
            ur_left.move_linear_to_tcp_pose(turn_pose, linear_speed=YAW_SPEED).wait()
        ur_left.move_linear_to_tcp_pose(leg_pose, linear_speed=MOVE_SPEED).wait()
    # ur_left.move_to_joint_configuration([ 0.06967844 ,-1.40953115 ,-1.61241627, -1.67278638 , 1.54791272 , 3.03650355])
    ur_left.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()

    print("[done] opening motion finished")
