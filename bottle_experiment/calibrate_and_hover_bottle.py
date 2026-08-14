"""
Live calibration + hover tool for the bottle-opening setup.

Shows a live camera view (ur_left's wrist camera) with:
  - the ArUco marker detection drawn live (outline, id, pose axes) whenever the marker is
    visible, so the marker/camera placement can be checked before calibrating;
  - after calibration (or with the DEFAULT_TCP_RIGHT_TO_BOTTLE transform), the bottle cap
    center projected live into the image (yellow, same as open_bottle_demo.py shows it)
    plus the cap's pose axes -- recomputed every frame from ur_right's live TCP pose, so
    moving the bottle (freedrive) moves the projected center with it: a live check that
    the grasp transform is right.

Keys (in the live window):
  c     calibrate the grasp transform from the current marker detection (marker must be
        visible; ur_right must be holding the bottle). Prints the matrix to paste into
        DEFAULT_TCP_RIGHT_TO_BOTTLE, and starts using it immediately.
  h     move ur_left to the hover pose HOVER_HEIGHT_METERS above the bottle cap
        (disables ur_left freedrive first).
  f     toggle freedrive on ur_right (reposition the bottle and watch the center track).
  g     toggle freedrive on ur_left (reposition the camera view).
  q/Esc quit (freedrive is switched off on both robots).
"""

import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    detect_aruco_markers,
    get_poses_of_aruco_markers,
    visualize_aruco_detections,
)
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_dataset_tools.data_parsers.pose import Pose
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from camera_utils import freeze_auto_exposure

LEFT_ROBOT_IP = "10.42.0.163"  # arm with gripper, wrist camera mounted on it
RIGHT_ROBOT_IP = "10.42.0.162"  # arm holding the bottle

camera_pose_path = "/home/rtalwar/robot-imitation-glue/calibration_2026-07-15_13:20:38/results_n=7/camera_pose_Andreff.json"
with open(camera_pose_path, "r") as f:
    # eye-in-hand: transform from ur_left's TCP frame to the wrist camera frame.
    tcp_left_to_camera = Pose.model_validate_json(f.read()).as_homogeneous_matrix()

# Static offset between the two robot bases: ur_right's base sits 91cm along +x from
# ur_left's base (same orientation), expressed in ur_left's base frame.
BASE_LEFT_TO_BASE_RIGHT = np.eye(4)
BASE_LEFT_TO_BASE_RIGHT[0, 3] = 0.91

MARKER_SIZE = 0.031  # metres, side length of the printed marker on the bottle cap
HOVER_HEIGHT_METERS = 0.1  # how far above the bottle cap to hover before descending

# 10x slower than URrtde's default (0.1 m/s)
HOVER_LINEAR_SPEED = 0.01  # m/s

# ur_right's TCP -> bottle cap transform, calibrated via the 'c' key in the live view.
# Reused so the marker-based calibration isn't needed on every run -- only recalibrate if
# the bottle is re-grasped differently.
DEFAULT_TCP_RIGHT_TO_BOTTLE = np.array([[-0.99864415,  0.03057314,  0.04213239, -0.00804794],
       [ 0.04195899, -0.00629157,  0.99909952,  0.07657722],
       [ 0.03081069,  0.99951273,  0.00500022, -0.05562694],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
)

def detect_marker_in_image(image_bgr, intrinsics):
    """Detect a single ArUco marker in an already-captured frame. Returns
    (marker_pose_in_camera, detection_result) -- both None when no marker is visible."""
    result = detect_aruco_markers(image_bgr, AIRO_DEFAULT_ARUCO_DICT)
    if result is None:
        return None, None
    poses = get_poses_of_aruco_markers(result, MARKER_SIZE, intrinsics)
    if poses is None:
        return None, None
    return poses[0], result


def calibrate_bottle_grasp_transform(marker_pose_in_camera, ur_left, ur_right):
    """
    Run while the ArUco marker is visible on the bottle cap and ur_right is holding the
    bottle. Returns the fixed transform from ur_right's TCP to the bottle cap -- this stays
    valid (as long as the bottle doesn't slip in the gripper) even after the marker is
    removed and the bottle is moved to a new place by moving ur_right.
    """
    marker_pose_in_base_left = ur_left.get_tcp_pose() @ tcp_left_to_camera @ marker_pose_in_camera
    marker_pose_in_base_right = np.linalg.inv(BASE_LEFT_TO_BASE_RIGHT) @ marker_pose_in_base_left
    tcp_pose_right = ur_right.get_tcp_pose()
    return np.linalg.inv(tcp_pose_right) @ marker_pose_in_base_right


def get_bottle_cap_pose_in_base_left(tcp_right_to_bottle, ur_right):
    """
    Recompute the bottle cap's current pose in ur_left's base frame from ur_right's live TCP
    pose -- no marker needed. Valid as long as ur_right hasn't re-grasped the bottle.
    """
    tcp_pose_right = ur_right.get_tcp_pose()
    bottle_pose_in_base_right = tcp_pose_right @ tcp_right_to_bottle
    return BASE_LEFT_TO_BASE_RIGHT @ bottle_pose_in_base_right


def compute_gripper_orientation_facing_marker(marker_rotation_in_base):
    """
    Build a gripper tool orientation whose z-axis points along the marker's own -z axis, i.e.
    the gripper approaches head-on from the opposite side of whatever the marker is stuck to
    -- regardless of how the bottle is tilted. Flips the marker's y and z axes (rather than
    just z) to keep a valid right-handed rotation matrix.
    """
    marker_x_axis = marker_rotation_in_base[:, 0]
    marker_y_axis = marker_rotation_in_base[:, 1]
    marker_z_axis = marker_rotation_in_base[:, 2]
    return np.column_stack([-marker_x_axis, marker_y_axis, -marker_z_axis])


def orthonormalize_rotation(rotation_matrix):
    """
    Project a near-orthonormal 3x3 matrix back onto SO(3) via SVD. The chained pose
    multiplications/inversions above accumulate tiny floating-point drift -- not visible when
    printed, but enough for spatialmath's strict SE3 validity check to reject the matrix.
    """
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # keep a proper (right-handed) rotation, not a reflection
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def compute_hover_pose_above_bottle(bottle_cap_pose_in_base_left, hover_height=HOVER_HEIGHT_METERS):
    """Target TCP pose for ur_left: offset along the marker's own normal, gripper facing it."""
    marker_rotation = bottle_cap_pose_in_base_left[:3, :3]
    marker_position = bottle_cap_pose_in_base_left[:3, 3]
    marker_z_axis = marker_rotation[:, 2]

    hover_pose = np.eye(4)
    hover_pose[:3, :3] = orthonormalize_rotation(compute_gripper_orientation_facing_marker(marker_rotation))
    hover_pose[:3, 3] = marker_position + marker_z_axis * hover_height
    return hover_pose


def project_point_to_pixel(point_in_base, intrinsics, camera_pose_in_base):
    """Project a 3D point (base frame) into image pixel coordinates. Returns None when the
    point is behind the camera."""
    point_in_camera = np.linalg.inv(camera_pose_in_base) @ np.append(point_in_base, 1.0)
    if point_in_camera[2] <= 0:
        return None
    pixel = intrinsics @ point_in_camera[:3]
    return pixel[:2] / pixel[2]


def draw_pose_axes(image_bgr, pose_in_camera, intrinsics, axis_length):
    """Draw a pose's coordinate axes (x red, y green, z blue) on the live image."""
    rvec, _ = cv2.Rodrigues(orthonormalize_rotation(pose_in_camera[:3, :3]))
    cv2.drawFrameAxes(image_bgr, intrinsics, None, rvec, pose_in_camera[:3, 3], axis_length, 2)


class FreedriveState:
    """Tracks/toggles freedrive per robot so quitting can always restore normal control."""

    def __init__(self):
        self.active = {}

    def toggle(self, name, robot):
        if self.active.get(name, False):
            robot.rtde_control.endTeachMode()
            self.active[name] = False
        else:
            robot.rtde_control.teachMode()
            self.active[name] = True
        print(f"[freedrive] {name}: {'ON' if self.active[name] else 'off'}")

    def disable(self, name, robot):
        if self.active.get(name, False):
            robot.rtde_control.endTeachMode()
            self.active[name] = False

    def disable_all(self, robots):
        for name, robot in robots.items():
            self.disable(name, robot)


if __name__ == "__main__":
    camera = Realsense(fps=15, enable_depth=False, enable_pointcloud=False)
    freeze_auto_exposure(camera)
    intrinsics = camera.intrinsics_matrix()

    ur_left = URrtde(ip_address=LEFT_ROBOT_IP)
    ur_right = URrtde(ip_address=RIGHT_ROBOT_IP)
    robots = {"ur_right": ur_right, "ur_left": ur_left}
    freedrive = FreedriveState()

    tcp_right_to_bottle = DEFAULT_TCP_RIGHT_TO_BOTTLE
    transform_source = "default"

    window_name = "calibrate + hover (c=calibrate, h=hover, f/g=freedrive right/left, q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(__doc__)

    while True:
        image_rgb = camera.get_rgb_image_as_int()
        image_bgr = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        vis = image_bgr.copy()

        camera_pose_in_base = ur_left.get_tcp_pose() @ tcp_left_to_camera

        # --- live marker detection ---
        marker_pose_in_camera, marker_result = detect_marker_in_image(image_bgr, intrinsics)
        if marker_result is not None:
            vis = visualize_aruco_detections(vis, marker_result)
            draw_pose_axes(vis, marker_pose_in_camera, intrinsics, MARKER_SIZE * 0.75)
            x, y, z = marker_pose_in_camera[:3, 3]
            cv2.putText(vis, f"marker at [{x:+.3f} {y:+.3f} {z:+.3f}] m (camera frame)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- live bottle center (same yellow point as open_bottle_demo's rerun view) ---
        cap_pose = get_bottle_cap_pose_in_base_left(tcp_right_to_bottle, ur_right)
        cap_pose_in_camera = np.linalg.inv(camera_pose_in_base) @ cap_pose
        center_pixel = project_point_to_pixel(cap_pose[:3, 3], intrinsics, camera_pose_in_base)
        if center_pixel is not None:
            center_px = tuple(int(round(v)) for v in center_pixel)
            cv2.drawMarker(vis, center_px, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis, "bottle center", (center_px[0] + 14, center_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            draw_pose_axes(vis, cap_pose_in_camera, intrinsics, 0.03)

        status = f"transform: {transform_source}   freedrive: " + ", ".join(
            f"{name}={'ON' if freedrive.active.get(name) else 'off'}" for name in robots
        )
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord("f"):
            freedrive.toggle("ur_right", ur_right)
        elif key == ord("g"):
            freedrive.toggle("ur_left", ur_left)
        elif key == ord("c"):
            if marker_pose_in_camera is None:
                print("[calibrate] no marker visible -- can't calibrate")
                continue
            tcp_right_to_bottle = calibrate_bottle_grasp_transform(marker_pose_in_camera, ur_left, ur_right)
            transform_source = "calibrated (this session)"
            print("[calibrate] done -- you can remove the marker now. The projected bottle")
            print("[calibrate] center should stay glued to the cap when you move ur_right (press f).")
            print("[calibrate] Copy this into DEFAULT_TCP_RIGHT_TO_BOTTLE to keep it:")
            print(repr(tcp_right_to_bottle))
        elif key == ord("h"):
            freedrive.disable_all(robots)
            hover_pose = compute_hover_pose_above_bottle(get_bottle_cap_pose_in_base_left(tcp_right_to_bottle, ur_right))
            print(f"[hover] moving ur_left above the bottle cap at {np.round(hover_pose[:3, 3], 4)}")
            ur_left.move_linear_to_tcp_pose(hover_pose, linear_speed=HOVER_LINEAR_SPEED).wait()
            print("[hover] done")

    freedrive.disable_all(robots)
    cv2.destroyAllWindows()
