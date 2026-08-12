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

import asyncio
import json
import os
import struct
import threading
import time

import cv2
import numpy as np
import rerun as rr
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from bleak import BleakClient, BleakScanner
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

# same BLE device/characteristic as instrumentation_ble_plot3.py
SENSOR_DEVICE_NAME = "CaptainHook"
SENSOR_CHARACTERISTIC_UUID = "1A3AC130-31EE-758A-BC50-54A61958EF81"
SENSOR_LOG_DIR = "/home/rtalwar/robot-imitation-glue/bottle_experiment/sensor_logs"
# how far back (seconds) to average when reading the "current" sensor state right after a
# move -- rejects transient noise from the move itself/gripper vibration, not a covered
# vs. uncovered threshold (that needs to come from looking at real recorded data first).
SENSOR_DEBOUNCE_WINDOW_S = 0.3


class SensorLogger:
    """Background BLE reader for the 3-channel cap sensor, logging a continuous time
    series plus timestamped event markers (leg boundaries, etc.) for later analysis.

    Connects on a background thread (bleak needs its own asyncio loop) exactly like
    instrumentation_ble_plot3.py; call start() and wait for it to report connected before
    moving the robot, so no samples are missed at the beginning of the motion.
    """

    def __init__(self, device_name=SENSOR_DEVICE_NAME, characteristic_uuid=SENSOR_CHARACTERISTIC_UUID):
        self.device_name = device_name
        self.characteristic_uuid = characteristic_uuid
        self.samples = []  # list of [t, v0, v1, v2]
        self.events = []  # list of {"time": t, "name": ..., **extra}
        self._lock = threading.Lock()
        self._start_time = None
        self._connected = threading.Event()
        self._failed = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)

    def start(self, timeout=15.0):
        self._thread.start()
        if not self._connected.wait(timeout=timeout) or self._failed.is_set():
            raise RuntimeError(f"Could not connect to sensor device '{self.device_name}' within {timeout}s")
        print(f"[sensors] connected to '{self.device_name}', logging started")

    async def _run(self):
        try:
            device = await BleakScanner.find_device_by_name(self.device_name, timeout=10.0)
            if device is None:
                print(f"[sensors] device '{self.device_name}' not found")
                self._failed.set()
                self._connected.set()
                return
            async with BleakClient(device) as client:
                self._start_time = time.time()

                def handler(_sender, data: bytearray):
                    v0, v1, v2 = struct.unpack("<3f", data)
                    t = time.time() - self._start_time
                    with self._lock:
                        self.samples.append([t, v0, v1, v2])

                await client.start_notify(self.characteristic_uuid, handler)
                self._connected.set()
                while True:
                    await asyncio.sleep(0.1)
        except Exception as exc:  # keep the main script alive even if BLE drops/fails
            print(f"[sensors] BLE error: {exc}")
            self._failed.set()
            self._connected.set()

    def log_event(self, name, **extra):
        """Timestamp a motion event (e.g. a leg's move completing) against the sensor
        stream's own clock, so events and samples share one timeline."""
        t = (time.time() - self._start_time) if self._start_time is not None else None
        self.events.append({"time": t, "name": name, **extra})
        print(f"[sensors] event '{name}' @ t={t:.2f}s reading={self.current_reading()}" if t is not None else f"[sensors] event '{name}' (no samples yet)")

    def current_reading(self, window_s=SENSOR_DEBOUNCE_WINDOW_S):
        """Debounced [v0, v1, v2]: mean of samples within the last `window_s` seconds, or
        None if nothing has arrived yet."""
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return None
        now = samples[-1][0]
        recent = [s[1:] for s in samples if now - s[0] <= window_s]
        return list(np.mean(recent if recent else [samples[-1][1:]], axis=0))

    def save(self, path):
        with self._lock:
            samples = list(self.samples)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"samples": samples, "events": self.events}, f)
        print(f"[sensors] saved {len(samples)} samples, {len(self.events)} events to {path}")

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

# --- sensor-verified retry -------------------------------------------------------------
# Each channel's own covered/uncovered voltage separation, derived from the largest gap in
# 3 recorded runs (sensor_logs/run_0000-0002.json): S0 in [2.25-2.86] covered / [3.24] open,
# S1 in [2.69-3.12] covered / [3.24-3.28] open, S2 in [2.24-2.78] covered / [3.22-3.23] open.
# Re-derive (see the analysis in this file's git history / chat) if the sensor mounting,
# cap, or LEGS geometry changes -- these numbers are specific to this physical setup.
PER_CHANNEL_THRESHOLDS = [3.05, 3.18, 3.00]  # S0, S1, S2, in volts

# Which sensor channels (indices into PER_CHANNEL_THRESHOLDS) must be uncovered by the end
# of each named event, per the same 3 runs. Only leg_3_end and leg_5_end are real
# milestones: S0 and S1 both pop open by leg_3_end, S2 (the last tab) by leg_5_end.
# push_end/leg_2/leg_4/leg_6 show no reliable NEW transition at these thresholds and are
# deliberately not gated (leg_6 in particular showed zero sensor change in any run).
SENSOR_CHECKPOINTS = {
    "leg_3_end": [0, 1],
    "leg_5_end": [0, 1, 2],
}

DEPTH_NUDGE_M = 0.003  # metres to press deeper per retry attempt, along -cap_normal
MAX_RETRIES_PER_CHECKPOINT = 3


def verify_checkpoint_and_retry(ur_left, sensor_logger, event_name, pose, cap_normal, move_speed=MOVE_SPEED):
    """After reaching `pose` for `event_name`, check whether the sensor channels required
    to be uncovered by this checkpoint (SENSOR_CHECKPOINTS) actually are. If not, nudge the
    gripper DEPTH_NUDGE_M deeper along -cap_normal from its CURRENT position (same lateral
    target, not a restart from the grip point) and retry in place, up to
    MAX_RETRIES_PER_CHECKPOINT times -- the idea being the gripper likely slipped and
    didn't press the tab open.

    Returns (pose_reached, retries_used). `pose_reached` is possibly deeper than the input
    `pose` -- the caller should use it as the base for whatever move comes right after
    this checkpoint. `retries_used * DEPTH_NUDGE_M` is the total depth correction applied
    here; the caller should also carry that same correction into every LATER leg's
    precomputed target (those were computed before any retry and don't know about it), or
    the next leg's move will spring back up to the original, too-shallow height.

    Events with no entry in SENSOR_CHECKPOINTS aren't gated -- returns (pose, 0) immediately.
    """
    required_channels = SENSOR_CHECKPOINTS.get(event_name)
    if not required_channels:
        return pose, 0

    for attempt in range(MAX_RETRIES_PER_CHECKPOINT + 1):
        reading = sensor_logger.current_reading()
        sensor_logger.log_event(f"{event_name}_verify", attempt=attempt, reading=reading)
        uncovered = [reading[ch] >= PER_CHANNEL_THRESHOLDS[ch] for ch in required_channels]
        if all(uncovered):
            if attempt > 0:
                print(f"[verify] {event_name}: OK after {attempt} retry(ies), reading={np.round(reading, 2)}")
            return pose, attempt

        missing = [f"S{ch}={reading[ch]:.2f}V(<{PER_CHANNEL_THRESHOLDS[ch]}V)"
                   for ch, ok in zip(required_channels, uncovered) if not ok]
        if attempt == MAX_RETRIES_PER_CHECKPOINT:
            print(f"[verify] {event_name}: WARNING -- still not uncovered after {MAX_RETRIES_PER_CHECKPOINT} "
                  f"retries ({', '.join(missing)}) -- continuing anyway")
            return pose, attempt

        print(f"[verify] {event_name}: not yet uncovered ({', '.join(missing)}) -- pressing "
              f"{DEPTH_NUDGE_M * 100:.1f}cm deeper and retrying (attempt {attempt + 1}/{MAX_RETRIES_PER_CHECKPOINT})")
        pose = pose.copy()
        pose[:3, 3] = pose[:3, 3] - DEPTH_NUDGE_M * cap_normal
        ur_left.move_linear_to_tcp_pose(pose, linear_speed=move_speed).wait()
    return pose, MAX_RETRIES_PER_CHECKPOINT


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

    # start the sensor BLE connection first -- it's the slowest thing to set up (scan +
    # connect can take several seconds), so let it run in the background while the camera
    # and robots initialize below, rather than delaying the start of logging until right
    # before the motion.
    sensor_logger = SensorLogger()
    sensor_logger.start()

    # 720p to match touch_point_detector's radius prior (calibrated on 720p frames);
    # intrinsics_matrix() returns the intrinsics for this stream resolution.
    camera = Realsense(resolution=Realsense.RESOLUTION_720, fps=15, enable_depth=False, enable_pointcloud=False)
    freeze_auto_exposure(camera)
    intrinsics = camera.intrinsics_matrix()
    ur_left = URrtde(ip_address=LEFT_ROBOT_IP)
    ur_left.move_to_joint_configuration([ 0.06967844 ,-1.40953115 ,-1.61241627, -1.67278638 , 1.54791272 , 3.03650355],joint_speed=0.02).wait()
    # time.sleep(2)
    ur_right = URrtde(ip_address=RIGHT_ROBOT_IP)
    ur_right.rtde_control.teachMode()
    input("Move the right arm where ever you want")
    ur_right.rtde_control.endTeachMode()

    cap_pose = get_bottle_cap_pose_in_base_left(DEFAULT_TCP_RIGHT_TO_BOTTLE, ur_right)
    hover_pose = compute_hover_pose_above_bottle(cap_pose, 0.1)
    print(f"hover pose: {hover_pose}")
    print(f"Moving ur_left above the bottle cap at {hover_pose[:3, 3]}")

    ur_left.move_linear_to_tcp_pose(hover_pose, linear_speed=0.01).wait()
    sensor_logger.log_event("hover_reached")
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
    ur_left.move_linear_to_tcp_pose(yaw_pose, linear_speed=MOVE_SPEED).wait()

    # descend to the grip point (TANGENTIAL_OFFSET left of the touch point, APPROACH_OFFSET
    # along the cap normal), keeping the yawed orientation
    approach_pose = np.eye(4)
    approach_pose[:3, :3] = yawed_rotation
    approach_pose[:3, 3] = approach_position
    print(f"[move] descending to {np.round(approach_position, 4)}")
    ur_left.move_linear_to_tcp_pose(approach_pose, linear_speed=MOVE_SPEED).wait()
    sensor_logger.log_event("grip_point_reached", target=approach_position.tolist())

    input("Press Enter to start the opening motion (Ctrl+C to abort)...")
    # a single moveL is already a straight Cartesian line -- no waypoints needed
    push_pose = approach_pose.copy()
    push_pose[:3, 3] = push_target
    print(f"[push] straight line to {np.round(push_target, 4)}")
    ur_left.move_linear_to_tcp_pose(push_pose, linear_speed=MOVE_SPEED).wait()
    sensor_logger.log_event("push_end", target=push_target.tolist())
    push_pose, retries = verify_checkpoint_and_retry(ur_left, sensor_logger, "push_end", push_pose, cap_normal)
    depth_correction = retries * DEPTH_NUDGE_M  # cumulative depth nudge, carried into every later leg's target

    # chained legs, each at its angle relative to the previous leg's direction. Every
    # target is offset by `depth_correction` along -cap_normal, so a retry's deeper press
    # persists through the rest of the motion instead of springing back up on the next leg.
    leg_pose = push_pose.copy()
    for i, ((angle_deg, offset), leg_target) in enumerate(zip(LEGS, leg_targets)):
        event_name = f"leg_{i + 2}_end"
        leg_pose = leg_pose.copy()
        leg_pose[:3, 3] = leg_target - depth_correction * cap_normal
        print(f"[push] leg {i + 2} ({angle_deg:.0f} deg from previous direction, {offset * 100:.0f}cm) to {np.round(leg_pose[:3, 3], 4)}")
        ur_left.move_linear_to_tcp_pose(leg_pose, linear_speed=MOVE_SPEED).wait()
        sensor_logger.log_event(event_name, leg_index=i + 2, angle_deg=angle_deg, offset_m=offset, target=leg_pose[:3, 3].tolist())
        leg_pose, retries = verify_checkpoint_and_retry(ur_left, sensor_logger, event_name, leg_pose, cap_normal)
        depth_correction += retries * DEPTH_NUDGE_M
    ur_left.move_to_joint_configuration([ 0.06967844 ,-1.40953115 ,-1.61241627, -1.67278638 , 1.54791272 , 3.03650355])
    sensor_logger.log_event("retreat_home")

    run_index = len([n for n in os.listdir(SENSOR_LOG_DIR) if n.endswith(".json")]) if os.path.isdir(SENSOR_LOG_DIR) else 0
    sensor_logger.save(os.path.join(SENSOR_LOG_DIR, f"run_{run_index:04d}.json"))

    print("[done] opening motion finished")
