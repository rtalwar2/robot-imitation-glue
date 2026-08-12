"""
Interactive annotation + detection tool for the bottle-cap lid touch point (where the
gripper should press to start the circular opening motion).

Click on the displayed image to mark the ground-truth touch point. The tool runs the
current candidate detection algorithm (see `detect_touch_point`) on the same frame and
overlays both points (green = your click, red = detected) with the pixel error, so you
can see immediately how well it's doing.

Each save also writes the raw frame + click/detected coordinates to disk, so the
detection algorithm can be iterated on offline (against saved frames) without needing the
camera/robot every time.

Keys:
  c - capture a fresh frame from the camera (clears the current click)
  s - save the current frame + click + detection to disk
  q - quit

Usage:
    python lid_touch_point_annotator.py                    # live camera
    python lid_touch_point_annotator.py --replay path.jpg   # offline, re-annotate a saved frame
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from calibrate_and_hover_bottle import DEFAULT_TCP_RIGHT_TO_BOTTLE, compute_hover_pose_above_bottle, get_bottle_cap_pose_in_base_left
from camera_utils import freeze_auto_exposure
from touch_point_detector import detect_touch_point

OUTPUT_DIR = "/home/rtalwar/robot-imitation-glue/lid_touch_point_debug"

# 10x slower than URrtde's default (0.1 m/s) -- gives more time to click/annotate mid-move
# and less risk while iterating on hover poses.
HOVER_LINEAR_SPEED = 0.01  # m/s


class ClickState:
    def __init__(self) -> None:
        self.point: tuple[int, int] | None = None

    def on_mouse_event(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)


def draw_overlay(bgr_image: np.ndarray, click: tuple[int, int] | None, detected: tuple[int, int] | None) -> np.ndarray:
    overlay = bgr_image.copy()
    if click is not None:
        cv2.drawMarker(overlay, click, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(overlay, "click", (click[0] + 12, click[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if detected is not None:
        cv2.drawMarker(overlay, detected, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(overlay, "detected", (detected[0] + 12, detected[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if click is not None and detected is not None:
        error = float(np.linalg.norm(np.array(click) - np.array(detected)))
        cv2.putText(overlay, f"error: {error:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", default=None, help="re-annotate a saved *_raw.jpg frame instead of the live camera")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_samples = len([n for n in os.listdir(OUTPUT_DIR) if n.endswith("_raw.jpg")])

    camera = None if args.replay else Realsense(resolution=Realsense.RESOLUTION_720, fps=15, enable_depth=False, enable_pointcloud=False)
    if camera is not None:
        freeze_auto_exposure(camera)
    ur_left = URrtde(ip_address="10.42.0.163")  # arm with gripper, wrist camera mounted on it
    ur_right = URrtde(ip_address="10.42.0.162")  # arm holding the bottle

    window_name = "Lid touch-point annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    click_state = ClickState()
    cv2.setMouseCallback(window_name, click_state.on_mouse_event)

    def capture_frame() -> np.ndarray:
        if args.replay:
            return cv2.imread(args.replay)
        image = camera.get_rgb_image_as_int()
        return ImageConverter.from_numpy_int_format(image).image_in_opencv_format

    frame_bgr = capture_frame()
    save_count = existing_samples

    print("Keys: click=mark ground truth, s=save sample, c=recapture frame, n/space=next height, q=quit")
    print(f"Saving samples to {OUTPUT_DIR}")

    quit_requested = False
    while not quit_requested:
        ur_right.rtde_control.teachMode()
        input("Move/reposition the bottle (via ur_right) if you like, then press Enter to move the gripper above it...")
        ur_right.rtde_control.endTeachMode()
        bottle_cap_pose = get_bottle_cap_pose_in_base_left(DEFAULT_TCP_RIGHT_TO_BOTTLE, ur_right)
        for height in [0.05, 0.1, 0.15]:
            hover_pose = compute_hover_pose_above_bottle(bottle_cap_pose, height)
            print(f"hover pose: {hover_pose}")
            print(f"Moving ur_left above the bottle cap at {hover_pose[:3, 3]}")

            ur_left.move_linear_to_tcp_pose(hover_pose, linear_speed=HOVER_LINEAR_SPEED).wait()
            frame_bgr = capture_frame()
            click_state.point = None
            print(f"height={height:.2f}m -- annotate/save, then press n/space for the next height")

            move_to_next_height = False
            while not move_to_next_height:
                detected = detect_touch_point(frame_bgr, height)
                overlay = draw_overlay(frame_bgr, click_state.point, detected)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    quit_requested = True
                    move_to_next_height = True
                elif key == ord("c") and camera is not None:
                    frame_bgr = capture_frame()
                    click_state.point = None
                elif key == ord("s"):
                    prefix = os.path.join(OUTPUT_DIR, f"sample_{save_count:04d}")
                    cv2.imwrite(f"{prefix}_raw.jpg", frame_bgr)
                    cv2.imwrite(f"{prefix}_overlay.jpg", overlay)
                    with open(f"{prefix}.json", "w") as f:
                        json.dump({"height": height, "click_xy": click_state.point, "detected_xy": detected}, f)
                    print(f"saved {prefix}  height={height}  click={click_state.point}  detected={detected}")
                    save_count += 1
                elif key in (ord("n"), ord(" ")):
                    move_to_next_height = True

            if quit_requested:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
