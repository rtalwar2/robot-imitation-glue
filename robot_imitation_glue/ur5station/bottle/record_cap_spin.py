"""Film the cap head-on while you spin the bottle in ur_right's fingers, to find the cap's rotation axis.

ur_right goes to a split pose holding the bottle. ur_left then parks so that the WRIST CAMERA (not the
gripper) looks straight down the modelled cap normal, with the modelled cap centre on the optical axis,
CAMERA_DISTANCE_M away. ur_left stays still; you turn the bottle by hand in the fingers. Any point on the
cap then traces a circle, and because the view is head-on that circle's image centre is where the
rotation axis -- the true cap centre -- projects. click_cap_spin.py turns clicks on the video into that
centre and into DEFAULT_TCP_RIGHT_TO_BOTTLE's in-plane x/z.

    python -m robot_imitation_glue.ur5station.bottle.record_cap_spin --pose-index 0

Live window: r starts/stops recording, q (or Esc) quits. Each recording goes to outputs/cap_spin/<time>/
as video.mp4 + meta.json (intrinsics, both arms' poses, the cap model, per-frame timestamps).
By default the bottle must already be in ur_right's fingers; --regrasp opens them at the neutral pose and
waits for you to hand the bottle over.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_imitation_glue.ur5station.bottle.check_hover_center_bias import (  # noqa: E402
    DEFAULT_TCP_RIGHT_TO_BOTTLE,
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_OPEN_WIDTH_M,
    GRIPPER_PORT_RIGHT,
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    POSE_BLACKLIST,
    RANDOM_SEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    SPLIT_SEED_OFFSETS,
    BottleStation,
    _move_gripper_to,
    _pt,
    compute_hover_pose_above_bottle,
    generate_reachable_bottle_poses,
    make_gripper,
    project_point_to_pixel,
    rotation_about_axis,
    tcp_left_to_camera,
    to_bgr,
)
from calibrate_and_hover_bottle import BASE_LEFT_TO_BASE_RIGHT  # noqa: E402  (on sys.path via the import above)

CAMERA_DISTANCE_M = 0.25  # camera to cap centre, along the cap normal -- about the usual hover's view depth
ROLL_SEARCH_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90)  # camera roll tries if IK refuses
WINDOW = "cap spin (r = record on/off, q = quit)"
OUT_ROOT = Path("outputs/cap_spin")
FPS = 30


def head_on_tcp_pose(cap_pose: np.ndarray, roll_deg: float) -> np.ndarray:
    """ur_left TCP pose that puts the camera on the cap normal, looking at the cap centre.

    The camera's in-image "up" is taken from the usual hover so the view looks familiar, then rolled about
    the optical axis by roll_deg (only to find a reachable arm configuration).
    """
    normal = cap_pose[:3, 2]  # points out of the cap, toward the camera
    z = -normal
    usual_camera = compute_hover_pose_above_bottle(cap_pose) @ tcp_left_to_camera
    x = usual_camera[:3, 0] - z * (usual_camera[:3, 0] @ z)
    x = rotation_about_axis(z, np.radians(roll_deg)) @ (x / np.linalg.norm(x))
    camera = np.eye(4)
    camera[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    camera[:3, 3] = cap_pose[:3, 3] + CAMERA_DISTANCE_M * normal
    return camera @ np.linalg.inv(tcp_left_to_camera)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=list(SPLIT_SEED_OFFSETS))
    parser.add_argument("--pose-index", type=int, default=0, help="which pose of the split's seeded stream")
    parser.add_argument("--regrasp", action="store_true", help="open at neutral and wait for the bottle to be handed over")
    args = parser.parse_args()

    env = BottleStation(make_gripper(GRIPPER_PORT_RIGHT), with_instrumentation=False,
                        with_spectogram_model=False, with_spectogram=False, use_internal_ft=True)
    intrinsics = env.get_camera_intrinsics()
    rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS[args.split])
    poses = generate_reachable_bottle_poses(args.pose_index + 1, env.robot, env.robot_right, rng,
                                            blacklist=POSE_BLACKLIST[args.split])
    tcp_right_pose, _ = poses[args.pose_index]
    writer = None
    try:
        env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
        if args.regrasp:
            _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
            input("Hold the bottle between the open fingers, then press Enter...")
        env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
        if args.regrasp:
            _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)

        cap_pose = env.get_bottle_cap_pose()
        for roll in ROLL_SEARCH_DEG:
            target = head_on_tcp_pose(cap_pose, roll)
            if env.robot.is_tcp_pose_reachable(target):
                break
        else:
            raise RuntimeError("no camera roll in ROLL_SEARCH_DEG gives a reachable head-on pose for ur_left")
        print(f"[move] ur_left to the head-on view (camera roll {roll:+d} deg, {CAMERA_DISTANCE_M * 100:.0f} cm from the cap)")
        env.move_robot_to_tcp_pose(target, joint_speed=LEFT_TRANSIT_JOINT_SPEED)

        tcp_left = env.get_robot_pose_se3()
        camera_pose = tcp_left @ tcp_left_to_camera
        center_px = project_point_to_pixel(cap_pose[:3, 3], intrinsics, camera_pose)
        meta = {"intrinsics": np.asarray(intrinsics).tolist(), "tcp_left_to_camera": tcp_left_to_camera.tolist(),
                "base_left_to_base_right": BASE_LEFT_TO_BASE_RIGHT.tolist(),
                "tcp_right_to_bottle": DEFAULT_TCP_RIGHT_TO_BOTTLE.tolist(),
                "tcp_right_base_right": env.robot_right.get_tcp_pose().tolist(), "tcp_left_base_left": tcp_left.tolist(),
                "cap_pose_base_left": cap_pose.tolist(), "modelled_center_px": [float(v) for v in center_px],
                "split": args.split, "pose_index": args.pose_index, "camera_roll_deg": roll}
        print(f"[view] the modelled cap centre projects to {_pt(center_px)} -- r starts recording, q quits")

        directory, stamps = None, []
        while True:
            frame = to_bgr(env.get_observations()["wrist_image_original"])
            if writer is not None:
                writer.write(frame)
                stamps.append(time.time())
            shown = frame.copy()
            cv2.drawMarker(shown, _pt(center_px), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
            status = f"REC {len(stamps)} frames" if writer is not None else "not recording"
            cv2.putText(shown, f"{status} -- yellow = modelled cap centre (x = {DEFAULT_TCP_RIGHT_TO_BOTTLE[0, 3] * 1000:.1f} mm)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if writer is not None else (255, 255, 255), 2)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and writer is None:
                directory = OUT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}"
                directory.mkdir(parents=True)
                writer = cv2.VideoWriter(str(directory / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                         (frame.shape[1], frame.shape[0]))
                stamps = []
                print(f"[rec] recording to {directory} -- spin the bottle")
            elif (key == ord("r") or key in (27, ord("q"))) and writer is not None:
                writer.release()
                writer = None
                (directory / "meta.json").write_text(json.dumps({**meta, "frame_times": stamps}, indent=1))
                print(f"[rec] saved {len(stamps)} frames to {directory}")
            if key in (27, ord("q")):
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        try:
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        except Exception as exc:  # cleanup must not hide the failure that triggered it
            print(f"[cleanup] could not send ur_left home: {exc}")
        env.close()


if __name__ == "__main__":
    main()
