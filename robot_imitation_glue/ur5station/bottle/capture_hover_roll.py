"""Live check of the cap model: the modelled cap centre drawn on the wrist camera while you move ur_left by hand.

ur_right goes to a split pose holding the bottle, ur_left goes to the hover pose above the MODELLED cap,
then ur_left is put in freedrive (teach mode). Every frame the cap centre is recomputed from ur_right's
live TCP (DEFAULT_TCP_RIGHT_TO_BOTTLE, BASE_LEFT_TO_BASE_RIGHT) and projected through ur_left's live TCP
(tcp_left_to_camera), so the yellow cross must stay on the cap's sticker wherever you push the camera.
If it slides off as the view changes, one of those transforms is wrong; if it is off by a constant
amount at the hover, the hover itself is off-centre.

    python -m robot_imitation_glue.ur5station.bottle.capture_hover_roll --pose-index 0
    python -m robot_imitation_glue.ur5station.bottle.capture_hover_roll --pose-index 2 --regrasp

By default the bottle must already be in ur_right's fingers and stays there; --regrasp opens the fingers
at the neutral pose and waits for you to hand the bottle over. In the live window: x toggles the cap
model between the measured transform and the same one with its x translation set to 0 (to see which
sits on the sticker), s saves a frame to outputs/hover_live/, q (or Esc) quits -- freedrive is always
switched off on the way out.
"""

import argparse
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
    tcp_left_to_camera,
    to_bgr,
)
from calibrate_and_hover_bottle import get_bottle_cap_pose_in_base_left  # noqa: E402  (on sys.path via the import above)

WINDOW = "cap model live (s = save, q = quit)"
OUT_DIR = Path("outputs/hover_live")
NORMAL_TICK_M = 0.02  # length of the drawn cap-normal tick, to show the modelled cap orientation

# The x toggle: the measured grasp transform, and the same with the cap centre on the flange axis (x = 0).
X_ZERO = DEFAULT_TCP_RIGHT_TO_BOTTLE.copy()
X_ZERO[0, 3] = 0.0
CAP_MODELS = [(f"measured x = {DEFAULT_TCP_RIGHT_TO_BOTTLE[0, 3] * 1000:.1f} mm", DEFAULT_TCP_RIGHT_TO_BOTTLE),
              ("x = 0 mm", X_ZERO)]


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
    control = env.robot.rtde_control
    try:
        env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
        if args.regrasp:
            _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
            input("Hold the bottle between the open fingers, then press Enter...")
        env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
        if args.regrasp:
            _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)
        env.move_robot_to_tcp_pose(compute_hover_pose_above_bottle(env.get_bottle_cap_pose()),
                                   joint_speed=LEFT_TRANSIT_JOINT_SPEED)

        control.teachMode()
        print("[freedrive] ur_left is in teach mode -- move the camera by hand; x toggles the model, s saves, q quits")
        model_index = 0
        while True:
            image = to_bgr(env.get_observations()["wrist_image_original"])
            model_name, grasp_transform = CAP_MODELS[model_index]
            cap_pose = get_bottle_cap_pose_in_base_left(grasp_transform, env.robot_right)
            tcp_left = env.get_robot_pose_se3()
            camera_pose = tcp_left @ tcp_left_to_camera
            center = project_point_to_pixel(cap_pose[:3, 3], intrinsics, camera_pose)
            tick = project_point_to_pixel(cap_pose[:3, 3] + NORMAL_TICK_M * cap_pose[:3, 2], intrinsics, camera_pose)
            height_mm = float(cap_pose[:3, 2] @ (tcp_left[:3, 3] - cap_pose[:3, 3])) * 1000
            if center is not None:
                cv2.drawMarker(image, _pt(center), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
                if tick is not None:
                    cv2.line(image, _pt(center), _pt(tick), (0, 255, 255), 1)
            cv2.putText(image, f"cap model: {model_name} (x toggles)   ur_left TCP {height_mm:.0f}mm above the cap plane",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("x"):
                model_index = 1 - model_index
                print(f"[model] {CAP_MODELS[model_index][0]}")
            if key == ord("s"):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                path = OUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{'measured' if model_index == 0 else 'x0'}.jpg"
                cv2.imwrite(str(path), image)
                print(f"[saved] {path}")
    finally:
        try:
            control.endTeachMode()
        except Exception as exc:  # never leave the arm in freedrive silently
            print(f"[freedrive] could not end teach mode -- end it on the pendant: {exc}")
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
