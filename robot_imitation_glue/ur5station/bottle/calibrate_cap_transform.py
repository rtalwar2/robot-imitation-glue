"""Re-measure DEFAULT_TCP_RIGHT_TO_BOTTLE: the cap centre triangulated by the wrist camera, checked by touch.

Model: the cap plane is parallel to ur_right's TCP x-z plane, so the cap normal is ur_right's TCP +y
and only the cap centre's position in ur_right's TCP frame is measured.

Per pose:
  1. ur_right holds the bottle at a split pose, ur_left goes to the usual hover pose;
  2. the wrist rolls to -90 / 0 / +90 deg about the tool axis. The camera sits 69 mm off that axis,
     so the three views are ~100 mm apart, and the rays to the sticker cross at the cap centre in 3D.
     This gives the in-plane x/z; its height is the weak axis (along the view direction);
  3. ur_left descends along its tool axis until the controller detects contact with the cap
     (moveUntilContact). Its fingertips sit FINGERTIPS_SHORT_OF_TCP_M short of its TCP, so the contact
     gives the cap-top height (y) directly -- repeatable to ~1 mm where the camera height scattered 6.

The rim circle is detected and drawn but not used: its triangulated height jumped by up to 30 mm
between poses. Several poses are measured so the per-pose transforms can be compared: if they scatter
by more than a couple of mm, a single constant cannot absorb it (suspect BASE_LEFT_TO_BASE_RIGHT).

Fully automatic by default, no Enter gates: the bottle must already be in ur_right's fingers and stays
there for the whole run (one grasp for all poses); keep a hand on the e-stop. --regrasp releases it at
every pose instead and waits for you to hand it back, as the collector does -- that is the run that
tests re-grasp repeatability.

    python -m robot_imitation_glue.ur5station.bottle.calibrate_cap_transform --split train --n-poses 8
    python -m robot_imitation_glue.ur5station.bottle.calibrate_cap_transform --regrasp   # validation

Uses tcp_left_to_camera from calibrate_and_hover_bottle, so the camera calibration must be current.
Nothing is written to the code: the result is printed for pasting and saved to outputs/cap_transform/.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_imitation_glue.ur5station.bottle.check_hover_center_bias import (  # noqa: E402
    GRAB_SETTLE_S,
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_OPEN_WIDTH_M,
    GRIPPER_PORT_RIGHT,
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    MOVE_SPEED,
    N_GRABS,
    POSE_BLACKLIST,
    RANDOM_SEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    SPLIT_SEED_OFFSETS,
    STICKER_SEARCH_RADIUS_PX,
    BottleStation,
    LiveView,
    StickerModel,
    _draw_marks,
    _grab_frame,
    _move_gripper_to,
    _pt,
    compute_hover_pose_above_bottle,
    detect_rim_centre,
    generate_reachable_bottle_poses,
    lock_sticker_for_pose,
    make_gripper,
    project_point_to_pixel,
    rim_detector,
    rolled_pose,
    tcp_left_to_camera,
)
from calibrate_and_hover_bottle import (  # noqa: E402  (bottle_experiment/ is on sys.path by now)
    BASE_LEFT_TO_BASE_RIGHT,
    DEFAULT_TCP_RIGHT_TO_BOTTLE,
    HOVER_HEIGHT_METERS,
)
from motion_constants import YAW_SPEED  # noqa: E402  (rolls are pure rotations: MOVE_SPEED would time out)

ROLLS_DEG = (0, -90, 90)
LANDMARKS = ("sticker",)
# Measured 2026-10-08 over 8 poses (camera height vs touch): ur_left's TCP lies this far past its fingertips.
FINGERTIPS_SHORT_OF_TCP_M = 0.0211

# The touch uses ur_rtde's moveUntilContact: the CONTROLLER detects contact and retracts to the exact
# point of first contact. A client-side speedL loop polling the F/T sensor was tried first and failed
# on the rig -- with the camera threads running, Python missed the 2 ms periods, the pendant raised
# C271A1 "runtime is too much behind" and the RTDE control script died mid-descent.
TOUCH_SPEED = 0.003  # m/s
# The descent starts with the TCP this far above the MODELLED cap plane. Contact is then expected
# ~20-25 mm BELOW that plane, because the controller's TCP (185 mm from the flange) lies past the
# fingertips; a contact much deeper than MAX_BELOW_MODEL_M hit something other than the cap top.
SEARCH_START_ABOVE_M = 0.010
MAX_BELOW_MODEL_M = 0.040

RETREAT_M = 0.08

OUT_DIR = Path("outputs/cap_transform")


def pixel_ray(pixel, intrinsics, camera_pose):
    """(origin, unit direction) of the camera ray through a pixel, in ur_left's base frame."""
    direction = camera_pose[:3, :3] @ (np.linalg.inv(intrinsics) @ np.array([pixel[0], pixel[1], 1.0]))
    return camera_pose[:3, 3], direction / np.linalg.norm(direction)


def triangulate(rays):
    """Least-squares point closest to all rays, and each ray's miss distance from it."""
    a, b = np.zeros((3, 3)), np.zeros(3)
    for origin, direction in rays:
        perp = np.eye(3) - np.outer(direction, direction)
        a += perp
        b += perp @ origin
    point = np.linalg.solve(a, b)
    misses = [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (point - o))) for o, d in rays]
    return point, misses


def cap_rotation(old: np.ndarray) -> np.ndarray:
    """Cap frame with z exactly along ur_right's TCP +y; x keeps the old in-plane direction."""
    z = np.array([0.0, 1.0, 0.0])
    x = old[:3, 0] - z * old[1, 0]
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def locate_landmarks(env, model, anchor):
    """Sticker and rim centre in pixels, each averaged over N_GRABS frames at a stationary pose."""
    radius = rim_detector._predict_rim_radius(HOVER_HEIGHT_METERS)
    sticker, rim, image = [], [], None
    for _ in range(N_GRABS):
        image = _grab_frame(env)
        hit = model.find(image, anchor, STICKER_SEARCH_RADIUS_PX)
        if hit and hit.get("px"):
            sticker.append(hit["px"])
        circle = detect_rim_centre(image, anchor, radius, HOVER_HEIGHT_METERS)
        if circle and "rejected" not in circle:
            rim.append((circle["px"], circle["radius_px"]))
    found = {}
    if sticker:
        found["sticker"] = np.mean(sticker, axis=0)
    if rim:
        found["rim"] = np.mean([p for p, _ in rim], axis=0)
        found["rim_radius_px"] = float(np.mean([r for _, r in rim]))
    return found, image


def view_from_rolls(env, live, model, intrinsics, hover_pose, cap_center_model, tag):
    """Landmark rays from each roll angle; returns them per landmark plus the overlay frames."""
    rays = {name: [] for name in LANDMARKS}
    pixels, overlays = [], []
    for roll in ROLLS_DEG:
        env.robot.move_linear_to_tcp_pose(rolled_pose(hover_pose, np.radians(roll)), linear_speed=YAW_SPEED).wait()
        time.sleep(GRAB_SETTLE_S)
        camera_pose = env.get_robot_pose_se3() @ tcp_left_to_camera
        anchor = tuple(project_point_to_pixel(cap_center_model, intrinsics, camera_pose))
        if roll == ROLLS_DEG[0]:
            model = lock_sticker_for_pose(env, live, model, anchor)
        found, image = locate_landmarks(env, model, anchor)
        for name in LANDMARKS:
            if name in found:
                rays[name].append(pixel_ray(found[name], intrinsics, camera_pose))
        pixels.append({"roll_deg": roll, "camera_pose": camera_pose.tolist(),
                       **{name: [float(v) for v in found[name]] for name in LANDMARKS if name in found}})
        status = "  ".join(f"{name} {_pt(found[name]) if name in found else 'NOT FOUND'}" for name in LANDMARKS)
        print(f"[view] roll {roll:+4d}: {status}")
        marks = {"predict": anchor, "sticker": found.get("sticker"),
                 "rim": (found["rim"], found["rim_radius_px"]) if "rim" in found else None}
        lines = [f"{tag} roll {roll:+d}: yellow = current model, green = sticker, blue = rim fit"]
        overlay = image.copy()
        _draw_marks(overlay, marks, lines)
        overlays.append(overlay)
        live.show(image, lines, marks)
        cv2.waitKey(300)
    env.robot.move_linear_to_tcp_pose(hover_pose, linear_speed=YAW_SPEED).wait()
    return rays, pixels, overlays, model


def touch_cap(env, hover_pose, cap_center_model, cap_normal):
    """Descend along the tool axis until the controller feels contact; the TCP pose there, or None."""
    axis = hover_pose[:3, 2]  # tool z, pointing at the cap

    def height(pose):  # the TCP above the MODELLED cap plane
        return float(cap_normal.dot(pose[:3, 3] - cap_center_model))

    start = hover_pose.copy()
    start[:3, 3] += (height(hover_pose) - SEARCH_START_ABOVE_M) * axis
    print("[touch] descending until contact -- it only stops on contact, keep the e-stop at hand")
    env.robot.move_linear_to_tcp_pose(start, linear_speed=MOVE_SPEED).wait()
    control = env.robot.rtde_control
    direction = list(axis) + [0.0, 0.0, 0.0]
    control.moveUntilContact([v * TOUCH_SPEED for v in direction], direction, 0.1)
    contact = env.get_robot_pose_se3()
    env.robot.move_linear_to_tcp_pose(hover_pose, linear_speed=MOVE_SPEED).wait()
    print(f"[touch] contact with the TCP {height(contact) * 1000:+.1f}mm from the modelled plane")
    if height(contact) < -MAX_BELOW_MODEL_M:
        print("[touch] that is deeper than any cap-top contact can be -- discarded, check what was hit")
        return None
    return contact


def measure_pose(env, live, model, intrinsics, tag):
    """One pose: triangulate the cap centre, then touch it."""
    tcp_right_base_left = BASE_LEFT_TO_BASE_RIGHT @ env.robot_right.get_tcp_pose()
    cap_pose = env.get_bottle_cap_pose()  # the CURRENT model, only used to aim
    hover_pose = compute_hover_pose_above_bottle(cap_pose)
    print(f"[move] ur_left to the hover pose at {np.round(hover_pose[:3, 3], 4)}")
    env.move_robot_to_tcp_pose(hover_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)
    hover_pose = env.get_robot_pose_se3()  # roll about where the arm actually is

    rays, pixels, overlays, model = view_from_rolls(env, live, model, intrinsics, hover_pose, cap_pose[:3, 3], tag)
    normal = tcp_right_base_left[:3, 1]  # the cap normal by assumption: ur_right's TCP +y
    to_right_tcp = np.linalg.inv(tcp_right_base_left)
    result = {"tcp_right_base_left": tcp_right_base_left.tolist(), "hover_tcp": hover_pose.tolist(),
              "views": pixels, "translation_mm": {}, "ray_miss_mm": {}}
    centers = []
    for name in LANDMARKS:
        if len(rays[name]) < 2:
            print(f"[{name}] seen in {len(rays[name])} view(s) -- needs 2 to triangulate")
            continue
        center, misses = triangulate(rays[name])
        centers.append(center)
        result["translation_mm"][name] = [float(v) for v in (to_right_tcp @ np.append(center, 1.0))[:3] * 1000]
        result["ray_miss_mm"][name] = [round(m * 1000, 2) for m in misses]
        print(f"[{name}] cap centre in ur_right's TCP frame {np.round(result['translation_mm'][name], 2)} mm, "
              f"rays pass it at {result['ray_miss_mm'][name]} mm")

    try:
        contact = touch_cap(env, hover_pose, cap_pose[:3, 3], normal)
    except Exception as exc:  # the triangulation above is the measurement; never lose it to the check
        print(f"[touch] failed, keeping the triangulation without it: {exc}")
        contact = None
    if contact is not None:
        result["touch_tcp"] = contact.tolist()
        fingertips = contact[:3, 3] - FINGERTIPS_SHORT_OF_TCP_M * contact[:3, 2]
        result["touch_height_mm"] = float((to_right_tcp @ np.append(fingertips, 1.0))[1] * 1000)
        print(f"[touch] cap-top height in ur_right's TCP frame: y = {result['touch_height_mm']:.2f} mm")
        if centers:
            # The TCP's depth below the triangulated cap top at contact = how far it lies past the fingertips.
            result["tcp_past_fingertips_mm"] = round(-float(normal.dot(contact[:3, 3] - np.mean(centers, axis=0))) * 1000, 1)
            print(f"[touch] camera-vs-touch check: TCP {result['tcp_past_fingertips_mm']}mm past the fingertips "
                  f"(expected {FINGERTIPS_SHORT_OF_TCP_M * 1000:.1f})")
    return result, model, overlays


def retreat_left(env):
    """Back ur_left out along its own tool axis before any joint move, in case a crash left it at the cap."""
    pose = env.get_robot_pose_se3()
    pose[:3, 3] -= RETREAT_M * pose[:3, 2]
    if env.robot.is_tcp_pose_reachable(pose):
        env.robot.move_linear_to_tcp_pose(pose, linear_speed=MOVE_SPEED).wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=list(SPLIT_SEED_OFFSETS))
    parser.add_argument("--n-poses", type=int, default=3)
    parser.add_argument("--regrasp", action="store_true",
                        help="release the bottle at every pose and wait for it to be handed back (validation)")
    args = parser.parse_args()

    env = BottleStation(make_gripper(GRIPPER_PORT_RIGHT), with_instrumentation=False,
                        with_spectogram_model=False, with_spectogram=False, use_internal_ft=True)
    intrinsics = env.get_camera_intrinsics()
    print(f"[gripper] ur_right fingers at {env.gripper.get_current_width() * 1000:.1f}mm -- the bottle must be held")
    rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS[args.split])
    poses = generate_reachable_bottle_poses(args.n_poses, env.robot, env.robot_right, rng,
                                            blacklist=POSE_BLACKLIST[args.split])

    live, model, results = LiveView("cap transform"), StickerModel(), []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    try:
        retreat_left(env)
        for slot, (tcp_right_pose, _) in enumerate(poses):
            print(f"\n===== pose {slot + 1}/{len(poses)} =====")
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
            # Without --regrasp the bottle stays in ur_right's fingers for the whole run: no gates, and one
            # grasp for every pose, so pose-to-pose scatter cannot be re-grasp noise.
            env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
            if args.regrasp:
                _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
                input("Hold the bottle between the open fingers, then press Enter -- ur_right moves to the "
                      "pose and closes...")
            env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
            if args.regrasp:
                _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)
            tag = f"pose{slot + 1:02d}"
            result, model, overlays = measure_pose(env, live, model, intrinsics, tag)
            for roll, overlay in zip(ROLLS_DEG, overlays):
                cv2.imwrite(str(OUT_DIR / f"{stamp}_{tag}_roll{roll:+04d}.jpg"), overlay)
            results.append(result)
    finally:
        try:
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        except Exception as exc:  # cleanup must not stop the report from being written
            print(f"[cleanup] could not send ur_left home: {exc}")
        live.close()
        report(results, intrinsics, OUT_DIR / f"cap_transform_{stamp}.json")
        env.close()


def report(results, intrinsics, out_file):
    """Per-pose numbers, their spread, and the matrix to paste."""
    if not results:
        return
    old_mm = DEFAULT_TCP_RIGHT_TO_BOTTLE[:3, 3] * 1000
    print("\n===== result (ur_right TCP frame, mm; y = along the cap normal) =====")
    print(f"current constant      {np.round(old_mm, 2)}")
    per_pose = []
    for i, r in enumerate(results):
        for name, t in r["translation_mm"].items():
            print(f"pose {i + 1} {name:7s}       {np.round(t, 2)}   delta {np.round(np.asarray(t) - old_mm, 2)}")
        if "sticker" in r["translation_mm"]:
            x, y, z = r["translation_mm"]["sticker"]
            if "touch_height_mm" in r:
                y = r["touch_height_mm"]  # the touch, not the camera, sets the height
            per_pose.append([x, y, z])
            print(f"pose {i + 1} used          {np.round(per_pose[-1], 2)}   delta {np.round(np.asarray(per_pose[-1]) - old_mm, 2)}")
    offsets = [r["tcp_past_fingertips_mm"] for r in results if "tcp_past_fingertips_mm" in r]
    print(f"TCP past the fingertips, per pose (touch check): {offsets} mm")
    record = {"created": datetime.now().isoformat(timespec="seconds"), "intrinsics": np.asarray(intrinsics).tolist(),
              "tcp_left_to_camera": tcp_left_to_camera.tolist(),
              "base_left_to_base_right": BASE_LEFT_TO_BASE_RIGHT.tolist(),
              "old": DEFAULT_TCP_RIGHT_TO_BOTTLE.tolist(), "measurements": results}
    out_file.write_text(json.dumps(record, indent=2))
    print(f"[out] wrote {out_file}")
    if not per_pose:
        return
    per_pose = np.asarray(per_pose)
    mean = per_pose.mean(axis=0)
    spread = per_pose.std(axis=0, ddof=1) if len(per_pose) > 1 else np.zeros(3)
    print(f"mean over {len(per_pose)} poses   {np.round(mean, 2)}   spread (sd) {np.round(spread, 2)}")
    if np.any(spread > 2.0):
        print("WARNING the cap centre scatters by more than 2 mm across poses: a single constant cannot "
              "absorb that -- suspect BASE_LEFT_TO_BASE_RIGHT before pasting this")

    new = np.eye(4)
    new[:3, :3] = cap_rotation(DEFAULT_TCP_RIGHT_TO_BOTTLE)
    new[:3, 3] = mean / 1000
    print("\nDEFAULT_TCP_RIGHT_TO_BOTTLE = np.array(" + np.array2string(new, separator=", ", precision=8) + ")")
    out_file.write_text(json.dumps({**record, "new": new.tolist(), "per_pose_mm": per_pose.tolist()}, indent=2))


if __name__ == "__main__":
    main()
