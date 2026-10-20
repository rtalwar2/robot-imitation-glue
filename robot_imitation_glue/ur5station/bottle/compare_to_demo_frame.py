"""Put both arms where they were in one frame of a recorded demo and overlay the live wrist image on it.

    python -m robot_imitation_glue.ur5station.bottle.compare_to_demo_frame --episode 35 --frame 0

What it does:
  1. loads frame `--frame` of `--episode` from the raw expert dataset: the left arm's joints and the
     720p wrist image (wrist_image_original);
  2. finds the bottle pose that episode used: of the train split's poses (the same seeded list the
     collector used), the one whose hover pose lies closest to the recorded left TCP (printed, with the
     distance -- at --frame 0 the left arm is at hover, so this should be ~0 mm);
  3. left arm home -> right arm through the same re-grasp as collection/eval (neutral, open, you hand
     the closed bottle over, pose, close) -> left arm to the recorded joints. Every move is gated by Enter;
  4. a live window: the current wrist image blended with the recorded one.
     keys: 1 live | 2 recorded | 3 blend | 4 difference | a / d blend weight | s save a PNG | q quit

The arms are left where they are on quit.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from robot_imitation_glue.collect_data_bottle import GRIPPER_GRASP_WIDTH_M, GRIPPER_OPEN_WIDTH_M, _move_gripper_to
from robot_imitation_glue.hardware.gripper_startup import GRIPPER_PORT_RIGHT, make_gripper
from robot_imitation_glue.ur5station.bottle.splits import POSE_BLACKLIST, SPLIT_N_POSES, SPLIT_SEED_OFFSETS
from robot_imitation_glue.ur5station.bottle_station_env import BottleStation

# bottle_station_env already put bottle_experiment/ on sys.path.
from calibrate_and_hover_bottle import compute_hover_pose_above_bottle  # noqa: E402
from open_bottle_agent import (  # noqa: E402
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    RANDOM_SEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    generate_reachable_bottle_poses,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = _REPO_ROOT / "bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train"
WINDOW = "live vs recorded (1 live, 2 recorded, 3 blend, 4 diff, a/d weight, s save, q quit)"


def load_recorded_frame(raw_root: Path, episode: int, frame: int):
    """(left joints, left TCP position, recorded wrist image as RGB uint8) of one frame."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    columns = ["episode_index", "frame_index", "index", "joints", "robot_pose"]
    table = pd.concat(pd.read_parquet(f, columns=columns) for f in sorted(raw_root.glob("data/*/*.parquet")))
    row = table[(table.episode_index == episode) & (table.frame_index == frame)]
    if row.empty:
        n = (table.episode_index == episode).sum()
        raise SystemExit(f"episode {episode} has no frame {frame} ({n} frames)")
    row = row.iloc[0]

    dataset = LeRobotDataset(repo_id=None, root=str(raw_root), video_backend="pyav")
    image = dataset[int(row["index"])]["wrist_image_original"]  # float CHW in [0, 1]
    image = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return np.asarray(row["joints"], dtype=np.float64), np.asarray(row["robot_pose"][:3], dtype=np.float64), image


def render(mode: str, live: np.ndarray, recorded: np.ndarray, weight: float) -> np.ndarray:
    if mode == "live":
        out = live
    elif mode == "recorded":
        out = recorded
    elif mode == "diff":
        out = cv2.absdiff(live, recorded)
    else:
        out = cv2.addWeighted(live, weight, recorded, 1.0 - weight, 0)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    label = f"{mode}" + (f"  live {weight:.2f} / recorded {1 - weight:.2f}" if mode == "blend" else "")
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, default=35)
    parser.add_argument("--frame", type=int, default=0, help="frame within the episode (0 = the hover pose)")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    args = parser.parse_args()

    left_joints, recorded_tcp_position, recorded_image = load_recorded_frame(args.raw_root, args.episode, args.frame)
    print(f"[demo] episode {args.episode} frame {args.frame}: left joints {np.round(left_joints, 4)}")

    schunk = make_gripper(GRIPPER_PORT_RIGHT)
    env = BottleStation(schunk, with_instrumentation=False, with_spectogram_model=False,
                        with_spectogram=False, use_internal_ft=True)
    try:
        rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS["train"])
        bottle_poses = generate_reachable_bottle_poses(
            SPLIT_N_POSES["train"], env.robot, env.robot_right, rng, blacklist=POSE_BLACKLIST["train"]
        )
        distances = [np.linalg.norm(compute_hover_pose_above_bottle(cap)[:3, 3] - recorded_tcp_position)
                     for _, cap in bottle_poses]
        position = int(np.argmin(distances))
        print(f"[demo] bottle pose: train list position {position} (episode index {args.episode}); recorded left TCP "
              f"is {distances[position] * 1000:.1f} mm from its hover pose")
        if distances[position] > 0.005:
            print("[demo] WARNING: > 5 mm -- not at hover in this frame, or the pose list changed since collection")
        tcp_right_pose, _ = bottle_poses[position]

        input("Press Enter to move the LEFT arm home (Ctrl+C to abort)...")
        env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        input("Press Enter to move the RIGHT arm to its neutral pose and open the gripper (Ctrl+C to abort)...")
        env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
        _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
        input("Close the cap, hold the bottle between the open fingers, then press Enter -- ur_right moves to the "
              "episode's pose and closes (Ctrl+C to abort)...")
        env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
        _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)
        input("Press Enter to move the LEFT arm to the recorded joints (Ctrl+C to abort)...")
        env.robot.move_to_joint_configuration(left_joints, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        print(f"[move] left TCP now {np.linalg.norm(env.get_robot_pose_se3()[:3, 3] - recorded_tcp_position) * 1000:.1f} "
              "mm from the recorded one")

        mode, weight = "blend", 0.5
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            live = env._wrist_camera_subscriber.get_rgb_image_as_int()
            recorded = recorded_image
            if recorded.shape != live.shape:
                recorded = cv2.resize(recorded, (live.shape[1], live.shape[0]))
            shown = render(mode, live, recorded, weight)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            mode = {ord("1"): "live", ord("2"): "recorded", ord("3"): "blend", ord("4"): "diff"}.get(key, mode)
            if key == ord("a"):
                weight = max(0.0, weight - 0.1)
            elif key == ord("d"):
                weight = min(1.0, weight + 0.1)
            elif key == ord("s"):
                path = Path(f"outputs/compare_ep{args.episode}_f{args.frame}_{mode}_{time.strftime('%H%M%S')}.png")
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), shown)
                print(f"[save] {path}")
        cv2.destroyAllWindows()
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
