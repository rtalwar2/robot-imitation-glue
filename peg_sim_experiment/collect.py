"""Record scripted expert demonstrations into a LeRobotDataset laid out like the bottle --relative-actions datasets.

    python collect.py --root datasets/peg_sim/peg_train_200 --episodes 200

Only TRAIN holes are ever targeted. Frames are stored as PNG images (no video encoding: cheap to decode at 128 px
on a shared machine). After recording, meta/stats.json's action entry is replaced with the RELATIVE action
statistics the policy actually trains on (the bottle pipeline's write_relative_action_stats).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from expert import Expert
from peg_env import IMAGE_H, IMAGE_W, TRAIN_HOLES, PlugEnv, sample_episode

REPO = Path(__file__).resolve().parents[1]
STATE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "fx", "fy", "fz", "tx", "ty", "tz", "goal_x", "goal_y", "goal_z", "goal_nx", "goal_ny", "goal_nz"]
ACTION_NAMES = ["x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32"]
FEATURES = {
    "observation.images.wrist_image": {"dtype": "image", "shape": (IMAGE_H, IMAGE_W, 3),
                                       "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": (9,), "names": ACTION_NAMES},
    # the virtual sheet sensor, recorded for the instrumented arm; not an observation.* key, so lerobot never feeds
    # it to the policy (invariant 4) -- make_instrumented.py moves the chosen channel(s) into the action instead
    "instrumentation": {"dtype": "float32", "shape": (3,), "names": ["seated", "popped", "depth"]},
}
SEED_OFFSET = 0  # train episodes use seeds 0.. ; eval uses 100_000+ (eval.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(repo_id=args.root.name, fps=10, features=FEATURES, root=args.root, use_videos=False,
                               image_writer_threads=2)
    env = PlugEnv()
    lengths, seed, kept = [], SEED_OFFSET, 0
    while kept < args.episodes:
        spec = sample_episode(seed, TRAIN_HOLES)
        obs = env.reset(spec)
        expert = Expert(env, np.random.default_rng(seed))
        frames = []
        for _ in range(args.max_steps):
            clean, executed, done = expert.act()
            k = spec.target_hole
            depth = float(env.hole_coords()[0][k])
            sensor = np.array([env.inserted_hole() == k, env.popped[k], np.clip(depth, 0.0, 0.016)], np.float32)
            frames.append({"observation.images.wrist_image": obs["image"], "observation.state": obs["state"],
                           "action": clean, "instrumentation": sensor, "task": "insert the plug into the goal hole"})
            obs = env.step(executed)
            if done:
                break
        seed += 1
        if env.inserted_hole() != spec.target_hole:  # successful episodes only, as on the robot
            print(f"seed {seed - 1}: expert failed, skipped")
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        lengths.append(len(frames))
        kept += 1
        if kept % 20 == 0:
            print(f"{kept}/{args.episodes} episodes, median length {np.median(lengths):.0f} steps", flush=True)
    ds.finalize()
    env.close()

    sys.path.insert(0, str(REPO))
    from robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle import write_relative_action_stats
    write_relative_action_stats(args.root)
    (args.root / "peg_sim.json").write_text(json.dumps(
        {"episodes": kept, "seeds": [SEED_OFFSET, seed - 1], "train_holes": TRAIN_HOLES,
         "median_length": float(np.median(lengths))}, indent=2))


if __name__ == "__main__":
    main()
