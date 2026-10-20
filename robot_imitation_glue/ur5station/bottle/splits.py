"""The bottle experiment's pose-split definitions, shared by collection and evaluation.

A split is a seeded pose distribution: `RANDOM_SEED + SPLIT_SEED_OFFSETS[split]` feeds
`generate_reachable_bottle_poses`, so the same split name always reproduces the same poses. One
seeded stream per split -- sharing a single stream is how the original train/val pose leak happened
(the 25 val poses were the first 25 train poses).

Lives in its own module so collection (`collect_data_bottle_opening`) and evaluation
(`eval_bottle`) cannot drift apart on what a split means: eval rollouts must draw poses from the
same distribution the policy trained under (protocol section 1.8 -- OOD varies ONLY the appearance
axis, never the pose distribution), and a second hand-maintained copy of these numbers would be a
silent way to violate that.
"""

# Added to open_bottle_agent.RANDOM_SEED to seed each split's pose stream.
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}

# How many constraint-passing poses each split draws.
SPLIT_N_POSES = {"train": 100, "val": 25, "test": 5}

# Pose indices to skip, per split. Indices are per-split because each split is seeded differently,
# so index 0 of "test" is a different pose from index 0 of "train". Excluding an index does not
# shift the others (see generate_reachable_bottle_poses), so the generator just samples one extra
# pose to make up the count.
#
# Only exclude poses that are impractical to *collect or evaluate* on -- the bottle cannot be
# re-closed by hand between episodes, the right arm fouls something, and so on. Excluding a pose
# because the robot finds the task hard there biases the success rate upward, and for evaluation
# poses that directly inflates the headline result. Reachability is already filtered by
# is_tcp_pose_reachable / is_opening_motion_reachable, so anything reaching this list is a
# judgement call worth writing down.
# test 9, 10, 19, 23 (2026-10-12, generic_50ep eval): the poses that were at the rollout the operator
# stopped on -- 9 left-arm protective stop 4.4 s in, the rest rig problems of the pose as judged by the
# operator. Entries are POSE INDICES (eval prints "pose index N" per rollout), not rollout numbers: after
# one entry the two differ, and entering rollout numbers 17/20 first removed the already-evaluated poses
# 17 and 20 instead of 19 and 23. With this list the test split's 20 poses are exactly the ones
# generic_50ep was evaluated on (1-8, 11-18, 20, 21, 22, 24).
POSE_BLACKLIST = {"train": [6,10,68], "val": [], "test": [0, 9, 10, 19, 23]}


def split_for_dataset_name(dataset_name: str) -> str:
    if "train" in dataset_name:
        return "train"
    if "val" in dataset_name:
        return "val"
    return "test"
