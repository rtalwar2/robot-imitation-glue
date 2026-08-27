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
POSE_BLACKLIST = {"train": [], "val": [], "test": [0]}


def split_for_dataset_name(dataset_name: str) -> str:
    if "train" in dataset_name:
        return "train"
    if "val" in dataset_name:
        return "val"
    return "test"
