"""TEMPORARY repair script -- run ONCE, after `collect_data_bottle_opening_recover.py` has
successfully collected the 3 recovery episodes, to close the episode-index gap in
`bottle_opening_train`.

Background: meta/info.json's `total_episodes` had drifted ahead of what's actually on disk
(most likely a collection run that crashed mid-`save_episode()`), leaving episode indices 32, 33,
34 completely absent from both meta/episodes/ and data/ while info.json still counted them. This
made every `LeRobotDataset(repo_id=None, root=...)` load fall through to a Hub-download fallback
(local data judged "insufficient" because episode indices must be a contiguous 0..N-1 range) and
crash on `repo_id=None`. The 3 recovery episodes then got appended AFTER the inflated count, so
right after that script finishes, the real episodes are non-contiguous:
{0..31, 35..(last)}, informally.

The recovery run also produced one duplicate: the script crashed after episode 51 (pose 32) was
saved, and restarting it reset its internal recovery cursor, so it re-ran pose 32 as episode 52
before continuing on to episodes 53 (pose 33) and 54 (pose 34). Confirmed by comparing each
episode's starting `robot_pose` -- 51 and 52 match to within real-world repeat noise, 53 and 54
are clearly different poses. Episode 52 is deleted (EPISODES_TO_DELETE below); 51 is kept, since
that is the one collected first.

This script closes the gap unconditionally, without assuming exactly which indices are missing:
    1. Delete every episode listed in EPISODES_TO_DELETE: drop its rows from data/ and its row
       from meta/episodes/, and subtract its frame count from `total_frames`.
    2. Find every episode index actually present (after that deletion).
    3. Remap them to a dense 0..N-1 range in their original relative order (old index 0 -> new
       index 0, old index 1 -> new index 1, ..., old index 35 -> new index 32, and so on --
       whatever the true gap turns out to be, gaps from step 1 included).
    4. Rewrite the `episode_index` column in every affected file under meta/episodes/ and data/.
       Nothing else changes: chunk/file locations, the global `index`/`frame_index` columns, and
       every video file are untouched, because episode_index is just a data column here, not part
       of any file's physical location. (A deleted episode's frames stay physically present in
       its shared video file -- other episodes' frames live in the same file -- but nothing
       references them any more, so they're inert, not just relabelled.)
    5. Fix meta/info.json: `total_episodes` and `splits.train` to the real count.
    6. Verify: reload with LeRobotDataset(repo_id=None, root=...) and confirm it now succeeds,
       with the expected episode count and no gaps.

Always makes a full backup copy of the dataset directory first (see BACKUP_SUFFIX below) --
this is the only copy of real robot data, and it's cheap to double back up an already-crashed
dataset.

Delete this file (and collect_data_bottle_opening_recover.py) once the repair is verified -- both
exist only for this one incident.
"""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAW_ROOT = Path("/home/rtalwar/robot-imitation-glue/datasets/bottle_experiment/expert/bottle_opening_train")
BACKUP_SUFFIX = "_pre_gap_repair_backup"

# Duplicate collection of pose 32 caused by restarting the recovery script mid-run (its recovery
# cursor is not itself resumable, unlike the normal n_recorded_episodes-based cycling) -- keep
# episode 51 (collected first), drop 52.
EPISODES_TO_DELETE = [52]


def _rewrite_episode_index(path: Path, remap: dict[int, int]) -> bool:
    """Remap the episode_index column of one parquet file in place. Returns whether anything in
    this file actually needed remapping (files entirely below the gap are left untouched)."""
    table = pq.read_table(path)
    old_values = table.column("episode_index").to_pylist()
    if not any(v in remap and remap[v] != v for v in old_values):
        return False
    new_values = [remap[v] for v in old_values]
    table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", [new_values])
    pq.write_table(table, path)
    return True


def _delete_episode_rows(path: Path, episodes_to_delete: set[int]) -> int:
    """Drop every row belonging to episodes_to_delete from one parquet file in place. Returns
    the number of rows removed (0 if this file had none)."""
    table = pq.read_table(path)
    episode_index = table.column("episode_index").to_pylist()
    keep_mask = [e not in episodes_to_delete for e in episode_index]
    n_removed = keep_mask.count(False)
    if n_removed == 0:
        return 0
    table = table.filter(pa.array(keep_mask))
    pq.write_table(table, path)
    return n_removed


def _recompact_global_index(root: Path, data_files: list[Path], episode_meta_files: list[Path]) -> None:
    """After deleting whole episodes, the global `index` column (data/) and each episode's
    `dataset_from_index`/`dataset_to_index` (meta/episodes/) still reflect PHYSICAL offsets from
    before the deletion -- rows physically shifted left when the deleted rows were removed, but
    those bookkeeping columns did not. Left alone, every episode after a deletion point resolves
    to the wrong rows (confirmed: episodes downstream of a deleted one read back completely
    unrelated frames). This walks both file sets in the same physically-written order and
    reassigns a dense, contiguous numbering to close the gap left in the numbering itself, exactly
    mirroring what the episode_index remap does for episode numbers.

    `frame_index` (position within its own episode) is untouched -- unaffected by any of this,
    since no episode had rows removed from its OWN middle, only whole other episodes removed
    around it.
    """
    running_total = 0
    for f in data_files:
        table = pq.read_table(f)
        n = table.num_rows
        if n == 0:
            continue
        new_index = list(range(running_total, running_total + n))
        table = table.set_column(table.schema.get_field_index("index"), "index", [new_index])
        pq.write_table(table, f)
        running_total += n

    running_total = 0
    for f in episode_meta_files:
        table = pq.read_table(f)
        lengths = table.column("length").to_pylist()
        if not lengths:
            continue
        from_indices, to_indices = [], []
        for length in lengths:
            from_indices.append(running_total)
            running_total += length
            to_indices.append(running_total)
        table = table.set_column(
            table.schema.get_field_index("dataset_from_index"), "dataset_from_index", [from_indices]
        )
        table = table.set_column(
            table.schema.get_field_index("dataset_to_index"), "dataset_to_index", [to_indices]
        )
        pq.write_table(table, f)


def repair(root: Path, episodes_to_delete: list[int] = EPISODES_TO_DELETE) -> None:
    backup_dir = root.parent / (root.name + BACKUP_SUFFIX)
    if backup_dir.exists():
        raise RuntimeError(f"{backup_dir} already exists -- remove it first if you really want to re-backup")
    print(f"[repair] backing up {root} -> {backup_dir}")
    shutil.copytree(root, backup_dir)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    episode_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_files = sorted((root / "data").rglob("*.parquet"))

    if episodes_to_delete:
        to_delete = set(episodes_to_delete)
        removed_frames = sum(_delete_episode_rows(f, to_delete) for f in data_files)
        removed_meta_rows = sum(_delete_episode_rows(f, to_delete) for f in episode_meta_files)
        print(f"[repair] deleted episodes {sorted(to_delete)}: {removed_frames} frames, "
              f"{removed_meta_rows} meta/episodes row(s)")
        print("[repair] recompacting global index / dataset_from_index / dataset_to_index "
              "after deletion")
        _recompact_global_index(root, data_files, episode_meta_files)

    present = set()
    for f in episode_meta_files:
        present.update(pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist())
    present = sorted(present)

    # Recomputed from the actual retained data, never trusted from info.json: the crash that
    # caused the episode-index gap left `total_frames` inflated by the same amount as
    # `total_episodes` (confirmed: even before any of this repair, the real 48 episodes' rows
    # summed to 15208, not the 16324 info.json claimed) -- so delta-adjusting a stale number
    # would just carry the error forward.
    actual_total_frames = sum(
        length
        for f in episode_meta_files
        for length in pq.read_table(f, columns=["length"]).column("length").to_pylist()
    )

    remap = {old: new for new, old in enumerate(present)}
    n_changed_indices = sum(1 for old, new in remap.items() if old != new)
    print(f"[repair] {len(present)} real episodes found (was claiming {info['total_episodes']}); "
          f"{n_changed_indices} episode indices will be relabelled")

    changed_files = 0
    if n_changed_indices > 0:
        for f in episode_meta_files:
            if _rewrite_episode_index(f, remap):
                changed_files += 1
        for f in data_files:
            if _rewrite_episode_index(f, remap):
                changed_files += 1
        print(f"[repair] rewrote episode_index in {changed_files} parquet file(s)")

    print(f"[repair] total_frames: info.json claimed {info['total_frames']}, actual retained data "
          f"is {actual_total_frames}")
    info["total_episodes"] = len(present)
    info["total_frames"] = actual_total_frames
    info["splits"] = {"train": f"0:{len(present)}"}
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[repair] meta/info.json: total_episodes -> {len(present)}, total_frames -> {actual_total_frames}, "
          f"splits.train -> 0:{len(present)}")

    verify(root, expected_episodes=len(present), expected_frames=actual_total_frames)


def verify(root: Path, expected_episodes: int, expected_frames: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=None, root=str(root))
    assert dataset.meta.total_episodes == expected_episodes, (
        f"total_episodes is {dataset.meta.total_episodes}, expected {expected_episodes}"
    )
    assert dataset.meta.total_frames == expected_frames, (
        f"total_frames is {dataset.meta.total_frames}, expected {expected_frames}"
    )
    # Every episode boundary must be exactly contiguous (no residual gaps/overlaps from the
    # index recompaction) and independently readable.
    running_to = 0
    for episode_index in range(dataset.meta.total_episodes):
        bounds = dataset.meta.episodes[episode_index]
        assert bounds["dataset_from_index"] == running_to, (
            f"episode {episode_index}: dataset_from_index={bounds['dataset_from_index']}, expected {running_to}"
        )
        running_to = bounds["dataset_to_index"]
        _ = dataset[bounds["dataset_from_index"]]
        _ = dataset[bounds["dataset_to_index"] - 1]
    assert running_to == expected_frames, f"episode boundaries cover {running_to} frames, expected {expected_frames}"
    print(f"[repair] verified: loads cleanly, {dataset.meta.total_episodes} contiguous episodes, "
          f"{dataset.meta.total_frames} frames, every episode boundary contiguous and readable")


if __name__ == "__main__":
    repair(RAW_ROOT)
