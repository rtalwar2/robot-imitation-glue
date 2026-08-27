# Repairing a broken LeRobot (v3.0) dataset

This is a runbook for one specific, recurring failure mode of `LeRobotDataset` (the
`lerobot` library's on-disk dataset format, codebase_version `v3.0`) recorded with
`robot_imitation_glue.dataset_recorder.LeRobotDatasetRecorder`: **the dataset's episode
count/frame count in `meta/info.json` drifts ahead of what is actually written to disk**,
usually because the recording process was killed hard (crash, SIGKILL, driver fault, power
loss) instead of exiting cleanly.

Read this top to bottom before touching any files. Every destructive step below has a
non-destructive diagnostic step before it — do not skip the diagnosis and jump to the fix.

## 1. How to recognize this failure

You will see something like this when you try to load the dataset:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id=None, root="datasets/some_experiment/expert/some_dataset")
```

```
huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. ...
Repository Not Found for url: https://huggingface.co/api/datasets/None/refs.
```

**This error message is misleading.** It looks like a Hugging Face Hub / authentication
problem. It is not. `repo_id=None` is passed on purpose (local-only dataset, no Hub repo).
The actual problem is two or three layers upstream of this crash — see §2.

Do **not** try to "fix" this by passing a real `repo_id`, logging into the Hub, or setting
`HF_TOKEN`. None of that is the problem.

## 2. Why this happens (root cause)

This is a real bug/gap in `lerobot`'s writer, in
`lerobot/src/lerobot/datasets/dataset_metadata.py`, `LeRobotDatasetMetadata.save_episode()`:

```python
def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
    episode_dict = {...}
    self._save_episode_metadata(episode_dict)   # BUFFERED -- only physically flushed to
                                                  # meta/episodes/*.parquet every
                                                  # `metadata_buffer_size` episodes (default 10),
                                                  # or at finalize()

    # Update info
    self.info["total_episodes"] += 1
    self.info["total_frames"] += episode_length
    ...
    write_info(self.info, self.root)             # WRITTEN TO DISK IMMEDIATELY, every episode
```

So there are two different persistence speeds for two things that are supposed to move
together:

- **`meta/info.json`'s counters** (`total_episodes`, `total_frames`) are written to disk
  *immediately*, on every single `save_episode()` call.
- **The episode's own metadata row** (`meta/episodes/*.parquet`) and its **frame data**
  (`data/*.parquet`, and video files if there are any image/video features) are written in
  a *buffered/batched* way, for I/O efficiency.

If the process dies in the window between those two things — anything that skips the
normal clean-exit path (a `finally:` block calling `dataset.finalize()`) — `info.json` ends
up "knowing about" episodes whose actual rows never made it to disk. A clean `Ctrl+C` is
usually fine (it still unwinds through `finally:`); a hard kill, an out-of-memory kill, a
crash inside a video encoder or a camera/BLE driver, or a segfault is not.

The result: `info.json` says e.g. `total_episodes: 51`, but only 48 of those episode
indices actually have rows in `meta/episodes/` and `data/`. The missing ones are usually a
short contiguous run (e.g. episodes 32, 33, 34) — whatever hadn't been flushed yet when the
process died.

**Why this then breaks loading (not just resuming):** `LeRobotDataset.__init__` (full
read-mode construction, e.g. `LeRobotDataset(repo_id=None, root=...)`) builds a
`DatasetReader` and calls `try_load()`, which calls `_check_cached_episodes_sufficient()`
(`lerobot/src/lerobot/datasets/dataset_reader.py`):

```python
requested_episodes = set(range(self._meta.total_episodes))   # e.g. {0, 1, ..., 50}
available_episodes = set(hf_dataset.unique("episode_index"))  # e.g. {0..31, 35..50} -- 32,33,34 missing
if not requested_episodes.issubset(available_episodes):
    return False   # "local data insufficient"
```

Episode indices must be an exact, dense `0..total_episodes-1` range. If even one is
missing, the local cache is judged incomplete, and `LeRobotDataset.__init__` falls back to
"download the rest from the Hub" — which is what then calls `get_safe_version(repo_id=None, ...)`
and produces the confusing 404 above.

**Important asymmetry to know about:** `LeRobotDataset.resume()` (used by
`LeRobotDatasetRecorder` to append more episodes to an existing dataset) does **not** hit
this check — it only loads `meta/info.json` and friends, it never calls
`DatasetReader.try_load()`. So **you can keep recording/appending to an already-broken
dataset** (new episodes just land after the current, inflated `total_episodes`); you only
hit the crash when something tries to actually *read* the dataset (training, dataset
preparation scripts, `LeRobotDataset(repo_id=None, root=...)` directly, etc.).

## 3. Diagnose before doing anything else

Run this against the dataset root (adjust the path). It never writes anything.

```python
import pyarrow.parquet as pq
import json
from pathlib import Path

root = Path("PATH/TO/YOUR/DATASET")   # the directory containing meta/ and data/

with open(root / "meta" / "info.json") as f:
    info = json.load(f)
claimed_episodes = info["total_episodes"]
claimed_frames = info["total_frames"]

episode_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
present_episodes = set()
actual_frames = 0
for f in episode_meta_files:
    table = pq.read_table(f, columns=["episode_index", "length"])
    present_episodes.update(table.column("episode_index").to_pylist())
    actual_frames += sum(table.column("length").to_pylist())
present_episodes = sorted(present_episodes)

missing = sorted(set(range(claimed_episodes)) - set(present_episodes))

print(f"info.json claims: {claimed_episodes} episodes, {claimed_frames} frames")
print(f"actually present: {len(present_episodes)} episodes, {actual_frames} frames")
print(f"missing episode indices: {missing}")

# Cross-check against the data/ files directly (should agree with meta/episodes above --
# if it doesn't, something else is wrong and you should stop and investigate further
# before repairing anything).
data_files = sorted((root / "data").rglob("*.parquet"))
data_rows = sum(pq.read_table(f, columns=["episode_index"]).num_rows for f in data_files)
print(f"actual rows in data/: {data_rows} (should equal 'actually present' frames above)")
```

Expected output for the broken case looks like:
```
info.json claims: 51 episodes, 16324 frames
actually present: 48 episodes, 15208 frames
missing episode indices: [32, 33, 34]
actual rows in data/: 15208
```

**Do not trust `info.json`'s `total_frames` even after you know the missing episode
indices.** In the one real incident this was written from, `total_frames` was *also*
inflated (by exactly the frame count the missing episodes would have had) — it is not just
`total_episodes` that drifts. Always recompute `total_frames` from the actual data (sum of
`length` over the retained `meta/episodes/` rows, or equivalently count actual `data/` rows)
rather than doing arithmetic on the stored value.

If you additionally suspect a *duplicate* episode (e.g. a recovery/resume script was
restarted and re-ran a pose/task it had already completed successfully), compare each
candidate episode's first frame of some position-dependent feature (e.g. `robot_pose`,
`state`, whatever encodes "where was the robot/task at the start of this episode"):

```python
def first_frame_feature(root, episode_index, feature_key):
    """Read one feature's value at the very first frame of one episode, without needing
    the dataset to load cleanly (works even while the dataset is still broken)."""
    import pyarrow.parquet as pq
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(f, columns=["episode_index", "data/chunk_index", "data/file_index",
                                           "dataset_from_index"])
        rows = table.to_pylist()
        for row in rows:
            if row["episode_index"] == episode_index:
                data_path = root / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
                data_table = pq.read_table(data_path, columns=["episode_index", feature_key])
                eps = data_table.column("episode_index").to_pylist()
                vals = data_table.column(feature_key).to_pylist()
                first_local_row = eps.index(episode_index)
                return vals[first_local_row]
    raise KeyError(f"episode {episode_index} not found")
```

Two episodes recorded for the *same* real-world condition will match to within real-world
repeat noise (tiny floating-point differences); two episodes for genuinely different
conditions will differ substantially. Use this to decide which duplicate to keep (usually:
keep the one collected first) before repairing.

## 4. The repair, step by step

**Always back up the whole dataset directory first.** This is the only copy of real
collected data; a `shutil.copytree` is cheap insurance.

```python
import shutil
from pathlib import Path

root = Path("PATH/TO/YOUR/DATASET")
backup = root.parent / (root.name + "_pre_repair_backup")
assert not backup.exists(), f"{backup} already exists -- remove it first if you mean to re-backup"
shutil.copytree(root, backup)
```

### 4a. If you're deleting any episodes (duplicates, corrupted partials, etc.)

**This step is subtle — read it fully, do not skip straight to "just delete the rows".**
Deleting rows from `data/*.parquet` and their row from `meta/episodes/*.parquet` is not
enough by itself. Two other things reference the *physical position* of rows and go stale
the moment you delete anything before them:

- The global `index` column in every `data/*.parquet` file (a dataset-wide running frame
  counter, NOT reset per file).
- Every episode's `dataset_from_index` / `dataset_to_index` in `meta/episodes/` (the global
  frame range that episode occupies).

If you delete rows and don't recompute these, every episode whose data comes *after* the
deleted rows (in file-write order) will silently read back the **wrong frames** — not an
error, just quietly wrong data, which is much worse than a crash. This was confirmed by
testing on a scratch copy: episodes downstream of a deletion read completely unrelated
content until this recompaction was added.

```python
import pyarrow as pa
import pyarrow.parquet as pq


def delete_episode_rows(path, episodes_to_delete: set[int]) -> int:
    """Drop every row belonging to episodes_to_delete from one parquet file. Returns the
    number of rows removed (0 if this file had none)."""
    table = pq.read_table(path)
    episode_index = table.column("episode_index").to_pylist()
    keep_mask = [e not in episodes_to_delete for e in episode_index]
    n_removed = keep_mask.count(False)
    if n_removed == 0:
        return 0
    table = table.filter(pa.array(keep_mask))
    pq.write_table(table, path)
    return n_removed


def recompact_global_index(data_files, episode_meta_files):
    """After deleting whole episodes, reassign a dense, contiguous global `index` (data/)
    and `dataset_from_index`/`dataset_to_index` (meta/episodes/), in the same
    physically-written file order the data was originally recorded in. `frame_index`
    (position within its OWN episode) needs no change -- it was never affected."""
    running_total = 0
    for f in data_files:  # must be sorted in original chunk/file order
        table = pq.read_table(f)
        n = table.num_rows
        if n == 0:
            continue
        new_index = list(range(running_total, running_total + n))
        table = table.set_column(table.schema.get_field_index("index"), "index", [new_index])
        pq.write_table(table, f)
        running_total += n

    running_total = 0
    for f in episode_meta_files:  # must be sorted in original chunk/file order
        table = pq.read_table(f)
        lengths = table.column("length").to_pylist()
        if not lengths:
            continue
        from_indices, to_indices = [], []
        for length in lengths:
            from_indices.append(running_total)
            running_total += length
            to_indices.append(running_total)
        table = table.set_column(table.schema.get_field_index("dataset_from_index"), "dataset_from_index", [from_indices])
        table = table.set_column(table.schema.get_field_index("dataset_to_index"), "dataset_to_index", [to_indices])
        pq.write_table(table, f)
```

Note on video files: if the dataset has image/video features, a deleted episode's *video*
frames are usually **not** removed by this (they typically live in a shared `.mp4` file
alongside other episodes' frames, addressed by chunk/file + timestamp, not by
episode-index-shaped file names). That's fine — once no `meta/episodes/` row references
those frames any more, they're just inert bytes taking up a little extra disk space, not a
correctness problem. Re-encoding video to physically remove them is usually not worth the
risk/effort; leave it.

### 4b. Close the episode-index gap (always do this part)

Episode indices must be a dense `0..N-1` range with no gaps, in original relative order.
This closes gaps from missing episodes (§2) *and* any gaps you just created by deleting
duplicates (§4a) — one generic pass handles both, because it works from "whatever indices
are actually present now", not from any assumption about which indices are supposed to be
missing.

```python
def rewrite_episode_index(path, remap: dict[int, int]) -> bool:
    """Remap the episode_index column of one parquet file in place. Returns whether this
    file needed any change."""
    table = pq.read_table(path)
    old_values = table.column("episode_index").to_pylist()
    if not any(v in remap and remap[v] != v for v in old_values):
        return False
    new_values = [remap[v] for v in old_values]
    table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", [new_values])
    pq.write_table(table, path)
    return True


episode_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
data_files = sorted((root / "data").rglob("*.parquet"))

present = set()
for f in episode_meta_files:
    present.update(pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist())
present = sorted(present)

remap = {old: new for new, old in enumerate(present)}   # old index -> new dense index

for f in episode_meta_files:
    rewrite_episode_index(f, remap)
for f in data_files:
    rewrite_episode_index(f, remap)
```

Nothing else needs to change here: chunk/file locations and video files are untouched,
because `episode_index` is just a data column, not part of any file's physical location or
name.

### 4c. Fix `meta/info.json`

Recompute everything from the *actual retained data* — never adjust the old (possibly
already-wrong, see §3) values by a delta.

```python
import json

info_path = root / "meta" / "info.json"
info = json.loads(info_path.read_text())

actual_total_frames = sum(
    length
    for f in episode_meta_files
    for length in pq.read_table(f, columns=["length"]).column("length").to_pylist()
)

info["total_episodes"] = len(present)
info["total_frames"] = actual_total_frames
info["splits"] = {"train": f"0:{len(present)}"}   # adjust the split name if it isn't "train"
info_path.write_text(json.dumps(info, indent=4))
```

(If the dataset uses `meta/stats.json` for dataset-level normalization stats, that is
computed by aggregating each episode's own stats and does not need per-episode-position
information — it does not need touching here. The per-episode `stats/*` columns inside
`meta/episodes/*.parquet` do go slightly stale after a deletion+recompaction in step 4a
(they cache aggregates like `stats/index/min` computed under the old numbering) — this is a
known, accepted loose end: those columns are an internal read-speedup cache, not something
training or `meta/stats.json` reads from, so leaving them stale does not affect correctness
of data loading or training.)

### 4d. Verify, thoroughly, before trusting it

Do not just check that the dataset "loads" — check every episode boundary is truly
contiguous and that both a first and last frame of every episode is independently
readable. A silent off-by-something here reads back *wrong data*, not an error.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id=None, root=str(root))
assert dataset.meta.total_episodes == len(present)
assert dataset.meta.total_frames == actual_total_frames

running_to = 0
for episode_index in range(dataset.meta.total_episodes):
    bounds = dataset.meta.episodes[episode_index]
    assert bounds["dataset_from_index"] == running_to, (episode_index, bounds)
    running_to = bounds["dataset_to_index"]
    _ = dataset[bounds["dataset_from_index"]]   # first frame of this episode is readable
    _ = dataset[bounds["dataset_to_index"] - 1]  # last frame of this episode is readable
assert running_to == actual_total_frames

print(f"OK: {dataset.meta.total_episodes} contiguous episodes, {dataset.meta.total_frames} frames")
```

If you have any independent way to sanity-check *content* (not just structure) — e.g. a
feature that should differ meaningfully between episodes (a starting pose, a task label) —
spot-check a few episodes near any boundary you touched, the same way §3's duplicate-check
snippet does. Structural correctness (this step) and content correctness are different
things; both matter.

## 5. Test on a throwaway copy first — every time

Before running any of §4 against the real dataset, copy it somewhere disposable and run the
exact same repair there first:

```bash
rm -rf /tmp/repair_test
mkdir -p /tmp/repair_test
cp -r PATH/TO/YOUR/DATASET /tmp/repair_test/dataset_copy
# run the repair against /tmp/repair_test/dataset_copy, verify it fully, THEN run it for real
```

This is not optional caution — it is how the bug in §4a (stale `dataset_from_index` after
deletion) and the bug in §3 (stale `total_frames` even before any repair) were actually
caught, on scratch copies, before they could have corrupted the real dataset a second time.
A repair script that "looks obviously correct" is exactly the kind of thing worth testing
before trusting.

## 6. If you also need to re-collect genuinely missing episodes

If the missing episodes need to be re-recorded (not just numbered away), be aware of one
specific trap: if your collection script resumes an existing dataset (via
`LeRobotDataset.resume()` / `LeRobotDatasetRecorder`) and cycles through a fixed list of
"things to record" (poses, tasks, conditions, ...) using some kind of "which one to do next"
counter, **make sure that counter is derived from something persisted on disk, not from an
in-memory variable that resets to zero every time the script restarts.** If your script
crashes after successfully saving one recovery episode, and you restart it, and it does not
know it already did that one — it will silently re-record a duplicate instead of moving on
to the next missing item. (This is exactly what produced the duplicate-episode case in §3's
diagnosis snippet.) Prefer deriving "what's left to do" from the dataset's actual current
`n_recorded_episodes`/content rather than from a counter that only lives in the running
process's memory.

## 7. Minimal checklist

1. Diagnose (§3) — never skip this, never assume you know what's missing without checking.
2. Decide whether anything needs deleting (duplicates, partial junk) — confirm with a
   content check (§3's snippet), not just a guess.
3. Test the whole repair on a throwaway copy (§5) and verify it fully (§4d) before touching
   the real dataset.
4. Back up the real dataset.
5. Run: delete (§4a, if needed) → close the episode-index gap (§4b) → fix `info.json`
   (§4c) → verify (§4d) — against the real dataset.
6. Only after verification passes, proceed to whatever needed the dataset in the first
   place (training-data preparation scripts, screening scripts, etc.).
