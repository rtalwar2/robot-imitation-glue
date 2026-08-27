# Bottle experiment — operator guide

Step-by-step runbook with exact commands. Assumes the conda env and the repo root:

```bash
conda activate imitation_button_py312
cd /home/rtalwar/robot-imitation-glue
```

Design rationale lives in `Experiment_Protocol.md`; this file is only *what to run, in what order,
and what to check before moving on*. All bottle tooling is under
`robot_imitation_glue/ur5station/bottle/` (repair scripts in `bottle/repair/`, generated training
configs in `bottle/configs/`).

---

## 0. Pre-flight (once per rig change, not per session)

- [ ] **UR payload compensation** on the left arm: with nothing in contact, jog through the task's
  orientations and watch `ft` in rerun. Near-constant → OK. Swinging by newtons → fix the payload
  mass/CoG in the UR controller before collecting; no baseline subtraction can rescue it.
- [ ] **Stickers**: cap sensor voltages unchanged with stickers on (read a fully-open and
  fully-closed cap per appearance), and the sticker is visible in the wrist frame.
- [ ] **Sensor feeds running**: `bottle_ble_reader.py` (cap sensor → DDS topic `Bottle`) and the
  Kaldi spectrogram publisher (`KaldiSpectrogram`). The collection loop hard-fails on a dead cap
  sensor, but only at the first checkpoint — start them first.
- [ ] Wrist RealSense reachable (the env raises if 720p won't start — no silent fallback).

## 1. Collect demonstrations

Split is selected by the dataset name in
`robot_imitation_glue/ur5station/bottle/collect_data_bottle_opening.py` (`dataset_name = "bottle_opening_train"`;
`"...val"` / anything else selects the val/test pose seeds and counts: 100/25/5 poses). Pose
blacklist per split in the same file — **feasibility exclusions only, never difficulty**.

```bash
python -m robot_imitation_glue.ur5station.bottle.collect_data_bottle_opening
```

The loop stops at the target count on its own, retries failed poses, and resumes an interrupted
dataset (re-run the same command to top up). During touch-point verification: click = correct,
Enter = accept, `t` = retake a blurry frame, `q` = abort.

**If a run crashes hard** (killed mid-save, episode-count drift, `RepositoryNotFoundError` on
load): read `docs/repairing_broken_lerobot_datasets.md` *top to bottom* before touching files. The
one-off scripts from the previous incident are in
`robot_imitation_glue/ur5station/bottle/repair/` as worked examples.

## 2. Prepare datasets

```bash
python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle
```

Builds the 8 prepared datasets ({9,12}-dim action × {100,75,50,25}%) from the raw root, filters to
successful episodes, applies the FT drift correction, and finishes by writing per-level
`audio_stats.json` into every prepared root. If the datasets already exist and only the stats are
missing (or the raw data changed is *not* the case), retrofit without re-encoding video:

```bash
python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle --stats-only
```

Check: each `datasets/bottle_experiment/prepared/bottle_9d_*/audio_stats.json` exists, and its
`episodes` list matches `episode_order.json`'s level.

## 3. Screening (privilege gate + modality suitability)

**Fix the suitability threshold before looking at any score.** Run all four on the 100% dataset:

```bash
for m in proprio image audio ft; do
  python -m robot_imitation_glue.ur5station.bottle.screen_instrumentation \
      --dataset-root datasets/bottle_experiment/prepared/bottle_9d_100 \
      --input $m --report outputs/screening_report.json
done
```

Decision rules:

| Result | Action |
|---|---|
| **proprio** R² high (near the modality scores) | the signal is not privileged — the task is unsuitable; stop and rethink |
| a modality's R² clears the pre-registered margin over proprio | it passes the filter → design_a pretrains it |
| below the margin | it does not pass → inside design_a it keeps the *generic* init |

Status: already run once on the first 51 episodes — proprio 0.182, image 0.787, audio 0.606,
FT 0.096 (see the preliminary table in `Experiment_Protocol.md`). One run each; repeat with more
seeds before treating as final, and re-run if substantially more data is collected.

## 4. Baseline check — is there enough data? (protocol §1.6)

The iterative N-finding loop. Everything trains on the **100% level of what exists so far**.

1. Generate the baseline config and train it:

   ```bash
   python -m robot_imitation_glue.ur5station.bottle.generate_configs --arms from_scratch
   lerobot-train --config_path=robot_imitation_glue/ur5station/bottle/configs/bottle_from_scratch_100.json
   ```

2. Evaluate with **20 real rollouts** on training-distribution conditions (ID stickers):

   ```bash
   python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
       --checkpoint outputs/train/bottle/from_scratch_100/checkpoints/100000/pretrained_model \
       --condition id --n-rollouts 20 --timeout-seconds 65
   ```

   **The rollout timeout is 2× the median demonstration duration** (protocol §1.8). Measured on
   the current 51 episodes: median 311 frames @ 10 Hz = 31.1 s → **65 s** (the script's default).
   **Re-derive it whenever N changes** — after collecting more episodes, recompute and pass the
   new value explicitly:

   ```bash
   python - <<'PY'
   import numpy as np, pyarrow.parquet as pq
   from pathlib import Path
   lengths = [l for f in sorted(Path("datasets/bottle_experiment/prepared/bottle_9d_100/meta/episodes").glob("*/*.parquet"))
              for l in pq.read_table(f).column("length").to_pylist()]
   print(f"median {np.median(lengths)/10:.1f}s -> --timeout-seconds {2*np.median(lengths)/10:.0f}")
   PY
   ```

   Use the **same value for every arm and every condition** — the timeout is part of the success
   criterion, so a config evaluated with a different timeout is not comparable.

3. Decide:

   | Outcome | Action |
   |---|---|
   | success ≥ **90%** | freeze N; go to step 5 |
   | < 90% and improved > 5 pp over the previous batch | collect **20 more** successful episodes (step 1 command; it tops up — raise `n_episodes` / pose count accordingly), re-run step 2's prepare, retrain, re-evaluate |
   | ≤ 5 pp improvement over **two consecutive** batches | plateau — freeze N at the current count |
   | N reaches **100** episodes | hard cap — freeze N regardless |

   After any new collection: re-run `prepare_datasets_bottle` (full, not `--stats-only` — the
   subsets and all stats change) and regenerate all configs (step 6).

## 5. Design-A pretraining — per level, per passing modality

One checkpoint **per data level**, trained on **that level's** dataset. Never reuse the 100%
checkpoint at another level: its weights and its recorded normalization stats both embody episodes
the smaller level must not see.

Audio (if it passed the filter):

```bash
for level in 100 75 50 25; do
  python -m robot_imitation_glue.ur5station.bottle.train_ast_bottle \
      --dataset-root datasets/bottle_experiment/prepared/bottle_9d_$level \
      --output outputs/pretrain/bottle_audio_$level.pt
done
```

Image (if it passed the filter — the screening script doubles as the trainer):

```bash
for level in 100 75 50 25; do
  python -m robot_imitation_glue.ur5station.bottle.screen_instrumentation \
      --dataset-root datasets/bottle_experiment/prepared/bottle_9d_$level \
      --input image --save-encoder outputs/pretrain/bottle_rgb_$level.pt
done
```

Each checkpoint records the dataset it was trained on plus (for audio) the normalization stats and
`time_dimension`; the config generator verifies all of it and refuses a level mismatch.

## 6. Generate the training configs

Before pretraining exists (or whenever the datasets change):

```bash
python -m robot_imitation_glue.ur5station.bottle.generate_configs --arms from_scratch,generic,design_c
```

Once step 5's checkpoints exist — stating explicitly which modalities passed the filter (this is
the pre-registered call from step 3, e.g. image and audio pass, FT does not):

```bash
python -m robot_imitation_glue.ur5station.bottle.generate_configs \
    --arms design_a --design-a-modalities image,audio --pretrain-dir outputs/pretrain
```

The generator reads each level's `audio_stats.json` (random-init arms) and each level's checkpoint
metadata (design_a), asserts arm parity at every level, and refuses stale or cross-level inputs.
**Never hand-edit a generated JSON** — change the generator and re-run.

## 7. Train the 16 policies

```bash
for f in robot_imitation_glue/ur5station/bottle/configs/bottle_*.json; do
  lerobot-train --config_path=$f
done
```

100K steps each, final checkpoint only (no selection). Sanity checks on the first run of each arm:
the logged `global cond dim` is identical across arms; design_c's `final_conv` is 12-wide, the
others 9; design_a logs `initialized N encoder(s) from ...` for exactly the passing modalities.

## 8. Rollouts

20 per config — **10 ID + 10 OOD (unseen stickers)** — 16 configs = 320 rollouts. One command per
(checkpoint, condition); put the ID stickers on, run `--condition id`, swap to the OOD set, run
`--condition ood`:

```bash
for arm in from_scratch generic design_a design_c; do for level in 100 75 50 25; do
  python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
      --checkpoint outputs/train/bottle/${arm}_${level}/checkpoints/100000/pretrained_model \
      --condition id --n-rollouts 10 --timeout-seconds 65
done; done
# swap stickers to the held-out appearance set, then the same loop with --condition ood
```

The same `--timeout-seconds` for **every** config and condition (see step 4 for the derivation —
65 s at the current N; re-derive if N changed). The script uses the same eval path for every arm
(`n_env_action_dims=9`, a no-op except for design_c), draws poses from the test split's seeded
distribution (only appearance distinguishes ID from OOD — never the poses), scores success as all
three cap channels sustained above threshold for 0.5 s, records every rollout — failures included —
to `datasets/bottle_experiment/eval/`, and appends one row per rollout to
`outputs/eval/bottle_results.json`. It resumes: re-running the same command tops up to
`--n-rollouts`. Safety: a rollout aborts (scored as failure) if any drift-corrected force axis
exceeds 40 N.

## 9. Decide the mechanism

Fill the protocol's Table 2 (design A vs design C at every level). The winner carries forward to
the remaining tasks; the loser is reported as the controlled replication. Pooled statistics across
tasks per §1.8 — per-task n=10 is directional only.

---

## Common failure lookup

| Symptom | Cause / fix |
|---|---|
| `RepositoryNotFoundError ... datasets/None` on load | episode-count drift from a hard crash — `docs/repairing_broken_lerobot_datasets.md` |
| `Could not push packet to decoder` in a DataLoader worker | torchcodec vs the AV1 spectrogram stream — audio paths use `video_backend="pyav"` (already handled in the bottle scripts) |
| `audio_stats.json not found` from generate_configs | run step 2's `--stats-only` |
| `checkpoint ... does not look like level N` | a pretraining checkpoint from another level — retrain per level (step 5) |
| every collected episode `success=False` | cap-sensor feed dead or thresholds stale — check `bottle_ble_reader.py`, re-derive `PER_CHANNEL_THRESHOLDS` + `CALIBRATED_RANGE` together |
| collection loops forever on one pose | pose can't succeed — retry cap is unlimited by design; blacklist it **only** if it is a collection-feasibility problem, and write down why |
