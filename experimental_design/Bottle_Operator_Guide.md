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

## The design as of `ral_simplified` (2 October 2026)

Two live arms plus one verification cell, four data levels:

| Arm | Role | Encoder init | Dataset | Denoised vector |
|---|---|---|---|---|
| `generic` | **control** | ImageNet + AudioSet | `bottle_9d_{level}` | action(9) |
| `generic_c` | **treatment** | ImageNet + AudioSet | `bottle_12d_{level}` | action(9) ⊕ instrumentation(3) |
| `from_scratch` | init verification, **100% level only** | random | `bottle_9d_100` | action(9) |

`generic` vs `generic_c` is the whole experiment: identical architecture, identical initialization,
identical hyperparameters — the only difference is the three auxiliary sensor channels riding in the
denoised vector. The instrumentation is never a policy *input*; the extra channels are sliced off at
inference, so the deployed policy needs no sensor hardware.

**Design A (instrumentation-pretrained encoders) and the old random-init design-C arm are retired**
— the guide's former step 5 (per-level design-A pretraining via `train_ast_bottle.py`) is gone; that
script stays on disk for the button-experiment lineage, but nothing here runs it.

> ⚠️ **Rebaseline in progress.** The demonstration set is being recollected and the cap-sensor
> covered/uncovered thresholds recalibrated. Every number below that was derived from the old data —
> **N=51, median duration 31.1 s, timeout 65 s, the screening scores** — is **provisional until
> re-derived**. Steps 0–2 are the gate: never train curve policies on pre-recalibration datasets.

---

## 0. Pre-flight (once per rig change, not per session)

- [ ] **UR payload compensation** on the left arm: with nothing in contact, jog through the task's
  orientations and watch `ft` in rerun. Near-constant → OK. Swinging by newtons → fix the payload
  mass/CoG in the UR controller before collecting; no baseline subtraction can rescue it.
- [ ] **Recalibrate the sensor constants** (`hardware/bottle_sensor.py`, `PER_CHANNEL_THRESHOLDS` —
  currently 3.17/3.18/3.00 V). The live values derive from
  `bottle_experiment/sensor_logs/run_{0003,0005,0006}.json`, recorded under the *previous* sensor
  layout. Take fresh covered/uncovered readings for the current cap and mounting, re-derive
  `PER_CHANNEL_THRESHOLDS` **and** `CALIBRATED_RANGE` from the same new run set, and confirm the
  collection checkpoints (`SENSOR_CHECKPOINTS`) still line up with the motion. Everything is
  labelled and scored against these constants — this is the gate for the recollection in step 1.
- [ ] **Stickers**: cap sensor voltages unchanged with stickers on (read a fully-open and
  fully-closed cap per appearance), and the sticker is visible in the wrist frame. Re-confirm after
  the recalibration above, not before.
- [ ] **Sensor feeds running**: `bottle_ble_reader.py` (cap sensor → DDS topic `Bottle`) and the
  Kaldi spectrogram publisher (`KaldiSpectrogram`). The collection loop hard-fails on a dead cap
  sensor, but only at the first checkpoint — start them first.
- [ ] Wrist RealSense reachable (the env raises if 720p won't start — no silent fallback).

## 1. Collect demonstrations (currently: recollect under the recalibrated constants)

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
`episodes` list matches `episode_order.json`'s level. The per-level stats feed only the
`from_scratch` verification cell (both curve arms use AudioSet's dataset-independent constants),
but the generator builds the reference arm at every level, so every level's stats file must exist.

## 3. Screening (privilege gate; modality suitability is diagnostic-only since 2026-10)

With design A retired, nothing in the design branches on the per-modality scores — the privilege
gate (§1.4 step 1) is the only screening result that can still stop a task, and it is re-checked
per task rather than inherited from the pilot. Run the tests anyway: they are cheap, offline, and
the protocol reports them. **Fix the suitability threshold before looking at any score.** Run all
four on the recollected 100% dataset:

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
| **proprio** R² near-perfect (at the modality scores' level) | the signal is not privileged — the task is unsuitable; stop and rethink (this gate remains mandatory, §1.4) |
| a modality's R² clears the pre-registered margin over proprio | diagnostic: the instrumentation is perceivable through that modality — report it |
| below the margin | diagnostic: report the score; no design decision hangs on it anymore |

Status: already run once on the first 51 episodes — proprio 0.182, image 0.787, audio 0.606,
FT 0.096 (see the preliminary table in `Experiment_Protocol.md`). **Provisional twice over**: one
run each, on the pre-recollection dataset. Re-run with more seeds on the new data before reporting.

## 4. Baseline check — is there enough data? (protocol §1.6)

The iterative N-finding loop. Everything trains on the **100% level of what exists so far**. It
runs on the **control** (`generic`) — since 2026-10 it is the pretrained-init arm that every curve
point shares; §1.6 explains why the control is the right arm for the gate.

1. Generate the control's 100% config and train it:

   ```bash
   python -m robot_imitation_glue.ur5station.bottle.generate_configs --arms generic --levels 100
   lerobot-train --config_path=robot_imitation_glue/ur5station/bottle/configs/bottle_generic_100.json
   ```

2. Evaluate with **20 real rollouts** on training-distribution conditions (ID stickers):

   ```bash
   python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
       --checkpoint outputs/train/bottle/generic_100/checkpoints/100000/pretrained_model \
       --condition id --n-rollouts 20 --timeout-seconds 65
   ```

   **The rollout timeout is 2× the median demonstration duration** (protocol §1.8). The current
   65 s is **provisional** — derived from the old 51 episodes (median 311 frames @ 10 Hz = 31.1 s).
   **Re-derive it on the recollected data** (and again whenever N changes), then pass the new value
   explicitly everywhere:

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
   subsets and all stats change) and regenerate all configs (step 5).

## 5. Generate the training configs

The eight curve configs (2 arms × 4 levels) — this is the default:

```bash
python -m robot_imitation_glue.ur5station.bottle.generate_configs
```

The verification cell, deliberately narrow (100% level only, per the pre-registered rule in §1.3):

```bash
python -m robot_imitation_glue.ur5station.bottle.generate_configs --arms from_scratch --levels 100
```

Flags: `--arms` (comma list from `generic,generic_c,from_scratch`), `--levels` (comma list from
`100,75,50,25`), `--steps`, `--output-dir`. The generator reads each level's `audio_stats.json`
(for the random-init reference it builds at every level), asserts arm parity at every level —
`arms differ only in 11 intended keys` should print — and refuses to run on missing stats. **Never
hand-edit a generated JSON** — change the generator and re-run.

## 6. Train the 8 curve policies (+1 verification run)

```bash
for f in robot_imitation_glue/ur5station/bottle/configs/bottle_generic*.json; do
  lerobot-train --config_path=$f
done
# the verification cell, once the recollection reaches its scheduled slot:
lerobot-train --config_path=robot_imitation_glue/ur5station/bottle/configs/bottle_from_scratch_100.json
```

100K steps each, final checkpoint only (no selection). Sanity checks on the first run of each arm:
the logged `global cond dim` is identical across arms; `generic_c`'s `final_conv` is 12-wide,
`generic`'s and `from_scratch`'s are 9; the `from_scratch` run logs the random-init path (no
ImageNet/AudioSet load lines) — the generic arms must show both pretrained loads.

## 7. Rollouts

**20 ID + 20 OOD (unseen stickers) per curve config** — 8 curve configs × 40 = 320; plus the
verification cell's 20 ID-only rollouts = **340 for the pilot** (360 if its optional OOD extension
is run — it can never gate anything). One command per (checkpoint, condition); put the ID stickers
on, run `--condition id`, swap to the OOD set, run `--condition ood`:

```bash
for arm in generic generic_c; do for level in 100 75 50 25; do
  python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
      --checkpoint outputs/train/bottle/${arm}_${level}/checkpoints/100000/pretrained_model \
      --condition id --n-rollouts 20 --timeout-seconds 65
done; done
# swap stickers to the held-out appearance set, then the same loop with --condition ood
# verification cell (ID only):
python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
    --checkpoint outputs/train/bottle/from_scratch_100/checkpoints/100000/pretrained_model \
    --condition id --n-rollouts 20 --timeout-seconds 65
```

Use the **same `--timeout-seconds` for every config and condition** (step 4's derivation —
provisionally 65 s until the recollection lands).

**What the operator does per rollout.** The script gates each stage with an Enter press: move the
left arm home → move the right arm to the sampled pose → hover → **hand control to the policy**.
There is **no sticker-confirmation prompt anymore** (removed 2026-10): the condition is whatever
`--condition` declares, so swapping the physical stickers between the ID and OOD loops is on the
operator — a mislabeled batch is an operating error, not something the script catches.

**Enter stops a running rollout.** Mid-policy, pressing Enter ends the episode immediately and asks
for a visual verdict:

```
[rollout] did the cap VISUALLY open? (y = success, anything else = failure):
```

Answer `y` when the cap is mostly open even though not all three sensor channels cleared — the
operator's answer becomes the episode's `success` label, recorded with `success_source: "operator"`
in both the eval dataset and `outputs/eval/bottle_results.json`. Sensor-decided rollouts carry
`success_source: "sensors"`; the verdict path never overrides a sensor-determined outcome. Rows
from before the change lack the field — treat absent as `"sensors"`.

Under the hood, unchanged: the same eval path for every arm (`n_env_action_dims=9`, a no-op except
for `generic_c`), poses drawn from the test split's seeded distribution (only appearance
distinguishes ID from OOD — never the poses), sensor success = all three cap channels sustained
above threshold for 5 consecutive steps (0.5 s), every rollout recorded — failures included — to
`datasets/bottle_experiment/eval/eval_bottle_{arm}_{level}_{condition}/`, one row appended per
rollout to `outputs/eval/bottle_results.json`. It resumes: re-running the same command tops up to
`--n-rollouts`. Safety: a rollout aborts (scored as failure) if any drift-corrected force axis
exceeds `MAX_ABS_FORCE_NEWTONS` = **100 N**.

## 8. Init verification, then the curves

Fill the protocol's **Table 2** with the verification cell's 20 ID rollouts against
`generic`@100% ID and apply the **pre-registered one-sided rule**:

| Result | Decision |
|---|---|
| `from_scratch` ties or beats `generic` at 100% ID | promotes it to a live arm (and reports the finding: AudioSet init misfits contact audio); extend it to the other levels and its optional OOD block |
| loses | the pretrained-init assumption holds; the run goes to the appendix as the init-verification row. Do **not** resurrect it on a better-looking level or condition — the rule is one-sided by pre-registration |

Then the analysis the pilot exists for (§1.9): **Figure A** — success rate vs. % data, ID only,
two curves (`generic`, `generic_c`); **Figure B** — ID vs. OOD per arm per level. Pooled
statistics across tasks per §1.8 are the pre-registered primary — per-task n=20 is directional
only.

---

## Common failure lookup

| Symptom | Cause / fix |
|---|---|
| `RepositoryNotFoundError ... datasets/None` on load | episode-count drift from a hard crash — `docs/repairing_broken_lerobot_datasets.md` |
| `Could not push packet to decoder` in a DataLoader worker | torchcodec vs the AV1 spectrogram stream — audio paths use `video_backend="pyav"` (already handled in the bottle scripts) |
| `audio_stats.json not found` from generate_configs | run step 2's `--stats-only` — the parity reference arm is built at every level, so every level's stats are required |
| `parity check: arms differ in unintended keys` (AssertionError) | a change to the base config leaked into an arm-specific branch, or a new key was added on one side only — fix `apply_arm`; do not paper over it by extending `INTENDED_ARM_DIFFERENCES` unless the key is genuinely an intended arm difference |
| every collected episode `success=False` | cap-sensor feed dead or thresholds stale — check `bottle_ble_reader.py`, re-derive `PER_CHANNEL_THRESHOLDS` (+ `CALIBRATED_RANGE`) per step 0 |
| rollout ends as `manual_stop` you did not cause | a stray newline in the terminal buffer — the script flushes pending input before each rollout, but a second process writing to the tty would defeat it |
| collection loops forever on one pose | pose can't succeed — retry cap is unlimited by design; blacklist it **only** if it is a collection-feasibility problem, and write down why |
