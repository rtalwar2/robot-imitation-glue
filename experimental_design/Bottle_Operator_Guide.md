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

Two arms, four data levels:

| Arm | Role | Encoder init | Dataset | Denoised vector |
|---|---|---|---|---|
| `generic` | **control** | ImageNet + AudioSet | `bottle_9d_{level}` | action(9) |
| `generic_c` | **treatment** | ImageNet + AudioSet | `bottle_12d_{level}` | action(9) ⊕ instrumentation(3) |

The `from_scratch` init-verification cell was **dropped 2026-10-14** (not run; config deleted).

`generic` vs `generic_c` is the whole experiment: identical architecture, identical initialization,
identical hyperparameters — the only difference is the three auxiliary sensor channels riding in the
denoised vector. The instrumentation is never a policy *input*; the extra channels are sliced off at
inference, so the deployed policy needs no sensor hardware.

**Design A (instrumentation-pretrained encoders) and the old random-init design-C arm are retired**
— the guide's former step 5 (per-level design-A pretraining via `train_ast_bottle.py`) is gone; that
script stays on disk for the button-experiment lineage, but nothing here runs it.

> **Status 2026-10-20: steps 0–7 are done for the pilot.** 90 expert + 20 recovery episodes collected,
> **N = 70 = 50 expert + 20 recovery** (levels 70/52/35/18), all 8 curve models trained at 150K steps,
> 160 ID rollouts done, OOD sticker done at 100% and half done at 50% (`generic_c_50` remaining).
> Results: `Bottle_Pilot_Results.md` and §9 below. The steps are kept as the procedure for the next
> tasks and for any re-run.

---

## 0. Pre-flight (once per rig change, not per session)

- [ ] **UR payload compensation** on the left arm: with nothing in contact, jog through the task's
  orientations and watch `ft` in rerun. Near-constant → OK. Swinging by newtons → fix the payload
  mass/CoG in the UR controller before collecting; no baseline subtraction can rescue it.
- [x] ~~**Recalibrate the sensor constants**~~ **Done, last on 2026-10-09** for the 6-leg retuned
  motion. `PER_CHANNEL_THRESHOLDS` is **3.20/3.12/3.10 V** and `CALIBRATED_RANGE` (1st/99th
  percentiles) was re-derived with it, from the fully-closed "test"-split runs
  `sensor_logs/run_0024, 0026–0028, 0030, 0032–0038` recorded with
  `open_bottle_demo_analyze_sensors.py` (it drives ur_right through the split's seeded poses and
  saves one run per pose). Only S1 + S2 decide "open" (`OPEN_CHANNELS`); S0 has no usable gap and is
  logged only. `SENSOR_CHECKPOINTS` = `leg_3_end`→S1, `leg_7_end`→S2. The collector's success gate is
  resolved by position, so a future re-map cannot silently kill the label. **To redo after any future
  remount/re-tune:** re-record the batch, re-run `derive_thresholds.py`, move both constants together
  (`bottle_sensor.py` + `train_ast_bottle.py`) — every other runner imports them.
- [ ] **Stickers**: cap sensor voltages unchanged with stickers on (read a fully-open and
  fully-closed cap per appearance, against the thresholds), and the sticker is visible in the
  wrist frame. The pilot's OOD sticker sits on the lid and the sticker runs went ahead (results doc
  §4.5); no voltage re-read is recorded — log one before reporting the OOD numbers.
- [ ] **Sensor feeds running**: `bottle_ble_reader.py` (cap sensor → DDS topic `Bottle`) and the
  Kaldi spectrogram publisher (`KaldiSpectrogram`). The collection loop hard-fails on a dead cap
  sensor, but only at the first checkpoint — start them first.
- [ ] Wrist RealSense reachable (the env raises if 720p won't start — no silent fallback).

## 1. Collect demonstrations (pilot: done — 90 expert 2026-10-09/10, 20 recovery 2026-10-13/14)

Split is selected by the dataset name in
`robot_imitation_glue/ur5station/bottle/collect_data_bottle_opening.py` (`dataset_name = "bottle_opening_train"`;
`"...val"` / anything else selects the val/test pose seeds and counts: 100/25/5 poses). Pose
blacklist per split in the same file — **feasibility exclusions only, never difficulty**.

```bash
python -m robot_imitation_glue.ur5station.bottle.collect_data_bottle_opening
```

The recovery episodes (induced too-deep contact, then a recorded lift-off + scripted opening) come
from a separate collector into `.../expert/bottle_opening_train_recovery`:

```bash
python -m robot_imitation_glue.ur5station.bottle.collect_recovery_bottle
```

The loop stops at the target count on its own, retries failed poses, and resumes an interrupted
dataset (re-run the same command to top up). During touch-point verification: click = correct,
Enter = accept, `t` = retake a blurry frame, `q` = abort.

**If a run crashes hard** (killed mid-save, episode-count drift, `RepositoryNotFoundError` on
load): read `docs/repairing_broken_lerobot_datasets.md` *top to bottom* before touching files. The
one-off scripts from the previous incident are in
`robot_imitation_glue/ur5station/bottle/repair/` as worked examples.

## 2. Prepare datasets

The pilot's curve datasets (N = 70 = the first 50 expert episodes of the seeded order + all 20
recovery episodes, relative actions):

```bash
python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle \
    --pool-count 50 \
    --extra-raw-root bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train_recovery \
    --relative-actions
```

Builds the 8 prepared datasets ({9,12}-dim action × {100,75,50,25}% = 70/52/35/18 episodes, manifest
`prepared/episode_order_pool50.json`) from the raw roots, filters to successful episodes, applies the
FT drift correction (`--ft-orientation-correction` is available but was **not** used for the curve
datasets — results doc §5.2), and finishes by writing per-level
`audio_stats.json` into every prepared root. If the datasets already exist and only the stats are
missing (or the raw data changed is *not* the case), retrofit without re-encoding video:

```bash
python -m robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle --stats-only
```

Check: each `datasets/bottle_experiment/prepared/bottle_9d_*/audio_stats.json` exists, and its
`episodes` list matches the manifest's level. **Both arms normalize audio with these per-level
stats** (since 2026-10-15 — AudioSet's constants made every earlier checkpoint ignore audio; protocol
§1.7), so every level's stats file must exist.

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

Status: run once, on the first 51-episode set (pre-recalibration) — proprio 0.182, image 0.787,
audio 0.606, FT 0.096 (see the preliminary table in `Experiment_Protocol.md`). **Not yet re-run on
the final N = 70 data** — do that, with more seeds, before reporting. The linear probe on the
trained encoders (results doc §4.3b) is the more informative check for the pilot.

## 4. Baseline check — is there enough data? (protocol §1.6)

> **Done 2026-10-14: N = 70 = 50 expert + 20 recovery episodes.** Expert-only models with delta
> actions plateaued (50 ep 60%, 90 ep 30%: too-deep first contact, no recovery). Adding the 20
> recovery episodes and switching to relative actions at 150K steps gave **19/20 = 95%** for both
> 50+20 and 90+20, so the smallest level, 50+20 = 70, was frozen. The timeout stays **65 s**.
> Full log in the protocol, Part 2, Task 1, "N-finding log". (These models predate the 2026-10-15
> audio fix.)

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
       --checkpoint outputs/train/bottle/generic_100/checkpoints/150000/pretrained_model \
       --condition id --n-rollouts 20 --timeout-seconds 65 --num-inference-steps 10 --n-action-steps 8
   ```

   **The rollout timeout is 2× the median demonstration duration, rounded up to the
   next 5 s** (protocol §1.8). The pilot fixed it at **65 s** on 2026-10-14 and used it for every
   curve rollout. For a new task, derive it on that task's 100% data, then pass the value explicitly
   everywhere:

   ```bash
   python - <<'PY'
   import math
   import numpy as np, pyarrow.parquet as pq
   from pathlib import Path
   lengths = [x for f in sorted(Path("datasets/bottle_experiment/prepared/bottle_9d_100/meta/episodes").glob("*/*.parquet"))
              for x in pq.read_table(f).column("length").to_pylist()]
   median_s = np.median(lengths) / 10
   print(f"median {median_s:.1f}s -> --timeout-seconds {5 * math.ceil(2 * median_s / 5)}")
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

Flags: `--arms` (comma list from `generic,generic_c`; `from_scratch` still exists in the generator
but its cell was dropped 2026-10-14), `--levels` (comma list from `100,75,50,25`), `--steps`
(default 150000), `--output-dir`. The generator reads each level's `audio_stats.json`, asserts arm
parity at every level —
`arms differ only in 11 intended keys` should print — and refuses to run on missing stats. **Never
hand-edit a generated JSON** — change the generator and re-run.

## 6. Train the 8 curve policies (pilot: done, on GPULab)

```bash
for arm in generic generic_c; do for level in 100 75 50 25; do
  lerobot-train --config_path=robot_imitation_glue/ur5station/bottle/configs/bottle_${arm}_${level}.json
done; done
```

The pilot trained on GPULab (see the `gpulab` skill / `run_training_queue.sh`); checkpoints are under
`outputs/train/bottle/<arm>_<level>/checkpoints/150000/pretrained_model` and on HF
(`ramen-noodels/bottle_<arm>_<level>`). `bottle_generic_fusion_100.json` is the optional ManiWAV-style
fusion ablation (trained, not robot-evaluated).

150K steps each, final checkpoint only (no selection). Sanity checks on the first run of each arm:
the logged `global cond dim` is identical across arms; `generic_c`'s `final_conv` is 12-wide,
`generic`'s is 9; both arms must show the ImageNet and AudioSet load lines.

## 7. Rollouts

**20 ID + 20 OOD (sticker) per curve config** — 8 curve configs × 40 = **320 for the pilot**. One
command per (checkpoint, condition):

```bash
for arm in generic generic_c; do for level in 100 75 50 25; do
  python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
      --checkpoint outputs/train/bottle/${arm}_${level}/checkpoints/150000/pretrained_model \
      --condition id --n-rollouts 20 --timeout-seconds 65 --num-inference-steps 10 --n-action-steps 8
done; done
# sticker on the lid, then per checkpoint (alternate the two arms of a level within one session):
python -m robot_imitation_glue.ur5station.bottle.eval_bottle \
    --checkpoint outputs/train/bottle/generic_c_50/checkpoints/150000/pretrained_model \
    --condition ood --tag sticker --n-rollouts 20 --timeout-seconds 65 --num-inference-steps 10 --n-action-steps 8
```

Results rows are keyed by config name, e.g. `generic_50_ddim10_a8` (ID) and
`generic_50_ddim10_a8_sticker` (OOD). Use the **same `--timeout-seconds`, `--num-inference-steps`
and `--n-action-steps` for every config and condition**. Control the lighting (blinds, fixed lamps)
and check the logged `hover_brightness` stays in the training range (0.33–0.62) — the pilot's ID
sessions were confounded with daylight (results doc §5.3).

**Pilot status 2026-10-20:** ID done for all 8 (160 rollouts). Sticker: `generic_100` 17/20,
`generic_c_100` 14/20, `generic_50` 13/20; **`generic_c_50` still to run**. Levels 75 and 25 have
no sticker rollouts.

**What the operator does per rollout.** The script gates three stages with an Enter press — move
the left arm home → move the right arm to the sampled pose → **hand control to the policy** (the
hover move above the cap runs on its own between the second and third gates) — plus one
"close the bottle by hand" press between rollouts. There is **no sticker-confirmation prompt
anymore** (removed 2026-10): the condition is whatever
`--condition` declares, so swapping the physical stickers between the ID and OOD loops is on the
operator — a mislabeled batch is an operating error, not something the script catches.

**Enter stops a running rollout.** Mid-policy, pressing Enter ends the episode immediately and asks
for a visual verdict:

```
[rollout] did the cap VISUALLY open? (y = success, anything else = failure):
```

Answer `y` when the cap is mostly open even though not all three sensor channels cleared — the
operator's answer becomes the episode's `success` label. It reaches the eval dataset as the
recorded episode's `next.success`; the provenance lives in `outputs/eval/bottle_results.json`,
where the row carries `success_source: "operator"` (sensor-determined rows carry
`"sensors"` — the dataset itself has no source field, so audit provenance through the results
JSON). The verdict path never overrides a sensor-determined outcome. Rows from before the change
lack the field — treat absent as `"sensors"`.

Under the hood, unchanged: the same eval path for every arm (`n_env_action_dims=9`, a no-op except
for `generic_c`), poses drawn from the test split's seeded distribution (only appearance
distinguishes ID from OOD — never the poses), sensor success = the deciding channels S1 + S2 (`OPEN_CHANNELS`) sustained at
or above threshold for 5 consecutive steps (0.5 s; `is_cap_open` compares with `>=`), every
rollout recorded — failures included — to
`datasets/bottle_experiment/eval/eval_bottle_{arm}_{level}_{condition}/`, one row appended per
rollout to `outputs/eval/bottle_results.json`. It resumes: re-running the same command tops up to
`--n-rollouts`. Safety: a rollout aborts (scored as failure) if any drift-corrected force axis
exceeds `MAX_ABS_FORCE_NEWTONS` = **96 N** (since 2026-10-16: twice the p95 per-episode peak of the
110 expert + recovery training episodes, 48.0 N; it was 52 N before -- see the protocol §1.6).

## 8. Analysis

(The init-verification step that used to come first here was dropped with the `from_scratch` cell,
2026-10-14.) The analysis the pilot exists for (§1.9), plotted with `ur5station/bottle/plot_curves.py`: **Figure A** — success rate vs. % data, ID only,
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
| `AssertionError: arms differ in unintended keys` from generate_configs | a change to the base config leaked into an arm-specific branch, or a new key was added on one side only — fix `apply_arm`; do not paper over it by extending `INTENDED_ARM_DIFFERENCES` unless the key is genuinely an intended arm difference |
| every collected episode `success=False` | cap-sensor feed dead or thresholds stale — check `bottle_ble_reader.py`, re-derive `PER_CHANNEL_THRESHOLDS` (+ `CALIBRATED_RANGE`) per step 0 |
| rollout ends as `manual_stop` you did not cause | a stray newline in the terminal buffer — the script flushes pending input before each rollout, but a second process writing to the tty would defeat it. A known residual gap: `input()` reads through Python's buffer, so a **double Enter at a stage gate** can leave the second newline unreachable to the kernel-level flush, and the verdict prompt would then auto-answer (empty answer = failure, so the mislabel is fail-safe but still wrong — answer deliberately after reading the prompt) |
| collection loops forever on one pose | pose can't succeed — retry cap is unlimited by design; blacklist it **only** if it is a collection-feasibility problem, and write down why |





generic 25 during opening motion loses track of bottle ood, random movements

generic 25 c almost complete opening motion them it goes to finishhover pose thoinking cap is fully open while it is almost open


  for M in generic_100 generic_c_100 generic_75 generic_c_75 generic_50 generic_c_50 generic_25 generic_c_25 generic_fusion_100; do

python -m robot_imitation_glue.ur5station.bottle.eval_bottle  --checkpoint outputs/train/bottle/$M/checkpoints/150000/pretrained_model --condition id --n-rollouts 20 --timeout-seconds 65    --num-inference-steps 10 --n-action-steps 8

---

## 9. Results summary (2026-10-19): why didn't instrumentation help?

**Setup.** Two arms with the same architecture, pretrained initialization (ImageNet image encoder, AudioSet audio encoder) and training budget (150K steps):
- `generic`: the control, no sensor supervision.
- `generic_c`: also predicts the 3-channel cap sensor alongside the action.

Four data levels (18 / 35 / 52 / 70 episodes), 20 in-distribution rollouts per model, the same 20 poses in the same order for both arms.

| Episodes | `generic` | `generic_c` |
|---|---|---|
| 18 (25%) | 6/20 = 30% | 5/20 = 25% |
| 35 (50%) | 13/20 = 65% | 12/20 = 60% |
| 52 (75%) | 17/20 = 85% | 15/20 = 75% |
| 70 (100%) | 18/20 = 90% | 18/20 = 90% |

Pairing the 80 rollouts by pose, instrumented minus control is **−5 percentage points, 95% CI −16 to +6**. Cochran–Mantel–Haenszel test over the levels: p = 0.57. Figures: `outputs/figures/bottle_curve_id.png`, `bottle_outcomes_id.png` (`plot_curves.py`).

**Q1. Was there an effect we were just too underpowered to see?**
**A.** Partly. With 20 rollouts per cell, differences under about 25–30 percentage points at a single level are invisible. Pooled over the levels, though, an average benefit above about 6 points is unlikely, and the point estimate slightly favours the control. So the data can't rule out a small effect, but it argues against a large one.

**Q2. Did the instrumented policy actually learn to predict the sensor?**
**A.** Yes, easily. On episodes it never saw, `generic_c` predicts the three channels with R² 0.91 / 0.95 / 0.70 and classifies cap open vs closed **97–98% correctly, already with 18 episodes**. On its training episodes it is perfect (R² 1.00). The auxiliary task was learned; the question is whether it changed anything.

**Q3. Was the sensor data too noisy to learn from?**
**A.** No.
- Noise is about 0.003 V, against closed→open steps of 0.17 / 0.61 / 1.11 V (signal-to-noise 60–385), with no flicker and 4% repeated readings.
- Minor issues: 7 of the 90 expert episodes start with channel 3 already reading "open" (episodes 11, 14, 55–58 and 73: wrong labels), S0 barely moves (0.17 V), and only about 26% of frames are mid-transition.
- None of these is large enough to explain a null result.

**Q4. Does the policy without sensor supervision already learn the cap state by itself?** *(the key check)*
**A.** Yes. Linear probes on the frozen encoders, using 36 episodes none of the 8 models trained on:
- An untrained ImageNet ResNet-18 already reaches R² 0.75. After policy training, **`generic`'s image encoder reaches 0.89–0.91 at every data level, including 18 episodes**, without ever being given the sensor.
- `generic_c` minus `generic` is at most ±0.06 R² at any level. At 25%, where the faster-learning hypothesis predicts the largest gain, every interval includes zero (image +0.017 [−0.024, +0.056]).
- The action loss alone already teaches the encoder the cap state, so the auxiliary signal has nothing extra to teach on this task.

**Q5. Is the sensor redundant with what the camera sees?**
**A.** Largely. In screening, the wrist image predicts the sensor with R² 0.79, against 0.18 for proprioception. This alone doesn't disqualify a task: an auxiliary target has to be learnable from the inputs to help at all. What matters is Q4, whether the control learns it anyway. Here it does.

**Q6. Do the demonstrations even depend on the cap state?**
**A.** Only partly.
- The motion is scripted.
- Sensor-gated retries show up in about 5 of the 90 expert episodes; recovery behaviour in 17 of the 20 recovery episodes.
- Stop-on-open doesn't count, because the eval also ends the rollout when the sensor reads open.
- The reactive parts start from a visibly jammed state, so the camera already shows what triggers them.

**Q7. Do the policies fail because they don't know the cap state?**
**A.** No. The failures are in execution, not perception.
- **First contact is fine:** contact depth is the same in failures and successes (124.0 vs 124.1 mm below hover, p = 0.55), close to the demos' 126 mm.
- **The lid starts opening, then the gripper loses it (31 of 56 failures):** S1 opens and S2 rises to 2.97 V against a 3.10 V threshold. The arm keeps moving, but with lower contact force (3.7 vs 5.4 N): the gripper loses the lid edge.
- **No lid movement at all: 13 failures**, 9 of them force aborts.
- Predicting the cap state doesn't teach recovering a lost edge.

**Q8. Was the audio being used at all?**
**A.** Not before the fix; it is now.
- A normalization bug plus coupled weight decay had pushed the audio encoder weights to about zero in all earlier models. It was fixed with per-level audio statistics and AdamW before the curve runs.
- In the offline sensitivity test, swapping the audio now moves the actions 4–9× the run-to-run noise. The fusion variant (`generic_fusion_100`, trained but not yet evaluated on the robot) uses audio about 1.5× more.
- Audio alone carries little cap-state information (probe R² 0.23–0.37). The images dominate.

**Q9. Were the evaluation conditions fair?**
**A.** Not entirely.
- **Lighting:** the eval camera runs auto-exposure, which can't compensate strong daylight. All `generic` sessions and `generic_c_100` ran under bright light (image brightness 0.53–0.57, 15–25% saturated pixels). `generic_c` 25/50/75 ran under normal light (0.39–0.40, close to the training data's 0.38). This probably handicapped the control, which still matched or beat the instrumented arm.
- **Pose 0:** it fails for all 8 curve models, after succeeding for the 4 earlier ones. The image drives the failure; whether it's the pose or being the first rollout of every session is unresolved.
- **Force abort:** the 96 N threshold was fixed before the curve evaluation, and aborts are similar between arms (6 vs 8).

**Q10. Were any inputs corrupted in a way that could hide an effect?**
**A.** Yes, but equally for both arms.
- The 90 expert demos were collected without re-zeroing the force sensor, so their free-air force reads about 7 N (up to 17 N) and varies with the arm's pose.
- The policies essentially ignore the state input anyway (swapping it moves the actions by 0.3–0.4× the run-to-run noise).
- The collector is fixed (2026-10-19), and `prepare_datasets_bottle --ft-orientation-correction` repairs the existing data.

**Q11. Did instrumentation change behaviour in any way?**
**A.** Slightly, and not in a helpful direction. Successful `generic_c` rollouts keep pushing longer: force × contact time 145 vs 101 N·s (p < 0.001), contact 6.8 vs 6.2 s. Peak force doesn't differ. Both arms push 2–3× harder downward than the demos.

### Conclusion

On the bottle task, supervising the cap sensor did not make learning more data-efficient. Success rates are statistically indistinguishable at every data level, and an average benefit above about 6 percentage points is unlikely. The mechanism-level checks explain why.
- **There was nothing extra to teach.** The un-instrumented policy's encoder learns the cap state from the action loss alone, already with 18 episodes (probe R² ≈ 0.9). The instrumented encoder represents it no better at any level.
- **The failures aren't about the cap state.** They are execution errors, mostly losing the lid edge mid-opening, which knowing the cap state doesn't fix.

The result is still preliminary:
- one task, one training seed per configuration;
- in-distribution rollouts only, with OOD not run;
- a lighting confound between the arms;
- a matched-lighting re-run at 50% (40 rollouts) is still pending.

So the bottle pilot is indecisive about a small effect, but it gives a clear diagnostic for the next tasks. Instrumentation can only speed up learning when the un-instrumented policy does *not* already learn the instrumented quantity at low data. That can be checked cheaply before evaluating anything on the robot: train a low-data control and probe its encoder. Plug insertion, where seating is largely hidden from the camera, is the natural next test.
