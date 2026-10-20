This file orients AI agents working on the instrumented-imitation-learning experiment. It summarizes; `Experiment_Protocol.md` in this folder is the authoritative document, and where they disagree, the protocol wins.

## What this folder is

| File | Status |
|---|---|
| `Experiment_Protocol.md` | **Authoritative.** Part 1 = shared methodology, Part 2 = per-task checklists, Part 3 = open promotor items, Part 4 = code issues (FIXED = committed) |
| `Bottle_Pilot_Results.md` | **The bottle pilot's complete results record** (setup, data, training, results, bugs, open items) — start here for anything about outcomes |
| `Bottle_Operator_Guide.md` | The runbook: exact commands in order, decision rules, failure lookup |
| `Task_Specifications.md` | Superseded (8 July 2026). Kept as history; do not follow it |
| `session.json` | Transcript of the original design conversation with a local assistant |
| `Promotor_Meeting_Report.pdf`, `Promotor_Meeting_Decison_map.svg` | Materials from the 1 July 2026 promotor meeting |

## The paper (target: RA-L)

**Research question.** Does *task-specific instrumentation* — privileged sensor signals added to the environment, available during training but absent at inference — improve the **data efficiency** of imitation learning?

**Hypothesis.** Supervising a policy with the instrumentation signal is an inductive bias toward task-relevant features: the treatment arm reaches a given success rate with **fewer demonstrations** than a matched control — identical architecture, identical (generic ImageNet/AudioSet) initialization, no instrumentation. Since the `ral_simplified` revision (2026-10) the comparison is single-factor: both curve arms start pretrained. The planned random-init verification cell was dropped on 2026-10-14 (§1.3), so the paper must not claim that pretrained init helps.

**Contribution boundary — important for framing anything written about this work.** A colleague's workshop paper already shows the mechanism helps at 100% data, on one task, one seed — and its C arm differed from its baselines in *two* factors at once (random init AND auxiliary channels). This paper's claim is therefore the **data-efficiency curves** (success rate vs. fraction of demonstrations) and the **ID/OOD generalization split**, on a design that **de-confounds** the workshop comparison — not the mechanism itself. The old A-vs-C mechanism table was retired with `ral_simplified`; nothing "decides the design" anymore, the design is fixed. The proprioception privilege test (§1.4) is a supporting methodological note, not a headline claim; per-modality suitability scores are diagnostic-only since design A's retirement.

## The bottle pilot

One physical bottle with a flick-open cap instrumented by a **3-channel phototransistor sensor** (voltage per channel; a channel "uncovers" as its tab opens). Two UR5s: the right arm presents the bottle at ~100 sampled poses, the left arm opens the cap. Demonstrations are **scripted**, gated by sensor checkpoints (`SENSOR_CHECKPOINTS`: S1 at `leg_3_end`, S2 at `leg_7_end` since 2026-10-09; S0 is logged but decides nothing; invariant 7), with failed checkpoints triggering recorded retries. On top of the expert demos, 20 **recovery episodes** (an induced too-deep contact, then a recorded scripted recovery; `collect_recovery_bottle.py`) are part of the data. The OOD axis is **appearance only** (stickers on the bottle; one cap, so no dynamics variation — the paper must call this appearance robustness, not generalization unqualified). Details: protocol Part 2, Task 1.

Modalities: wrist camera (720p, no fallback), mel spectrogram (AST), and `observation.state` = TCP pose (6) ⊕ drift-corrected internal FT (6). Actions are **UMI-style relative poses** since 2026-10-14: the absolute target `[xyz(3), rot6d(6)]` expressed in the frame of the TCP at the latest observation (`prepare_datasets_bottle --relative-actions`). Tool-frame deltas were used before; §1.7 records the switch and the rejected alternatives.

## The arms (`ral_simplified`: 2 arms × 4 data levels = 8 curve runs)

| Arm | Role | Encoder init | Output | Isolates |
|---|---|---|---|---|
| `generic` | **control** | ImageNet + AudioSet | action(9) | what standard pretrained init alone achieves |
| `generic_c` | **treatment** | ImageNet + AudioSet | action(9) ⊕ sensor(3) = 12-dim denoised vector | instrumentation *as auxiliary prediction* — the paper's single factor |

`design_a` and the random-init `design_c` arm are **retired** (2026-10); `generic_c` keeps design C's mechanism and moves it onto the standard pretrained init, which makes treatment-minus-control a one-factor difference. The `from_scratch` init-verification cell was **dropped 2026-10-14** (not run; its config is deleted). **N = 70** = 50 expert episodes + 20 recovery episodes (fixed 2026-10-14, protocol §1.6 log), pooled and shuffled once (`EPISODE_ORDER_SEED` 20260812), then cut into nested levels 100/75/50/25% = **70/52/35/18 episodes** (manifest `prepared/episode_order_pool50.json`). Fixed **150K steps** (100K until 2026-10-14), **final checkpoint, no selection**. Evaluation: 40 real rollouts per curve config (20 ID + 20 OOD sticker) = 320 pilot rollouts, at 10 DDIM steps / 8 action steps, 65 s timeout, 96 N force abort; statistics pooled across tasks (CMH, stratified by task) because n=20 per cell cannot carry the claim alone. `generic_c`'s 3 extra channels take 3/12 of the ε-prediction loss automatically (no λ) and are sliced off at inference — the sensor is never a policy input, so the deployed policy needs no sensor hardware ("no reliance").

## Invariants — do not break these

1. **Arms differ only in intended keys.** `generate_configs.py` asserts this at build time (`INTENDED_ARM_DIFFERENCES`). Never hand-edit a generated config; change the generator.
2. **Input normalization:** images get ImageNet stats in the generic arms. Audio gets **per-level dataset stats in every arm** (since 2026-10-15).
   - Our spectrogram is `(fbank + 20) / 60` of an ADC-unit waveform, so AudioSet's constants do not fit it.
   - Using them, raw or rescaled, made every pre-2026-10-15 checkpoint ignore audio (protocol §1.7).
   - `audio_norm_mean/std` default to 0/1 = *no normalization*, so they must always be set explicitly.
   - Optimizer: AdamW with weight decay 1e-6, never lerobot's coupled `adam` preset.
2b. **Audio stats are per LEVEL, never global.** Configs use that level's `audio_stats.json`, computed over exactly the episodes the level trains on.
   - Reusing the 100% run's stats at 25% leaks episodes the run never sees: invisible to the arm comparison, poison to the data-efficiency claim.
   - Both arms of a level share the same stats.
2c. **The instrumentation is never a policy input.** Sensor values live only as the 3 auxiliary channels of `generic_c`'s 12-dim *action* feature (added at dataset-prep time), and `LerobotAgent(n_env_action_dims=9)` slices them off at inference. "Available during training, absent at inference" is the whole no-reliance claim — anything that puts a cap-sensor reading on the observation side breaks the design, not just a metric. (The rule binds only the *instrumentation*: fusing the two deploy-available observation streams, image × audio, is legal by construction and was still declined as a design-time change — reasoning in protocol §1.2; don't propose it mid-campaign.)
3. **`observation.state` is the only state key the policy reads** — anything the policy should see as proprioception must be concatenated into it. Other `observation.*` vectors are typed but never reach the conditioning.
4. **`bottle_sensor` cannot sneak in through the observation side**: lerobot's `batch_to_transition` drops every key that is not `observation.*`/`action`/bookkeeping. That is the mechanical enforcement of 2c — the only route the signal has into training is *inside* `action` (done at dataset-prep time), which is exactly where `generic_c`'s treatment lives.
5. **Data-level subsets are separate prepared datasets**, never `dataset.episodes` lists — that field is broken for non-prefix lists (absolute vs. relative frame indices). Every prepared dataset is built from the raw root, one video generation each.
6. **The pose blacklist is for collection feasibility only, never task difficulty** — excluding hard poses inflates the success rate, on the test split directly. See the comment at its definition.
7. **Sensor calibration constants travel together**: `PER_CHANNEL_THRESHOLDS` (`hardware/bottle_sensor.py`, the single source — the collector, eval and the calibration recorder all import it) and `CALIBRATED_RANGE` (defined in `train_ast_bottle.py`, imported by `screen_instrumentation.py`). **Last re-derived 2026-10-09** for the 6-leg retuned motion, from the fully-closed "test"-split runs `run_0024, 0026–0028, 0030, 0032–0038`: `PER_CHANNEL_THRESHOLDS` = **3.20 / 3.12 / 3.10 V**, `CALIBRATED_RANGE` as 1st/99th percentiles. `OPEN_CHANNELS` = S1 + S2 decide "open" (success, stop-on-open); S0 has no usable closed/open gap and is logged only. `SENSOR_CHECKPOINTS` = `leg_3_end`→S1, `leg_7_end`→S2 (the comment in `bottle_sensor.py` is the provenance). The 2026-10-07 values (3.16/3.12/3.08, three gates) are superseded. Whenever the cap, mounting or motion geometry changes, re-record the calibration batch and re-derive **both** together — the collector's success gate now resolves the last checkpoint by position, so a re-map is safe *for the label*, but the numbers must still move as a set. Since design A's retirement the ranges no longer feed a *training* normalization (`generic_c` normalizes its auxiliary channels with lerobot's action MIN_MAX), but screening still reads them, so keep them current, not dead.
8. **Obs keys define the dataset schema** (`dataset_recorder.py`) — adding/removing a key in `get_observations` makes existing datasets unresumable. One-way door; decide before collecting.

## Where the code lives

| Script (under `robot_imitation_glue/`) | Does |
|---|---|
| `ur5station/bottle/collect_data_bottle_opening.py` | collection entrypoint (per-split pose seeds, pose blacklist) |
| `collect_data_bottle.py` | the collection loop: servo recording, checkpoint retries, FT-bias capture, rerun view, stop-at-target |
| `ur5station/bottle/collect_recovery_bottle.py` | the 20 recovery episodes (induced too-deep contact, recorded recovery + scripted opening) |
| `ur5station/bottle/prepare_datasets_bottle.py` | the 8 prepared datasets ({9,12}-dim action × 4 levels), success filtering, FT drift subtraction, relative actions; pool = `--pool-count 50 --extra-raw-root .../bottle_opening_train_recovery --relative-actions`; `--ft-orientation-correction` exists but was **not** used for the curve datasets |
| `ur5station/bottle/screen_instrumentation.py` | privilege gate (mandatory) + per-modality suitability (§1.4, **diagnostic-only since design A's retirement**) |
| `ur5station/bottle/train_ast_bottle.py` | ~~design-A audio pretraining~~ **retired 2026-10** with design A; stays on disk for the button-experiment lineage (and still defines `CALIBRATED_RANGE`, which screening imports) |
| `ur5station/bottle/generate_configs.py` | the 8 curve configs (`--arms generic,generic_c`, default); parity assertion; emits into `bottle/configs/` |
| `ur5station/bottle/eval_bottle.py` | rollout entrypoint: seeded eval poses, 65 s timeout, 96 N force abort, sustained S1+S2 success, **Enter stops a running rollout → operator visual verdict** (`success_source: operator|sensors` in results rows), `hover_brightness` logged per rollout, `--condition ood --tag sticker` for the OOD runs; records every rollout + `outputs/eval/bottle_results.json`. The right-gripper re-grasp calls are currently commented out (open item) |
| `ur5station/bottle/plot_curves.py` | the ID curve and outcome figures (`outputs/figures/bottle_{curve,outcomes}_id.*`) |
| `ur5station/bottle/splits.py` | the split definitions (seeds, counts, blacklists), shared by collection and eval |
| `ur5station/bottle/repair/` | one-off dataset-repair scripts from past incidents (runbook: `docs/repairing_broken_lerobot_datasets.md`) |
| `agents/lerobot_agent.py` | `n_env_action_dims=9` slices `generic_c`'s auxiliary channels off at inference (pass `9` for **all** arms — a no-op for the others, byte-identical eval path) |

Fork changes live in the `lerobot` submodule. Committed at `dd8ad224`: the AST position-embedding resize (§4.7 — `ignore_mismatched_sizes` was silently randomizing them) and the `rgb_encoder_init_checkpoint` / `audio_encoder_init_checkpoint` fields (strict load — **retired with design A**, kept on disk for the button lineage). **Uncommitted** in the submodule's working tree (the curve checkpoints were trained with them): the audio-normalization and AdamW changes, the ManiWAV-style fusion option, and `se3_relative_action_processor.py`. The treatment arm needs **no** fork change: lerobot derives the denoised width from the dataset's `action` feature.

## Current state and what comes next

**The full results record is `Bottle_Pilot_Results.md`** — read it first. `Bottle_Operator_Guide.md` is the runbook (§9 is the "why didn't it help" Q&A).

As of 20 October 2026 (branch `ral_simplified`), the pilot is essentially finished:
- **Data:** 90 expert episodes collected 2026-10-09/10 under the recalibrated constants, plus 20 recovery episodes 2026-10-13/14. N = 70 (the first 50 expert in the seeded order + the 20 recovery).
- **Training:** all 8 curve models trained (150K steps, GPULab); checkpoints under `outputs/train/bottle/<arm>_<level>/checkpoints/150000/pretrained_model` and on HF (`ramen-noodels/bottle_<arm>_<level>`, public). `generic_fusion_100` is trained as an optional ablation, not evaluated on the robot.
- **ID curves (160 rollouts, done):** `generic` 30/65/85/90% vs `generic_c` 25/60/75/90% at 18/35/52/70 episodes; paired −5 pp [−16, +6]. **No benefit of instrumentation.** The linear probe shows `generic` already encodes the cap state at 18 episodes (R² ≈ 0.89); failures are execution errors after first contact.
- **OOD sticker:** 100% done (`generic` 17/20 vs `generic_c` 14/20, McNemar p = 0.38); 50%: `generic_50` done (13/20), `generic_c_50` still to run.

Still open (results doc §8): the `generic_c_50` sticker run; an optional matched-lighting 50% ID re-run (the ID sessions are confounded with daylight); pose 0 run as a session's third rollout; the eval re-grasp decision and pre-rollout cap/camera checks; committing the code (incl. the lerobot fork changes above).

The early numbers in the protocol from the first 51-episode set (N=51, median 31.1 s, the screening scores proprio 0.182 / image 0.787 / audio 0.606 / FT 0.096) were never re-derived on the final data; report screening only after a re-run. The models in the §1.6 N-finding log predate the 2026-10-15 audio fix and ignore audio.

Open issues that can bite: the FT offset in the 90 expert demos (collected without `zeroFtSensor()`; collector fixed 2026-10-19, data not corrected — results doc §5.2), the camera auto-exposure lighting confound (results doc §5.3), the spectrogram being AV1-compressed before the AST sees it (§4.8, open — also the reason audio paths force `video_backend="pyav"`). The old **sensor re-cover** worry is now measured and largely absorbed (Part 2, Task 1): pre-pop light leaks run to ~2% of samples but never set a threshold (percentile bounds + 0.3 s debounce + 5-step sustained success + a 0.5 s stop-open dwell), though it is pose-dependent — one "test" pose never opened the cap in 3/3 attempts.
