This file orients AI agents working on the instrumented-imitation-learning experiment. It summarizes; `Experiment_Protocol.md` in this folder is the authoritative document, and where they disagree, the protocol wins.

## What this folder is

| File | Status |
|---|---|
| `Experiment_Protocol.md` | **Authoritative.** Part 1 = shared methodology, Part 2 = per-task checklists, Part 3 = open promotor items, Part 4 = code issues (FIXED = committed) |
| `Bottle_Operator_Guide.md` | The runbook: exact commands in order, decision rules, failure lookup |
| `Task_Specifications.md` | Superseded (8 July 2026). Kept as history; do not follow it |
| `session.json` | Transcript of the original design conversation with a local assistant |
| `Promotor_Meeting_Report.pdf`, `Promotor_Meeting_Decison_map.svg` | Materials from the 1 July 2026 promotor meeting |

## The paper (target: RA-L)

**Research question.** Does *task-specific instrumentation* — privileged sensor signals added to the environment, available during training but absent at inference — improve the **data efficiency** of imitation learning?

**Hypothesis.** Supervising a policy with the instrumentation signal is an inductive bias toward task-relevant features: the treatment arm reaches a given success rate with **fewer demonstrations** than a matched control — identical architecture, identical (generic ImageNet/AudioSet) initialization, no instrumentation. Since the `ral_simplified` revision (2026-10) the comparison is single-factor: both curve arms start pretrained; random init survives only as a one-time verification cell (§1.3).

**Contribution boundary — important for framing anything written about this work.** A colleague's workshop paper already shows the mechanism helps at 100% data, on one task, one seed — and its C arm differed from its baselines in *two* factors at once (random init AND auxiliary channels). This paper's claim is therefore the **data-efficiency curves** (success rate vs. fraction of demonstrations) and the **ID/OOD generalization split**, on a design that **de-confounds** the workshop comparison — not the mechanism itself. The old A-vs-C mechanism table was retired with `ral_simplified`; nothing "decides the design" anymore, the design is fixed. The proprioception privilege test (§1.4) is a supporting methodological note, not a headline claim; per-modality suitability scores are diagnostic-only since design A's retirement.

## The bottle pilot

One physical bottle with a flick-open cap instrumented by a **3-channel phototransistor sensor** (voltage per channel; a channel "uncovers" as its tab opens). Two UR5s: the right arm presents the bottle at ~100 sampled poses, the left arm opens the cap. Demonstrations are **scripted**, gated by sensor checkpoints (`SENSOR_CHECKPOINTS`: S0 at `leg_2_end`, S1+S2 at `leg_3_end` as of the 2026-10-07 re-map; invariant 7), with failed checkpoints triggering recorded retries — recoveries are deliberately part of the data. The OOD axis is **appearance only** (stickers on the bottle; one cap, so no dynamics variation — the paper must call this appearance robustness, not generalization unqualified). Details: protocol Part 2, Task 1.

Modalities: wrist camera (720p, no fallback), mel spectrogram (AST), and `observation.state` = TCP pose (6) ⊕ drift-corrected internal FT (6). Actions are **tool-frame deltas** `[delta_xyz(3), rot6d(6)]` — kept deliberately for equivariance to bottle placement; §1.7 records the full argument and the rejected alternatives.

## The arms (`ral_simplified`: 2 live arms × 4 data levels = 8 curve runs, + 1 verification run)

| Arm | Role | Encoder init | Output | Isolates |
|---|---|---|---|---|
| `generic` | **control** | ImageNet + AudioSet | action(9) | what standard pretrained init alone achieves |
| `generic_c` | **treatment** | ImageNet + AudioSet | action(9) ⊕ sensor(3) = 12-dim denoised vector | instrumentation *as auxiliary prediction* — the paper's single factor |
| `from_scratch` | init verification (**100% level only**) | random | action(9) | checks the pretrained-init assumption; pre-registered one-sided rule (§1.3), appendix row unless it ties/beats `generic`@100% ID |

`design_a` and the random-init `design_c` arm are **retired** (2026-10); `generic_c` keeps design C's mechanism and moves it onto the standard pretrained init, which makes treatment-minus-control a one-factor difference. Data levels 100/75/50/25% of N successful episodes (nested, fixed-seed shuffled). Fixed 100K steps, **final checkpoint, no selection**. Evaluation: 40 real rollouts per curve config (20 ID + 20 OOD), 20 ID for the verification cell = 340 pilot rollouts; statistics pooled across tasks (CMH, stratified by task) because n=20 per cell cannot carry the claim alone. `generic_c`'s 3 extra channels take 3/12 of the ε-prediction loss automatically (no λ) and are sliced off at inference — the sensor is never a policy input, so the deployed policy needs no sensor hardware ("no reliance").

## Invariants — do not break these

1. **Arms differ only in intended keys.** `generate_configs.py` asserts this at build time (`INTENDED_ARM_DIFFERENCES`). Never hand-edit a generated config; change the generator.
2. **Input normalization is matched to encoder initialization** (per-arm, deliberate): `generic` and `generic_c` get ImageNet/AudioSet stats; the `from_scratch` verification cell gets dataset stats. `audio_norm_mean/std` default to 0/1 = *no normalization* — they must always be set explicitly. §1.7.
2b. **Audio stats are per LEVEL, never global**: the random-init cell's configs use that level's `audio_stats.json` (computed over exactly the episodes the level trains on). Reusing the 100% run's stats at 25% leaks episodes the run never sees — invisible to the arm comparison, poison to the data-efficiency claim. The generic arms carry AudioSet's dataset-independent constants (−4.2677393 / 4.5689974) and have no leak surface; the rule exists to protect the verification cell and the generator still enforces it.
2c. **The instrumentation is never a policy input.** Sensor values live only as the 3 auxiliary channels of `generic_c`'s 12-dim *action* feature (added at dataset-prep time), and `LerobotAgent(n_env_action_dims=9)` slices them off at inference. "Available during training, absent at inference" is the whole no-reliance claim — anything that puts a cap-sensor reading on the observation side breaks the design, not just a metric. (The rule binds only the *instrumentation*: fusing the two deploy-available observation streams, image × audio, is legal by construction and was still declined as a design-time change — reasoning in protocol §1.2; don't propose it mid-campaign.)
3. **`observation.state` is the only state key the policy reads** — anything the policy should see as proprioception must be concatenated into it. Other `observation.*` vectors are typed but never reach the conditioning.
4. **`bottle_sensor` cannot sneak in through the observation side**: lerobot's `batch_to_transition` drops every key that is not `observation.*`/`action`/bookkeeping. That is the mechanical enforcement of 2c — the only route the signal has into training is *inside* `action` (done at dataset-prep time), which is exactly where `generic_c`'s treatment lives.
5. **Data-level subsets are separate prepared datasets**, never `dataset.episodes` lists — that field is broken for non-prefix lists (absolute vs. relative frame indices). Every prepared dataset is built from the raw root, one video generation each.
6. **The pose blacklist is for collection feasibility only, never task difficulty** — excluding hard poses inflates the success rate, on the test split directly. See the comment at its definition.
7. **Sensor calibration constants travel together**: `PER_CHANNEL_THRESHOLDS` (`hardware/bottle_sensor.py`, the single source — the collector, eval and the calibration recorder all import it) and `CALIBRATED_RANGE` (defined in `train_ast_bottle.py`, imported by `screen_instrumentation.py`). **Re-derived 2026-10-07** from the calibration batch `run_{0013,0018,0020,0021,0022,0023}.json` (the re-tuned motion) via `bottle_experiment/derive_thresholds.py`; the old values came from `run_{0003,0005,0006}` under a previous layout. `SENSOR_CHECKPOINTS` was re-mapped in the same pass (`leg_2_end`→S0, `leg_3_end`→S1+S2). Whenever the cap, mounting or motion geometry changes, re-record the calibration batch and re-derive **both** together — the collector's success gate now resolves the last checkpoint by position, so a re-map is safe *for the label*, but the numbers must still move as a set. Since design A's retirement the ranges no longer feed a *training* normalization (`generic_c` normalizes its auxiliary channels with lerobot's action MIN_MAX), but screening still reads them, so keep them current, not dead.
8. **Obs keys define the dataset schema** (`dataset_recorder.py`) — adding/removing a key in `get_observations` makes existing datasets unresumable. One-way door; decide before collecting.

## Where the code lives

| Script (under `robot_imitation_glue/`) | Does |
|---|---|
| `ur5station/bottle/collect_data_bottle_opening.py` | collection entrypoint (per-split pose seeds, pose blacklist) |
| `collect_data_bottle.py` | the collection loop: servo recording, checkpoint retries, FT-bias capture, rerun view, stop-at-target |
| `ur5station/bottle/prepare_datasets_bottle.py` | the 8 prepared datasets ({9,12}-dim action × 4 levels), success filtering, FT drift subtraction |
| `ur5station/bottle/screen_instrumentation.py` | privilege gate (mandatory) + per-modality suitability (§1.4, **diagnostic-only since design A's retirement**) |
| `ur5station/bottle/train_ast_bottle.py` | ~~design-A audio pretraining~~ **retired 2026-10** with design A; stays on disk for the button-experiment lineage (and still defines `CALIBRATED_RANGE`, which screening imports) |
| `ur5station/bottle/generate_configs.py` | the 8 curve configs (`--arms generic,generic_c`, default) + the verification cell (`--arms from_scratch --levels 100`); parity assertion; emits into `bottle/configs/` |
| `ur5station/bottle/eval_bottle.py` | rollout entrypoint: seeded eval poses, 65 s timeout (2× median demo, **provisional pending recollection**), sustained-sensor success, **Enter stops a running rollout → operator visual verdict** (`success_source: operator|sensors` in results rows), records every rollout + results JSON |
| `ur5station/bottle/splits.py` | the split definitions (seeds, counts, blacklists), shared by collection and eval |
| `ur5station/bottle/repair/` | one-off dataset-repair scripts from past incidents (runbook: `docs/repairing_broken_lerobot_datasets.md`) |
| `agents/lerobot_agent.py` | `n_env_action_dims=9` slices `generic_c`'s auxiliary channels off at inference (pass `9` for **all** arms — a no-op for the others, byte-identical eval path) |

Fork changes live in the `lerobot` submodule at `dd8ad224`: the AST position-embedding resize (§4.7 — `ignore_mismatched_sizes` was silently randomizing them) and the `rgb_encoder_init_checkpoint` / `audio_encoder_init_checkpoint` fields (strict load — **retired with design A**, kept on disk for the button lineage). The treatment arm needs **no** fork change: lerobot derives the denoised width from the dataset's `action` feature.

## Current state and what comes next

Follow `Bottle_Operator_Guide.md` — it is the runbook, with exact commands and the decision rules for each stage. As of 7 October 2026 (branch `ral_simplified`): the design was simplified to the two-arm curve + verification cell; `generate_configs.py` was rewritten to match and the retired `design_a`/`design_c` configs were deleted (the 8 curve configs + `bottle_from_scratch_100.json` regenerate cleanly, parity assertion green). The opening motion was **re-tuned leg by leg** and the cap sensor **recalibrated** against it (invariant 7): thresholds, `CALIBRATED_RANGE` and the checkpoint map all re-derived 2026-10-07 from `run_{0013,0018,0020,0021,0022,0023}` via the new `derive_thresholds.py`. The shared motion parameters were consolidated into a single `bottle_experiment/motion_constants.py` (demo/fork/agent/collector all import it; the recorder also imports the sensor constants from `bottle_sensor.py`) — the old "keep in sync by hand" copies are gone. **Nothing measured on the pre-recalibration data carries forward**: the train split (51 episodes) is being recollected under the new constants, so N=51, median 31.1 s, the 65 s timeout, and the screening scores (proprio 0.182 — gate passed; image 0.787; audio 0.606; FT 0.096) are all **provisional**. Partial pre-revision eval exists for `from_scratch_100` (a handful of rollouts under the old thresholds and the old eval flow — rows without `success_source`); treat it as spent pilot effort, not a result. Still ahead: sticker-threshold re-confirm → recollection → rebuild prepared datasets → the §1.6 N-finding loop on `generic`@100% → the 8 curve trainings + 1 verification training → 340 rollouts via `ur5station/bottle/eval_bottle.py`.

Open issues that can bite: the payload-compensation check (§1.7 FT note), the spectrogram being AV1-compressed before the AST sees it (§4.8, open — also the reason audio paths force `video_backend="pyav"`). The old **sensor re-cover** worry is now measured and largely absorbed (Part 2, Task 1): pre-pop light leaks run to ~2% of samples but never set a threshold (percentile bounds + 0.3 s debounce + 5-step sustained success + a 0.5 s stop-open dwell), though it is pose-dependent — one "test" pose never opened the cap in 3/3 attempts.
