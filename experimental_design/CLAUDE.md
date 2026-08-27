This file orients AI agents working on the instrumented-imitation-learning experiment. It summarizes; `Experiment_Protocol.md` in this folder is the authoritative document, and where they disagree, the protocol wins.

## What this folder is

| File | Status |
|---|---|
| `Experiment_Protocol.md` | **Authoritative.** Part 1 = shared methodology, Part 2 = per-task checklists, Part 3 = open promotor items, Part 4 = code issues (FIXED = committed) |
| `Task_Specifications.md` | Superseded (8 July 2026). Kept as history; do not follow it |
| `session.json` | Transcript of the original design conversation with a local assistant |
| `Promotor_Meeting_Report.pdf`, `Promotor_Meeting_Decison_map.svg` | Materials from the 1 July 2026 promotor meeting |

## The paper (target: RA-L)

**Research question.** Does *task-specific instrumentation* — privileged sensor signals added to the environment, available during training but absent at inference — improve the **data efficiency** of imitation learning?

**Hypothesis.** Supervising a policy with the instrumentation signal is an inductive bias toward task-relevant features: the same success rate is reached with fewer demonstrations than random initialization or generic (ImageNet/AudioSet) pretraining achieve.

**Contribution boundary — important for framing anything written about this work.** A colleague's workshop paper already shows the mechanism helps at 100% data, on one task, one seed. This paper's claim is therefore the **data-efficiency curves** (success rate vs. fraction of demonstrations) and the **ID/OOD generalization split** — not the mechanism itself. The mechanism comparison below is a controlled replication that decides which design the rest of the paper uses. The proprioception privilege test (§1.4) is a supporting methodological note, not a headline claim.

## The bottle pilot

One physical bottle with a flick-open cap instrumented by a **3-channel phototransistor sensor** (voltage per channel; a channel "uncovers" as its tab opens). Two UR5s: the right arm presents the bottle at ~100 sampled poses, the left arm opens the cap. Demonstrations are **scripted**, gated by sensor checkpoints (S0 at `push_end`, S1 at `leg_3_end`, S2 at `leg_6_end`), with failed checkpoints triggering recorded retries — recoveries are deliberately part of the data. The OOD axis is **appearance only** (stickers on the bottle; one cap, so no dynamics variation — the paper must call this appearance robustness, not generalization unqualified). Details: protocol Part 2, Task 1.

Modalities: wrist camera (720p, no fallback), mel spectrogram (AST), and `observation.state` = TCP pose (6) ⊕ drift-corrected internal FT (6). Actions are **tool-frame deltas** `[delta_xyz(3), rot6d(6)]` — kept deliberately for equivariance to bottle placement; §1.7 records the full argument and the rejected alternatives.

## The four arms (× 4 data levels = 16 training runs)

| Arm | Encoder init | Output | Isolates |
|---|---|---|---|
| `from_scratch` | random | action(9) | lower baseline |
| `generic` | ImageNet + AudioSet | action(9) | is free pretraining enough? |
| `design_a` | generic → finetuned to predict the sensor | action(9) | instrumentation *as initialization*, on top of generic |
| `design_c` | random | action(9) ⊕ sensor(3) = 12-dim denoised vector | instrumentation *as auxiliary prediction* (expected winner) |

Data levels 100/75/50/25% of N successful episodes (nested, fixed-seed shuffled). Fixed 100K steps, **final checkpoint, no selection**. Evaluation: 20 real rollouts per config, 10 ID + 10 OOD; statistics pooled across tasks (CMH, stratified by task) because n=10 per cell cannot carry the claim alone. Design C's 3 extra channels take 3/12 of the ε-prediction loss automatically (no λ) and are sliced off at inference.

## Invariants — do not break these

1. **Arms differ only in intended keys.** `generate_configs.py` asserts this at build time (`INTENDED_ARM_DIFFERENCES`). Never hand-edit a generated config; change the generator.
2. **Input normalization is matched to encoder initialization** (per-arm, deliberate): `generic` gets ImageNet/AudioSet stats; the others get dataset stats. `audio_norm_mean/std` default to 0/1 = *no normalization* — they must always be set explicitly. §1.7.
3. **`observation.state` is the only state key the policy reads** — anything the policy should see as proprioception must be concatenated into it. Other `observation.*` vectors are typed but never reach the conditioning.
4. **`bottle_sensor` cannot leak into arms 1–3**: lerobot's `batch_to_transition` drops every key that is not `observation.*`/`action`/bookkeeping. For design C the signal must therefore live *inside* `action` (done at dataset-prep time).
5. **Data-level subsets are separate prepared datasets**, never `dataset.episodes` lists — that field is broken for non-prefix lists (absolute vs. relative frame indices). Every prepared dataset is built from the raw root, one video generation each.
6. **The pose blacklist is for collection feasibility only, never task difficulty** — excluding hard poses inflates the success rate, on the test split directly. See the comment at its definition.
7. **Sensor calibration constants travel together**: `PER_CHANNEL_THRESHOLDS` (`hardware/bottle_sensor.py`) and `CALIBRATED_RANGE` (`train_ast_bottle.py`) are both derived from `bottle_experiment/sensor_logs/run_{0003,0005,0006}.json`. Re-derive both whenever the cap, mounting or motion changes.
8. **Obs keys define the dataset schema** (`dataset_recorder.py`) — adding/removing a key in `get_observations` makes existing datasets unresumable. One-way door; decide before collecting.

## Where the code lives

| Script (under `robot_imitation_glue/`) | Does |
|---|---|
| `ur5station/collect_data_bottle_opening.py` | collection entrypoint (per-split pose seeds, pose blacklist) |
| `collect_data_bottle.py` | the collection loop: servo recording, checkpoint retries, FT-bias capture, rerun view, stop-at-target |
| `ur5station/prepare_datasets_bottle.py` | the 8 prepared datasets ({9,12}-dim action × 4 levels), success filtering, FT drift subtraction |
| `ur5station/screen_instrumentation.py` | privilege gate (mandatory) + per-modality suitability (§1.4) |
| `ur5station/train_ast_bottle.py` | design-A audio pretraining (fork of `train_ast_single.py`, which stays button-only) |
| `ur5station/lerobot_train/bottle/generate_configs.py` | the 16 configs + parity assertion |
| `agents/lerobot_agent.py` | `n_env_action_dims=9` slice for design C (pass for all arms) |

Fork changes live in the `lerobot` submodule at `dd8ad224`: the AST position-embedding resize (§4.7 — `ignore_mismatched_sizes` was silently randomizing them) and the `rgb_encoder_init_checkpoint` / `audio_encoder_init_checkpoint` fields (strict load). Design C needs **no** fork change: lerobot derives the denoised width from the dataset's `action` feature.

## Current state and what comes next

Tooling is committed; **no experiment data exists yet**. The order of work:

1. Collect (train/val/test splits, distinct seeds) — first verifying UR payload compensation for the FT baseline and the sticker conditions in Part 2, Task 1.
2. Screening: privilege gate, then modality suitability (fix thresholds *before* looking).
3. Design-A pretraining per data level → `audio_norm_mean/std` feed `generate_configs.py`.
4. 16 training runs → 320 rollouts → mechanism decision (Table 2) → carry the winner to the remaining tasks (plugs, petri dish; wiping conditional on its force reformulation).

Open issues that can bite: the payload-compensation check (§1.7 FT note), the sensor re-cover assumption behind the checkpoint scheme (Part 2, Task 1 warning), and the spectrogram being H.264-compressed before the AST sees it (§4.8, open).
