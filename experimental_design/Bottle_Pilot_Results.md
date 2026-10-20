# Bottle pilot — complete results record

**Status:** 2026-10-20. In-distribution curves done (160 rollouts). The sticker (OOD) test is done at 100%; 50% is in progress.

**One-sentence summary:** instrumentation didn't help here because the cap state is clearly visible to the wrist camera. The un-instrumented policy already learned it from the action loss alone with 18 demonstrations, so the auxiliary sensor signal had nothing new to teach. The remaining failures were execution errors (losing the lid edge mid-opening) that knowing the cap state doesn't fix.

Related documents:
- `Experiment_Protocol.md` §1.6 and Task 1: the pre-registered design and decision log.
- `Bottle_Operator_Guide.md` §9: a Q&A version of the "why didn't it help" analysis.
- `outputs/presentations/bottle_pilot_results.pptx`: the slide deck for the PI.

---

## 1. Setup

| | |
|---|---|
| **Task** | The left UR arm (Schunk gripper, wrist camera) opens a flip-cap bottle held by the right UR arm: approach the cap, push the lid edge, follow the opening motion until the lid is fully open |
| **Instrumentation** | 3-channel phototransistor sensor in the cap (BLE → DDS topic `Bottle`): voltages S0 / S1 / S2 that rise as each tab of the lid uncovers. Thresholds `[3.20, 3.12, 3.10]` V (`hardware/bottle_sensor.py`). S1 + S2 decide "open" (episode success); S0 opens first and has no usable gap, so it is logged only |
| **Policy inputs** | wrist RGB (720p → 320×240, random crop 216×288); audio spectrogram (Kaldi fbank, 128 mel × 298 frames, 3 s window at 16 kHz, stored as `(fbank+20)/60`); state = TCP pose (6) + drift-corrected F/T (6). 2 observation steps |
| **Arms** | `generic` (control): predicts the action. `generic_c` (instrumented): predicts the action **plus the 3 sensor channels** over the same 16-step horizon; the sensor channels are dropped at inference. Nothing else differs (`generate_configs.py` asserts this) |
| **Never an input** | the cap sensor. It is only a training target for `generic_c`, and the success detector at evaluation |

## 2. Data collection

### 2.1 Expert demonstrations (90 episodes, 2026-10-09/10)
- **Collector:** `robot_imitation_glue/collect_data_bottle.py` via `ur5station/bottle/collect_data_bottle_opening.py`. Raw root: `bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train`.
- **Scripted motion** (`bottle_experiment/open_bottle_agent.py`, `motion_constants.py`):
  - left arm home → per-episode F/T bias at home;
  - right arm presents the bottle at a seeded **train-split pose** (re-grasp at the neutral pose);
  - left arm hovers 10 cm above the cap; a touch-point detector (or an operator click) picks the rim point;
  - descend to the grip point, push 5 cm at 8°, then 6 "legs" that follow the lid with the gripper yawed 100°, at 5 cm/s, under a 10 Hz control loop.
- **Sensor-gated checkpoints with retries:** if a checkpoint reads "not open", the gripper retracts and resumes deeper. **Stop-on-open:** the motion ends once S1 and S2 read open; the retreat to hover is recorded.
- Retries show up in about 5 of the 90 episodes (estimated from the trajectories; the collector didn't log them).
- Only successful episodes are used (`next.success`).

### 2.2 Recovery demonstrations (20 episodes, 2026-10-13/14)
- **Collector:** `ur5station/bottle/collect_recovery_bottle.py`. Raw root: `.../expert/bottle_opening_train_recovery`.
- **Why:** the early policies failed by reaching the grip point too deep (129.8 mm vs 126 mm below hover), jamming at 20–38 N and never recovering.
- **Each episode:**
  - an *induced failure*, not recorded: descend deeper than planned, then push until jammed (stops at a jam-force limit);
  - the *recovery*, recorded: lift off along the cap normal, return over the grip point, then the normal scripted demo with checkpoints and stop-on-open.
- The F/T sensor **is** re-zeroed (`zeroFtSensor`) before the bias in this collector.

### 2.3 Poses and splits (`ur5station/bottle/splits.py`)
- **Seeded pose streams:** separate seeds for train / val / test (`RANDOM_SEED + {0, 1, 2}`).
- **Blacklists:** train poses 6, 10 and 68; test poses 0, 9, 10, 19 and 23 (rig problems). The evaluation always uses the same 20 test poses in the same order (`pose_index` = list position).
- **Appearance:** identical between ID training and ID evaluation. OOD = a sticker on the lid (§4.5).

### 2.4 Prepared datasets (`ur5station/bottle/prepare_datasets_bottle.py`)
- **Pool:** the first 50 expert episodes of the seeded order (`EPISODE_ORDER_SEED` 20260812) plus all 20 recovery episodes, giving **70 episodes**. They are shuffled and cut into nested levels: **100% = 70, 75% = 52, 50% = 35, 25% = 18 episodes**, each about 22–29% recovery. Manifest: `datasets/bottle_experiment/prepared/episode_order_pool50.json`.
- **Variants:** 9-dim (`generic`) and 12-dim (`generic_c`, the action plus `bottle_sensor`) per level, both built directly from the raw root so video compression is identical.
- **Actions:** UMI-style relative actions. Absolute target `[xyz, rot6d]` expressed in the frame of the TCP at the latest observation (`--relative-actions`).
- **Normalization:** images ImageNet mean/std; state and action min/max. Audio uses **per-level** dataset mean/std (`audio_stats.json`), the same for both arms of a level.

## 3. Training
- **Model:** lerobot fork (`lerobot/`, branch with the audio and fusion changes), Diffusion Policy with a U-Net (down dims 512/1024/2048), DDIM with 100 training timesteps, horizon 16, 2 observation steps, 8 action steps.
- **Encoders:**
  - image: ResNet-18 with ImageNet weights, 32 spatial-softmax keypoints, one encoder per camera;
  - audio: AST (`MIT/ast-finetuned-audioset-10-10-0.4593`), AudioSet weights, trained end to end;
  - fusion: concatenation.
- **Optimization:** 150K steps, batch 32, **AdamW** (lr 1e-4, betas 0.95/0.999, weight decay 1e-6), cosine schedule with 500 warmup steps, grad clip 10, seed 2025. Same budget for every model.
- **Where:** GPULab (H200 / 5090 / 4090). Checkpoints are on Hugging Face (`ramen-noodels/bottle_<arm>_<level>`, **public**) and local under `outputs/train/bottle/<name>/checkpoints/150000/pretrained_model`.
- **Configs:** `ur5station/bottle/configs/bottle_{generic,generic_c}_{25,50,75,100}.json`, plus `bottle_generic_fusion_100.json` (ManiWAV-style modality-attention fusion, trained but not yet evaluated on the robot).

## 4. Results

### 4.1 Finding N (2026-10-12 to 14; these models predate the audio fix in §5.1)
| Model | Data | ID success |
|---|---|---|
| `generic_50ep` | 50 expert, delta actions, 100K steps | 12/20 = 60% |
| `generic_90ep` | 90 expert, delta actions | 6/20 = 30% (too-deep first contact, no recovery) |
| `generic_50ep_recovery_rel` | 50 expert + 20 recovery, relative actions, 150K steps | 19/20 = 95% |
| `generic_90ep_recovery_rel` | 90 expert + 20 recovery, relative actions | 19/20 = 95% |

The recovery data plus the switch to relative actions fixed the jamming. N was fixed at 70 episodes (50 expert + 20 recovery).

### 4.2 In-distribution curves (main result; 2026-10-16 to 19)
20 rollouts per model, same poses for both arms. Evaluation settings: 10 DDIM steps, 8 action steps, 65 s timeout, 96 N force abort.

| Episodes | `generic` | `generic_c` | Pose by pose: only control / only instrumented succeeded | McNemar p |
|---|---|---|---|---|
| 18 (25%) | 6/20 = 30% [15–52] | 5/20 = 25% [11–47] | 3 / 2 | 1.00 |
| 35 (50%) | 13/20 = 65% [43–82] | 12/20 = 60% [39–78] | 5 / 4 | 1.00 |
| 52 (75%) | 17/20 = 85% [64–95] | 15/20 = 75% [53–89] | 3 / 1 | 0.62 |
| 70 (100%) | 18/20 = 90% [70–97] | 18/20 = 90% [70–97] | 1 / 1 | 1.00 |

- **Pooled:** paired by pose over all 80 rollouts, instrumented − control = **−5 pp, 95% CI −16 to +6**. CMH over the levels: p = 0.57, common odds ratio 1.34 in favour of the control.
- **Figures:** `outputs/figures/bottle_curve_id.{png,pdf}`, `bottle_outcomes_id.{png,pdf}` (`ur5station/bottle/plot_curves.py`).
- **Outcomes:** at low data, failures are mostly operator stops. Force aborts: 6 control vs 8 instrumented. Robot protective stops: 3 vs 4.

### 4.3 Why it didn't help: mechanism checks (offline, no robot time)
| Check | Result |
|---|---|
| Did `generic_c` learn the sensor? | Yes. On unseen episodes R² 0.91 / 0.95 / 0.70 and 97–98% correct on cap open/closed, **already at 18 episodes**. Training episodes: R² 1.00 |
| Is the sensor data noisy? | No. Noise about 0.003 V vs steps of 0.17 / 0.61 / 1.11 V (signal-to-noise 60–385). About 26% of frames are mid-transition |
| **Does the control learn the cap state anyway?** (linear probe on frozen image features, 36 episodes no model trained on) | **Yes.** Untrained ImageNet ResNet-18: R² 0.75. `generic`: 0.89–0.91 at **every** level. `generic_c` − `generic` is at most ±0.06 at any level, and every 95% interval includes zero at 18 episodes (image +0.017 [−0.024, +0.056]). Audio features alone: R² 0.23–0.37 |
| Is the sensor visible to the camera? | Mostly. In screening the image predicts the sensor with R² 0.79 vs 0.18 for proprioception. This doesn't disqualify a task by itself; what matters is the probe result above |
| Do the demos react to the cap state? | Only partly: scripted motion, retries in ~5/90 expert episodes, recovery behaviour in 17/20 recovery episodes, started from a visibly jammed state |
| Where do policies fail? | **After first contact.** First-contact depth is the same in failures and successes (124.0 vs 124.1 mm below hover, p = 0.55; demos 125.9 mm). 31/56 failures open S1 and lift S2 to about 2.97 V (threshold 3.10) but never clear it, with lower contact force (3.7 vs 5.4 N): the gripper loses the lid edge. 12 have some lid movement without opening; 13 never move the lid (9 of them force aborts) |
| Does instrumentation change behaviour? | Slightly: successful `generic_c` rollouts push longer (force × contact time 145 vs 101 N·s, p < 0.001). Both arms push 2–3× harder downward than the demos (peak \|Fz\| 27 / 31 vs 12 N) |
| Is the audio used? | Since the fix (§5.1), yes: swapping the audio moves actions 4–9× the run-to-run noise. The fusion variant uses it about 1.5× more |

Scripts and data: `outputs/analysis/bottle_2026-10-19/`, with `RESULTS.md` and subfolders `linear_probe/`, `force_sensor_analysis/` and `audio_sensitivity/`.

### 4.3b Linear probe on the frozen encoders: full results

**Question:** does supervising the cap sensor make `generic_c`'s encoders represent the cap state better than `generic`'s at the same data level? This is the direct test of the faster-learning hypothesis at the representation level.

**Method:**
- **Frozen encoders:** each policy's image encoder (ResNet-18 + spatial softmax, 512-d) and audio encoder (AST, 768-d). Features are taken for the current observation step from the policy's own conditioning vector; the state is excluded, because it is identical in both arms and is not learned.
- **Baseline:** an untrained ImageNet ResNet-18 (512-d, global pooled).
- **Probe:** only a linear layer is fitted.
  - Ridge regression (alpha chosen by cross-validation) predicts the three raw sensor voltages; a separate R² per channel, then the mean.
  - Logistic regression predicts cap open (S1 ≥ 3.12 V and S2 ≥ 3.10 V).
- **Data:** 1851 frames (every 6th frame) from **36 expert episodes no curve model trained on**. These are the 40 not in the 70-episode pool, minus 4 with a mislabeled S2. The cap is open in 47% of frames.
- **Validation:** 6-fold cross-validation grouped by episode, so the probe is always tested on episodes it never saw. Differences between arms have 95% bootstrap CIs over episodes (1000 resamples).

| Model | Features | R² S0 | R² S1 | R² S2 | **mean R²** | cap-open balanced acc. | AUC |
|---|---|---|---|---|---|---|---|
| untrained ImageNet | image | 0.76 | 0.79 | 0.71 | **0.75** | 0.951 | 0.986 |
| `generic_25` | image | 0.88 | 0.93 | 0.87 | **0.89** | 0.984 | 0.999 |
| `generic_c_25` | image | 0.91 | 0.94 | 0.88 | **0.91** | 0.984 | 0.998 |
| `generic_50` | image | 0.89 | 0.93 | 0.85 | **0.89** | 0.978 | 0.997 |
| `generic_c_50` | image | 0.87 | 0.91 | 0.86 | **0.88** | 0.978 | 0.987 |
| `generic_75` | image | 0.90 | 0.94 | 0.87 | **0.90** | 0.974 | 0.995 |
| `generic_c_75` | image | 0.91 | 0.95 | 0.89 | **0.92** | 0.981 | 0.998 |
| `generic_100` | image | 0.91 | 0.95 | 0.86 | **0.91** | 0.975 | 0.996 |
| `generic_c_100` | image | 0.88 | 0.93 | 0.89 | **0.90** | 0.981 | 0.994 |
| `generic_25` | audio | 0.26 | 0.24 | 0.19 | 0.23 | 0.763 | 0.840 |
| `generic_c_25` | audio | 0.27 | 0.27 | 0.23 | 0.26 | 0.781 | 0.865 |
| `generic_50` | audio | 0.30 | 0.29 | 0.18 | 0.26 | 0.769 | 0.848 |
| `generic_c_50` | audio | 0.35 | 0.36 | 0.23 | 0.31 | 0.791 | 0.874 |
| `generic_75` | audio | 0.34 | 0.34 | 0.27 | 0.32 | 0.810 | 0.885 |
| `generic_c_75` | audio | 0.35 | 0.35 | 0.22 | 0.31 | 0.783 | 0.865 |
| `generic_100` | audio | 0.36 | 0.37 | 0.29 | 0.34 | 0.803 | 0.882 |
| `generic_c_100` | audio | 0.37 | 0.37 | 0.27 | 0.33 | 0.788 | 0.876 |
| `generic_25` | image + audio | 0.86 | 0.91 | 0.84 | 0.87 | 0.982 | 0.998 |
| `generic_c_25` | image + audio | 0.89 | 0.94 | 0.87 | 0.90 | 0.986 | 0.998 |
| `generic_50` | image + audio | 0.88 | 0.93 | 0.85 | 0.89 | 0.977 | 0.998 |
| `generic_c_50` | image + audio | 0.89 | 0.92 | 0.86 | 0.89 | 0.979 | 0.994 |
| `generic_75` | image + audio | 0.89 | 0.93 | 0.86 | 0.89 | 0.980 | 0.997 |
| `generic_c_75` | image + audio | 0.91 | 0.95 | 0.89 | 0.92 | 0.980 | 0.998 |
| `generic_100` | image + audio | 0.90 | 0.94 | 0.86 | 0.90 | 0.975 | 0.997 |
| `generic_c_100` | image + audio | 0.88 | 0.93 | 0.87 | 0.90 | 0.980 | 0.996 |

**`generic_c` − `generic`, mean R² [95% bootstrap CI over episodes]:**

| Level (episodes) | Image | Audio | Image + audio |
|---|---|---|---|
| 25% (18) | +0.017 [−0.024, +0.056] | +0.027 [−0.021, +0.078] | +0.026 [−0.013, +0.066] |
| 50% (35) | −0.010 [−0.060, +0.019] | +0.057 [+0.008, +0.106] | +0.005 [−0.034, +0.030] |
| 75% (52) | +0.013 [+0.004, +0.022] | −0.011 [−0.083, +0.051] | +0.020 [+0.011, +0.029] |
| 100% (70) | −0.009 [−0.064, +0.027] | −0.008 [−0.045, +0.032] | −0.003 [−0.060, +0.036] |

**Reading:**
- **Policy training adds a lot, instrumentation adds nothing.** It takes the image encoder from 0.75 (ImageNet) to about 0.89–0.91 at **every** level, already at 18 episodes, with or without the sensor target. The control learns the cap state from the action loss alone.
- **The arm differences are tiny:** at most ±0.06 R². Three cells exclude zero (50% audio, 75% image and image + audio), all small and with no trend.
- **No data-efficiency pattern:** the hypothesis predicts the largest gain at the lowest level, and at 25% every interval includes zero.
- **Audio carries little cap-state information** (R² 0.23–0.37), rising slowly with data in both arms alike. Adding audio to image features doesn't improve on the image features alone.
- **Open/closed is saturated:** balanced accuracy is 0.95 even for the untrained baseline. Closed vs fully open is visually obvious; the continuous R² is the sensitive measure.

Code: `outputs/analysis/bottle_2026-10-19/linear_probe/` (`probe_extract.py` extracts the frozen features, `probe_fit.py` fits the probes and computes the CIs, `probe_cache/` holds the cached cross-validated predictions).

### 4.4 Appearance robustness: offline pre-check
The images were edited digitally: 718 frames (sticker) and 386 frames (dot) from 36 unseen episodes, before the lid moves.

| Perturbation | Change in predicted motion, control vs instrumented | Verdict |
|---|---|---|
| Synthetic striped sticker on the lid (rim and dot left visible) | 10.1 vs 10.2 mm (70 demos), 9.6 vs 9.6 mm (18 demos); difference −0.8 mm [−1.3, +0.2]. Run-to-run noise ≈ 0.3 mm | both disturbed, equally |
| Orange dot recoloured blue | 0.1–0.3 mm, at or below the noise | neither cares |

Examples: `outputs/figures/sticker_examples.png`, `dot_recolor_examples.png`. Scripts: `outputs/analysis/bottle_2026-10-19/sticker_precheck/`.

### 4.5 Appearance robustness: real sticker on the robot (2026-10-20)
- **Command:** `--condition ood --tag sticker`.
- **Conditions:** arms alternated in one session; lighting in the training range (hover brightness 0.34–0.44).

| Arm (70 demos) | Sticker | ID | Poses lost to the sticker |
|---|---|---|---|
| `generic_100` | **17/20 = 85%** [64–95] | 18/20 | 1 (pose 13) |
| `generic_c_100` | **14/20 = 70%** [48–85] | 18/20 | 4 (poses 5, 9, 12, 19) |

- **Paired by pose:** only control succeeded on 4 poses, only instrumented on 1. McNemar p = 0.38.
- **No robustness advantage** for instrumentation; the direction favours the control.
- **Caveat:** the ID 100% sessions ran under bright light, so the ID→sticker drop partly reflects the lighting change. The comparison between arms within the sticker run is clean.
- **50% level:** in progress (`generic_50` / `generic_c_50`, same procedure).

## 5. Bugs and issues found, and their status

### 5.1 Audio was ignored by all early models — FIXED before the curve runs
- **Cause:** AudioSet normalization constants were applied to our spectrogram's stored `(fbank+20)/60` scale, so the AST input was nearly constant (0.540 ± 0.007). Coupled weight decay in lerobot's `torch.optim.Adam` preset then drove the AST weights to about 0 (median 1e-38).
- **Fix (2026-10-15):** per-level dataset audio statistics, and **AdamW** (weight decay 1e-6, as in the original Diffusion Policy and ManiWAV).
- **Check:** audio sensitivity went from ≤ 0.01× to 4–9× the noise floor.
- **Still affected:** the N-finding models of §4.1.

### 5.2 F/T offset in the expert demos — collector FIXED; data correction available, not applied
- **Cause:** the expert collector captured the home bias **without `zeroFtSensor()`**. The sensor offset is fixed in the sensor frame, so free-air force reads 6.7 N median (up to 17 N), varying with the arm's pose, in all 90 expert demos, vs about 2 N in the recovery demos and eval rollouts.
- **Fix (2026-10-19):** the collector now re-zeroes.
- **Correction for existing data:** `prepare_datasets_bottle --ft-orientation-correction` fits raw(t) = R_t·o + c per episode and brings rotated held-out free-air frames from 9.1 to 1.2 N. Not applied, because the curve datasets were built without it and applying it means retraining.
- **Impact:** small. The policies essentially ignore the state input (swapping it moves actions by 0.3–0.4× the noise). It's identical for both arms.

### 5.3 Camera exposure and lighting confound — LOGGED, not fixed in hardware
- **Cause:** the eval and collection camera runs auto-exposure, which can't compensate strong daylight.
- **Effect:** all `generic` ID sessions and `generic_c_100` ran bright (wrist-image brightness 0.53–0.57, 15–25% saturated pixels). `generic_c` 25/50/75 ran at 0.39–0.40, close to the training data's 0.38 ± 0.06.
- **Impact:** the ID arm comparison is confounded with lighting. It probably handicapped the control, which still matched or beat the instrumented arm.
- **Mitigation:** `eval_bottle` now logs `hover_brightness` per rollout and warns outside the training range (0.33–0.62). Lighting should be controlled (blinds, fixed lamps).
- **Pending:** a matched-lighting re-run at 50%.

### 5.4 Pose 0 (first test pose) fails for all 8 curve models — OPEN
- It succeeded for the 4 earlier models.
- **What's known:** contact is normal, then 6/8 models skip the sideways push and press down until the abort. An offline swap test points to the image.
- **Unresolved:** pose 0 is always the **first rollout of a session**, so a session-start effect can't be ruled out.
- **Test:** run it third once.

### 5.5 Sensor labels
- **Channel 3:** 7 of the 90 expert episodes start with S2 already reading open (episodes 11, 14, 55–58, 73), so their channel-3 targets are wrong.
- **Channel 1:** S0 has no usable closed/open gap (it crosses back and forth) and only a 0.17 V step, yet it is one of `generic_c`'s three targets. It's harmless but noisy.

### 5.6 Gripper re-grasp disabled in eval — CHECK
In `eval_bottle.py` both `_move_gripper_to(...)` calls are commented out (open at neutral, close on the bottle). The bottle is therefore not re-grasped between rollouts, unlike collection. Confirm whether this is intended.

### 5.7 Force-abort threshold changed 52 → 96 N (2026-10-16, before the curve evaluation)
- **Why:** 52 N (2× the expert demos' mean peak) scored demo-level pushing as failure, since 4 of the 110 training episodes exceed it.
- **New value:** 96 N = 2 × the 95th percentile of the expert + recovery per-episode peaks (48.0 N).
- **Discarded:** the 3 `generic_100` rollouts run at 52 N; the model was re-evaluated from scratch.

### 5.8 Discarded and redone rollouts (all archived in `datasets/bottle_experiment/eval_discarded/`; results backups in `outputs/eval/bottle_results.backup_*.json`)
| Model | What was removed | Why |
|---|---|---|
| `generic_100` | first 3 rollouts | run under the old 52 N threshold |
| `generic_25` | rollout 9 (pose 8) | operator asked for a redo |
| `generic_75` | rollout 8 (pose 7), a 5-step "success" | the cap was already open at the start |
| `generic_c_75` | 1 rollout | wrist camera frozen |
| `generic_100` sticker | 2 attempts at pose 0 | operator restarts |

### 5.9 Other, minor
- **The spectrogram is video-compressed (AV1)** before the AST sees it (protocol §4.8, open).
- **First-session-rollout issues:** the cap left open (§5.8) and the frozen camera were operator-caught. `eval_bottle` checks neither "the cap reads closed before the rollout" nor "the camera frames are changing". Both checks are worth adding.
- **The HF model repos are public.**

## 6. Interpretation
- **ID curves:** no benefit from instrumentation at any data level. Small effects can't be excluded with 20 rollouts per cell, but an average benefit above about 6 pp is unlikely.
- **Mechanism:** the control's image encoder already represents the cap state at 18 episodes, so the auxiliary target adds no information. The failures are execution errors, not perception.
- **Robustness:** no advantage under a real or synthetic sticker, or under a recoloured dot.
- **Limitations:**
  - one task, one training seed;
  - 20 rollouts per cell;
  - the lighting confound in the ID sessions;
  - 50% sticker pending.

### 6.1 Framing for the paper: instrumentation as a hardware-measured world model
`generic_c` jointly denoises future sensor readings with the action over a 1.6 s horizon. That makes it a world-action model whose "world" is a compressed, task-relevant state measured by hardware rather than predicted pixels. References from a short literature check of 2026-10-20 (verified on arXiv):

- **Closest prior work:** Touch Dreaming / HTD (arXiv 2604.13015). Future force and tactile prediction alongside action chunks, but the sensors are on the robot and are also inputs.
- **Same architecture with pixel targets:** PAD (2411.18179, NeurIPS 2024); UWM (2504.02792).
- **Compressed future targets:** FLARE (2505.15659); DreamVLA (2507.04447).
- **Training-only sensor:** BeadSight visuo-tactile pretraining (2406.15639); Learning by Cheating (1912.12294).
- **Test-time use of a predicted future:** FOREWARN (2502.01828); JEPA Policy (2609.09630, prediction error as a failure-ranking signal).

What looks new is the combination: an object-side task sensor, available only during training, predicted over the action horizon inside a Diffusion Policy, and possibly **used at test time** to rank sampled action chunks or detect stalls. This last part is untested.

### 6.2 Lessons for the next tasks
Before collecting data, ask of each task:
1. Is the instrumented quantity hidden from the camera?
2. Does the right action depend on it?
3. Run the linear probe on a low-data control: does it already learn the signal?

| Task | Instrumentation | Hidden from camera? | Action depends on it? | Expectation |
|---|---|---|---|---|
| Bottle cap | 3-channel cap sensor | No | Partly | No benefit (measured) |
| Plug insertion | contacts in the sheet close when seated | Mostly (mm of depth, occluded) | Partly (only if demos push until seated) | Most promising; risk: force spike or click may reveal seating |
| Petri dish rolling | load cell under the surface | Yes | Yes (force band is the task) | Promising if the band is narrow; risk: overlap with the wrist F/T |
| Wiping (conditional) | load cell through the cloth | Yes | Weakly | Overlaps petri dish; lowest priority |

## 7. Side thread: a simulation twin (ABC), audio in simulation, and the audio bug it uncovered

### 7.1 The ABC idea: sim-to-real evaluation correlation (2026-10-15, deferred)
- **Source:** the ABC paper (abc.bot, arXiv 2606.27375, github.com/amazon-far/abc). It evaluated the *same* checkpoints in simulation and in the real world and found high correlation between sim and real success (r ≈ 0.85–0.91).
- **The idea for us:** build a twin of the bottle setup (MuJoCo physics, wrist camera), evaluate the policies already evaluated on the robot in simulation with no retraining, and report Pearson, Spearman and MMRV correlation. If it correlated well, future evaluations could move partly to simulation.
- **Why it was deferred:**
  - In ABC, 8 policies were trained on their real dataset (ABC-130K) and 4 were co-trained with sim data; the paper presents co-training as part of how they reached that correlation.
  - Our policies have never seen simulated images **or simulated audio**. Out-of-distribution rendering and missing audio would likely dominate the result rather than measure the policies.
- **If revisited, decide first:**
  - how to provide audio (the policies take a 3 × 298 × 128 spectrogram);
  - how faithful the wrist camera must be (splat or ray-traced rendering);
  - whether co-training with sim data is acceptable.

  Use a separate environment: the aborted attempt's venv reached 12 GB.

### 7.2 Audio in simulation, and a neural spectrogram generator
No simulator (MuJoCo, Isaac Sim, Genesis) produces contact or friction sound usable as a policy input. Options considered, ranked:
1. **Retrieval:** look up a real recorded spectrogram by the nearest simulated state (contact force, cap angle, task phase).
2. **A neural spectrogram generator** (the user's idea): a conditional generative model trained on our own recordings, which produces spectrograms as close as possible to the recorded data, given the simulated state.
3. **Physics-based synthesis:** modal or friction sound models.

Caveat for options 1 and 2: audio generated from the simulated state can only echo the simulator's own physics errors; it cannot add information the simulation doesn't have.

### 7.3 How this uncovered the audio bug
- **The question:** how faithful does simulated audio need to be? To answer it, we first ran an **offline audio-sensitivity test** on recorded frames: replace the audio (silence, dataset mean, another episode's audio, a time-shifted clip) and measure how much the predicted actions change, compared with the run-to-run noise of the diffusion sampler.
- **The finding:** **every checkpoint trained up to then ignored audio completely**. Every audio perturbation changed the actions by ≤ 0.01× the noise. Only forcing the AST's internal input to zero moved them (0.4–2.6×). Images moved actions about 20–30×, the state about 0.3×.
- **The cause** (details in §5.1):
  - AudioSet's normalization constants were applied to our differently scaled spectrogram (`(fbank+20)/60` of an ADC-unit waveform), so the AST saw a nearly constant input;
  - lerobot's diffusion preset uses `torch.optim.Adam` with *coupled* weight decay, which then shrank the unused AST weights to about 0 (median 1e-38).
- **Comparison with ManiWAV** (real-stanford/maniwav): they train their AST from scratch and min-max normalize the spectrogram with dataset statistics; the original Diffusion Policy and ManiWAV both use AdamW.

### 7.4 The fix, and what it showed
- **Fix (2026-10-15), before the curve runs:**
  - per-level dataset statistics for the audio normalization;
  - AdamW with weight decay 1e-6.
  - We also added an optional ManiWAV-style modality-attention fusion (`generic_fusion_100`).
- **Verification:** after retraining, swapping the audio moves the actions **4–9× the noise** (about 2 mm), strongest during twisting (11× for concat, 14× for fusion). The effect holds equally on unseen episodes, so it isn't memorization. The fusion variant uses audio about 1.5× more than concatenation. Under the protocol's pre-registered rule the curve stayed on concatenation.
- **What audio carries:** the linear probe (§4.3b) shows audio features hold only a little cap-state information (R² 0.23–0.37, vs about 0.9 for images). The policy uses audio, but not as its main source of cap state.
- **Consequence for the simulation idea:** now that the policies *do* use audio, a simulation twin would need realistic audio. That makes retrieval or the neural spectrogram generator a required component if the twin is revisited. The swap test gives a target: simulated audio must be close enough that the actions move less than they do when real audio from another episode is swapped in (about 2 mm).
- **Files:** scripts and reports in `outputs/analysis/bottle_2026-10-19/audio_sensitivity/` (`REPORT.md` = pre-fix, `run_postfix.log` and `tables.md` = post-fix). Protocol §1.2 and §1.7 have the decision record.

## 8. Open items
- [ ] 50% sticker evaluation (both arms, alternating).
- [ ] Matched-lighting re-run at 50% ID (40 rollouts, arms alternated pose by pose).
- [ ] Pose 0 run as the third rollout of a session once.
- [ ] Decide on the gripper re-grasp in `eval_bottle` (§5.6).
- [ ] Add a "cap reads closed" pre-check and a frozen-camera abort to `eval_bottle`.
- [ ] Evaluate `generic_fusion_100` on the robot (optional appendix ablation).
- [ ] Decide whether to retrain on FT-corrected data (recommended: no; state it as a limitation).
- [ ] Commit the code changes (the lerobot fork's audio and fusion work, the FT and brightness changes, plot_curves).
