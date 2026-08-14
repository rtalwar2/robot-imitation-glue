# Instrumented Imitation Learning — Experiment Protocol

**Status:** supersedes `Task_Specifications.md` (8 July 2026)
**Date:** 12 August 2026
**Target:** RA-L
**Structure:** Part 1 is the methodology shared by all tasks. Part 2 is a per-task checklist. Part 3 lists what still needs the promotor. Part 4 lists code issues found along the way — items marked **FIXED** are applied and committed; the rest are open.

**Implementation** (committed; the pilot's tooling exists, the data does not yet):

| Script | Does |
|---|---|
| `ur5station/collect_data_bottle_opening.py` | collection entrypoint — wrist camera, audio, cap sensor, per-split pose seeds |
| `ur5station/prepare_datasets_bottle.py` | the 8 prepared datasets: {9,12}-dim action × 4 data levels |
| `ur5station/screen_instrumentation.py` | privilege gate + per-modality suitability (§1.4) |
| `ur5station/train_ast_bottle.py` | design-A audio pretraining (forked from `train_ast_single.py`, which is unchanged) |
| `ur5station/lerobot_train/bottle/generate_configs.py` | the 16 training configs, with a build-time arm-parity assertion |
| `agents/lerobot_agent.py` | `n_env_action_dims` — slices design C's auxiliary channels off before the robot |

Fork changes live in `lerobot` at `dd8ad224`: the AST position-embedding fix (§4.7) and the `rgb_encoder_init_checkpoint` / `audio_encoder_init_checkpoint` fields. Design C needs no fork change at all.

---

## What changed since `Task_Specifications.md`

| Item | Was | Now | Why |
|---|---|---|---|
| Mechanism | Encoder pretraining only | **Pretraining vs. auxiliary prediction, decided by the bottle pilot** | Colleague's workshop result (1 task, 1 seed, 100% data) suggests predicting the instrumentation alongside the action beats pretraining the encoder |
| Pilot task | Plugs | **Bottle** | Data collection is nearly built (`collect_data_bottle.py`) |
| Modality choice | Argmax over image vs. audio | **Proprioception privilege gate (mandatory)** + threshold over all modalities (pilot-mandatory, diagnostic after) | Argmax discards a modality that works; the privilege gate tests whether the instrumentation is privileged at all; under design C nothing branches on the modality score |
| Statistics | Per-task, n=10 per cell | **Pooled across tasks, stratified by task** | n=10 cannot support the primary claim |
| Wiping instrumentation | Vision grid coverage | **Force (load cell)** — coverage demoted to success criterion | Coverage index is redundant with proprioception under a non-revisiting snake path |
| Bottle generalization axis | 3D-printed variants (cap stiffness + appearance) | **Appearance only** — one bottle, one cap, different stickers | No printing capacity for variants; costs the dynamics-transfer half of the axis |
| Petri dish | 4 stages incl. lid open/close | **Lid pre-removed** — navigate + roll only | Confirmed: lid is off before the episode |
| Casting pieces, ziplock | Task 5 / deferred | **Both out of scope** | — |
| Instrumentation normalization | Sensor hardware range | **Calibrated covered/uncovered range** (see §1.7) | Hardware range compresses the useful signal to a fraction of its span |
| Design A starting point | Random init | **Generic (ImageNet/AudioSet), then instrumentation** | Holds generic pretraining constant between the generic and design-A arms, so their gap is attributable to the instrumentation stage |
| Encoder input normalization | Not specified | **Per arm, matched to each encoder's initialization** (see §1.7) | A pretrained encoder is (weights, expected input distribution); splitting the two handicaps the control arm |
| Step budget | Equalize total optimizer steps across arms | **Fixed 100K everywhere**, pretraining cost reported in text | The equalization is approximate anyway (encoder-only steps) and un-fixes the clean budget |

---

# Part 1 — General methodology

## 1.1 Research question

Does task-specific instrumentation — privileged sensor signals available during training but not at inference — improve the data efficiency of imitation learning?

**Hypothesis.** Supervising a policy with instrumentation signals acts as an inductive bias, teaching it to attend to task-relevant features. The same success rate is reached with fewer demonstrations than random initialization or generic pretraining achieve.

**Contribution boundary.** The colleague's workshop paper establishes that auxiliary instrumentation prediction helps at 100% data on one task, one seed. This paper's contribution is the **data-efficiency curves** and the **generalization split** — how the benefit scales as demonstrations are removed, and whether it survives out of distribution. The mechanism comparison in §1.3 is a controlled replication that fixes which design the rest of the paper uses, not the headline. The privilege test in §1.4 is a supporting methodological note: worth a subsection, not a claim.

## 1.2 Two candidate mechanisms

**Design A — encoder pretraining.** Take a generic-pretrained perception encoder (ImageNet/AudioSet), finetune it to predict the instrumentation signal, then use those weights to initialize the policy encoder and finetune everything. Implemented by `rgb_encoder_init_checkpoint` / `audio_encoder_init_checkpoint`, loaded strictly after the encoders are built.

**Design C — auxiliary prediction channels.** Append the instrumentation to the denoised output vector. Diffusion Policy predicts `[action (9), instrumentation (k)]` over the horizon; the instrumentation prediction is discarded at inference. Implemented entirely at the dataset level — a 12-dim `action` feature — because lerobot derives the denoised width from `config.action_feature.shape[0]`.

Design C is expected to be stronger because it is action-conditioned (the model learns what its chosen action chunk will *do* to the sensor, not just what the sensor reads now), it shapes the whole network rather than the encoder alone, and it is single-stage — which removes the "the treatment saw the data twice" objection entirely.

> An intermediate design B (auxiliary head hanging off the encoder, predicting the instrumentation at the current timestep) is **not** run. C subsumes what it was for, without the loss-weighting problem.

## 1.3 Variants

Four arms on the pilot task. Whichever mechanism wins is carried forward as a three-arm comparison on the remaining tasks.

| Variant | Encoder init | Output vector | Role |
|---|---|---|---|
| **From scratch** | random | action only | lower baseline |
| **Generic** | ImageNet / AudioSet | action only | is free pretraining enough? |
| **A: instrumentation-pretrained** | instrumentation | action only | mechanism A |
| **C: instrumentation channels** | random | action + instrumentation | mechanism C |

**Why the generic arm survives even if C wins.** Under C there is no pretraining phase, so generic pretraining is no longer the parallel control. It earns its place for a different and more important reason: ImageNet weights are free and universal, while instrumentation requires building custom hardware per task. If generic matches instrumentation, the hardware is not worth it and the paper has no contribution. That is the first question a reviewer asks.

All encoders are fine-tuned during policy training in every variant. Same architecture, same hyperparameters, same step budget. Only initialization and output dimensionality differ.

## 1.4 Screening — one mandatory gate, one conditional diagnostic

Two cheap offline tests, neither needing robot time. They have **different statuses**, and the difference matters: one decides whether a task happens at all, the other only decides something under design A.

### Step 1 — Privilege test — **mandatory, every task, before anything else**

Train a small MLP on **proprioception alone** (joints, TCP pose, gripper state) to predict the instrumentation signal on a held-out validation split.

> If proprioception predicts the instrumentation well, the robot already knows it. The signal is not privileged, the task is unsuitable, and neither mechanism can help — there is nothing to supervise on that the policy does not already observe.

This is a go/no-go on the task itself and is completely independent of which mechanism wins in §1.3. It is what disqualifies wiping-as-coverage: under a non-revisiting snake path the grid index is a deterministic function of end-effector position, so the MLP scores near-perfectly and the "instrumentation" is revealed as a redundant, noisier proprioception sensor.

Run it on every task, including any new task proposed later, before building hardware.

### Step 2 — Modality suitability — **mandatory for the pilot, diagnostic afterwards**

For each available modality — image (wrist camera), audio (AST spectrogram), force-torque (MLP over the 6-dim internal FT) — train an encoder to predict the instrumentation on the same 80/20 split.

**Why it is mandatory for the bottle pilot.** Design A requires choosing *which* encoder to pretrain. That choice must be made by a pre-registered rule rather than by intuition, or the one comparison the paper hangs on is open to the cherry-picking objection. The rule: every modality clearing the bar is pretrained; those below it are not — a threshold, not an argmax, so a modality that works is not discarded merely because another works slightly better.

The bar is **relative to a trivial baseline**, never an absolute percentage:

| Signal type | Metric | Bar |
|---|---|---|
| Binary (plug seated, cap open) | balanced accuracy or AUC | pre-registered margin above chance |
| Continuous (force, phototransistor) | R² | pre-registered margin above a mean-predictor |

A flat 70% does not travel across signal types: binary accuracy floors at 50%, R² floors at 0, a 25-way classification floors at 4%. Worse, the plug is unseated for most of every episode, so a constant "not seated" predictor may already clear 70% having learned nothing. Balanced accuracy and AUC are immune to that.

Read every score as *how much this modality adds over proprioception*, using step 1 as the floor.

**Why it demotes if design C wins.** Under C there is no encoder to select — the instrumentation is predicted from the fused representation of every modality present, and the generic arm simply loads ImageNet and AudioSet for all of them. No branch depends on the score, so the test stops being a gate. It remains worth running as a **results-section diagnostic**: it explains *why* the method works on a given task ("audio carries the seating click, image does not") and tells anyone reproducing the setup which sensors they actually need. Cheap, no robot time, but skippable under schedule pressure — the cost is an explanation, not a decision.

**Report both steps for every task attempted, including the ones that fail the gate.** Publishing the failures is what makes the protocol credible rather than post-hoc.

> ⚠️ **FT breaks three-variant parity.** There are no generic pretrained weights for a force-torque MLP — no ImageNet, no AudioSet. If FT is used as an instrumentation modality, its generic-pretrained control degenerates to random init. Either exclude FT from the variant comparison and run it as the standalone "can the internal FT predict the external load cell" experiment, or include it and state the asymmetry. **Needs promotor input.**

> **Not covered by either step:** whether a modality is worth carrying as a *policy input* at all (e.g. can the microphone come off the rig entirely). That is an input ablation on policy success rate, not a suitability question, and it is not in the critical path.

## 1.5 Data collection

**Scripted demonstrations for all tasks.** Chosen because exact force profiles are required (petri dish, wiping) and scripting makes them repeatable. All-or-none, so data-collection method is not confounded with task type.

**Diversity is parameterized, not incidental.** The same trajectory logic runs over a distribution of conditions: the bottle is held at a different pose every episode, plugs are reshuffled in the box, target locations and surfaces vary. In the bottle code this is `bottle_poses[n_recorded_episodes % len(bottle_poses)]` over 100 pre-generated reachable poses (`collect_data_bottle.py:227`).

**Instrumentation guides the scripting.** This is an advantage worth stating in the paper, not just an implementation detail. The bottle code already does it: `run_opening_motion_with_retry` gates each leg on a sensor checkpoint and, on failure, retracts and redoes the whole motion 3 mm deeper — up to three times (`collect_data_bottle.py:135-205`). The load cell can close a force loop the same way; circuit closure tells the plug script exactly when to stop.

**Retries stay in the demonstration.** A slipped grip and its recovery are recorded as ordinary steps. This is deliberate and worth a sentence in the paper: it gives the policy recovery behaviour that clean open-loop scripting would not.

**Successful episodes only**, filtered by the instrumentation at episode end. Expect to collect ~120 to keep ~100.

**Policy scope.** Scripted demonstrations ≠ scripted policy. Nothing is hardcoded into the policy — it learns the full task end to end from the recorded trajectories. State this explicitly in the paper; it is a predictable point of confusion.

## 1.6 Finding N

1. Collect a batch of 20 episodes.
2. Train the from-scratch policy on everything collected so far.
3. Evaluate with 20 rollouts.
4. Repeat until success ≥ 90%, or plateau (≤ 5% improvement over two consecutive batches), or a cap of 100 episodes.
5. N is then fixed for that task.

Data reduction levels: 100%, 75%, 50%, 25% of N. Uniform prefix after a fixed-seed shuffle. **Both pretraining and fine-tuning use the same reduced subset** — at 25%, everything sees only 25%. That is the honest data-efficiency test.

Report absolute episode counts alongside percentages ("25%, N=8"), since N will differ across tasks.

## 1.7 Training

- **Diffusion Policy** via lerobot, 10 Hz. Action representation: see below.
- **Fixed 100K steps for every variant.** The exact budget does not matter — what matters is that it is equal. A result that holds under an unoptimized-but-equal budget is a stronger result, not a weaker one.
- **Take the final checkpoint.** Not "select by rollout success" — that would mean real-robot rollouts on multiple checkpoints per config, silently multiplying the rollout budget. Fixed budget, final checkpoint, no selection.
- **Pretraining (design A only):** early stopping on validation loss, BCE for binary, MSE for continuous. LR is **per-modality**: 1e-4 for the resnet, 1e-5 for the AST (the paper convention for finetuning a pretrained transformer). Frame it in the paper as *weight initialization*, not extra training: the encoder is ~5-20% of total policy parameters and pretraining is <5% of its total optimization.
- **Design A starts from generic weights**, not random: ImageNet/AudioSet, *then* finetuned on the instrumentation signal. So generic pretraining is held constant between the generic and design-A arms, and the latter's gap over the former is attributable to the instrumentation stage.
- **No step equalization.** An earlier draft added design A's pretraining steps to the other arms to match total optimizer steps. Dropped: the equalization is approximate anyway (those are encoder-only steps, not full-policy steps) and adding them un-fixes the clean 100K budget. Report design A's pretraining cost in the text instead, under the weight-initialization framing above. If a reviewer presses, a step-equalized rerun of the from-scratch and design-A arms alone is cheap to add.

**Action representation: tool-frame deltas, kept.**

The policy predicts `[delta_xyz_tool(3), rot6d(R_delta)(6)]` — a translation offset in the **tool** frame and a relative rotation `R_delta = R_current^T · R_target`, applied at execution to the robot's live pose. Considered and rejected: absolute joint space (lerobot's default) and lerobot's `RelativeActionsProcessorStep` / `AbsoluteActionsProcessorStep`.

*The defense.* Tool-frame actions combined with a **wrist-mounted** camera make the policy equivariant to where the non-dominant arm presents the bottle: move the whole scene rigidly and both the correct action and the observed image are unchanged. The non-dominant arm presents the bottle at ~100 different poses, and the policy sees each as the same problem rather than a hundred separate ones. On a data-efficiency paper that equivariance is doing real work, which is also why **absolute joint space is the worse option here** — it would turn each presentation pose into a distinct configuration with no sharing between them.

*Why not lerobot's relative-action processors.* Four reasons, in order of weight:
1. It is orthogonal to the paper's claim. Action representation is a nuisance factor held identical across all four arms, so it moves absolute success rates but cannot affect the instrumentation comparison.
2. It would rewrite the eval path — the policy would emit absolute poses instead of deltas — which is the riskiest code to change immediately before collection.
3. Relative-to-chunk-start **cannot be precomputed per frame** (frame *t*'s action appears in up to `horizon` chunks with different reference poses), so it has to be a processor step, and `make_diffusion_pre_post_processors` has none — the machinery is wired for the pi family only.
4. `to_relative_actions` is elementwise subtraction, which cannot compose rotations: the rot6d dims would get a linear difference rather than a geometric delta. Invertible and therefore lossless, but not what the representation is supposed to mean.

*Proprioception stays in `observation.state`.* It entered this protocol as the §1.4 screening gate, not as a policy input — the policy input is a separate decision, and it is to keep it: with `n_obs_steps = 2`, pose[t-1] and pose[t] give velocity, which is not cleanly recoverable from two wrist frames and matters for a contact task; the petri-dish task feeds the policy base-frame target coordinates, so stripping pose here would make the pilot structurally different from the tasks it is pooled with; and it is identical across all four arms, so it cannot affect the comparison either way.

*Limitation to state in the paper.* This is UMI's "delta" category, and its objection applies within a chunk: action *k* is an offset from pose[t+k], which the policy never observes, so later actions assume the earlier ones executed as predicted. Two things blunt it — each action is applied to the robot's **live** pose rather than to an integrated prediction, and at `n_action_steps = 8` / 10 Hz the accumulation window is 0.8 s. The principled fix, if a reviewer presses, is to express all `horizon` actions as tool-frame offsets from the *chunk-start* pose: UMI-correct and equivariance-preserving, but it needs a custom processor step, since neither a dataset transform nor lerobot's elementwise version can do it.

*Revisitable.* Absolute target poses are recoverable from what is recorded — `policy_action_to_tcp_pose(robot_pose, action)` — so a relative-action arm can be run later on the same episodes as a clean A/B, without recollecting.

**Normalization of the instrumentation signal.**

*Design C:* normalize the instrumentation channels with the **same normalizer lerobot applies to the action dims** (dataset mean/std). Mixed scales inside one denoised vector give the network badly conditioned inputs even though the loss is fine.

*Design A:* normalize to each channel's **measured operating range**, not the raw hardware range. Against a 0–3.3 V ADC span, the bottle sensor only ever traverses S0 2.96–3.25 V, S1 2.48–3.29 V, S2 2.08–3.25 V (global min/max over `sensor_logs/run_{0003,0005,0006}.json`, the runs under the current sensor layout — runs 0000–0002 used a different layout and are not comparable). Dividing by 3.3 V would compress the entire useful signal into a fraction of the range. These ranges come from separate calibration runs rather than the demonstration set, and are fixed before training and identical across all reduction levels, so there is no leakage. Re-derive alongside `PER_CHANNEL_THRESHOLDS` whenever the cap, sensor mounting or motion geometry changes.

**Auxiliary loss weighting (design C) is nearly free.** With `prediction_type=epsilon` every channel's regression target is the sampled noise ε ~ N(0, I), so all channels sit on the same loss scale regardless of what the underlying quantity is. With mean reduction over 12 channels, the 3 instrumentation dims take 3/12 = 25% of the objective automatically. No λ sweep, no gradient-norm matching. Just be deliberate that channel count sets the weight: three phototransistors give the instrumentation 25%, one gives it 10%.

Accept and state one consequence: the *action* term is correspondingly scaled 9/12 = 0.75× relative to the action-only arms. That is inherent to the design rather than a bug, it is what the colleague's workshop result already did, and isolating it would need a non-standard loss patch.

**Normalization of the encoder *inputs* is matched to each encoder's initialization.** Distinct from the instrumentation-target normalization above, and it is not a parity violation — a pretrained encoder is (weights, expected input distribution), and splitting the two handicaps it. Mis-normalizing the generic arm to keep configs superficially identical would weaken exactly the control that has to be strong for the "is the custom hardware worth building" argument.

| Arm | image (`dataset.use_imagenet_stats`) | audio (`audio_norm_mean/std`) |
|---|---|---|
| from-scratch | dataset (`false`) | dataset-computed |
| generic | **ImageNet (`true`)** | **AudioSet: −4.2677393 / 4.5689974** |
| design A | dataset (`false`) | dataset-computed |
| design C | dataset (`false`) | dataset-computed |

Design A takes *dataset* stats despite starting from generic weights: the instrumentation stage is where its encoder last saw data, so it adapts to whatever normalization that stage used and the generic starting point is re-adapted away. Pretrain and deploy under the same stats. The random-init arms have no prior expectation, so dataset stats are simply the well-conditioned default.

`audio_norm_mean`/`audio_norm_std` default to `0.0`/`1.0` — i.e. **no normalization at all** — so every arm must set them explicitly. The pretraining script records the values it used in the checkpoint so the arm config can be asserted against them.

Framing for the paper: *only the encoder initialization and its matched input normalization differ.* Still controlled, because normalization is a deterministic consequence of the chosen init rather than a tuned knob. One caveat to state: the generic and design-A arms therefore differ in normalization as well as in the instrumentation stage, so their difference is not a pristine isolation of that stage. Cheap insurance in reserve: rerun the generic arm with dataset normalization at 100% data only.

## 1.8 Evaluation

- **20 real-world rollouts per configuration**: 10 in-distribution + 10 out-of-distribution.
- **Success rate** is the only metric. No secondary metrics.
- Rollout timeout: 2× the median demonstration duration.
- OOD varies **only** the generalization axis. Every other condition — including the bottle-holding arm's pose distribution — is drawn from the training distribution, or an OOD failure cannot be attributed to the axis under test.

**Statistics: pool across tasks, stratify by task.**

Ten rollouts of one trained policy are ten samples of *that policy*, not of *the method* — the training run is the real experimental unit. Pooling across tasks supplies genuinely independent replicates, so this is a correctness improvement as well as a power one.

Use **Cochran–Mantel–Haenszel** for pairwise variant comparisons at each data level, stratified by task, or a GLMM with variant and data level as fixed effects and task as a random effect. Do not simply concatenate the 2×2 tables — that invites Simpson's paradox if one task's effect runs the other way.

| Rollouts per cell | 95% CI half-width at p≈0.5 | 50% vs 80% detectable? |
|---|---|---|
| 10 (one task) | ±28 pp | no, p ≈ 0.35 |
| 30 (3 tasks) | ±17 pp | yes, p ≈ 0.03 |
| 50 (5 tasks) | ±13 pp | comfortably |

**Pre-register the pooled analysis as primary and per-task curves as descriptive.** Deciding to pool after seeing per-task results is the thing reviewers punish.

## 1.9 Reporting

Two distinct claims from the same rollouts. Never conflate them:

- **Figure A — data efficiency:** success rate vs. % training data, ID rollouts only, pooled across tasks.
- **Figure B — generalization:** ID vs. OOD success at each data level.
- **Table 1 — screening:** per task, the proprioception privilege-test score for every task attempted (including those that failed the gate), plus modality suitability scores wherever they were run — mandatory for the pilot, diagnostic elsewhere.
- **Table 2 — mechanism (pilot only):** design A vs. design C at every data level on the bottle.

---

# Part 2 — Per-task checklists

## Task 1 — Bottle opening (pilot)

The only task running all four variants. It decides the mechanism for everything else.

**Setup.** Left UR5 + Schunk gripper opens a flick-switch cap; right UR5 holds the bottle at a different pose each episode. Cap pose is recomputed live from the right arm's TCP (`bottle_station_env.py:47-50`), which is what makes the opening motion scriptable.

**Instrumentation.** Three phototransistor channels inside the cap, published over DDS on topic `Bottle`, read into `obs["bottle_sensor"]` (`bottle_sensor.py:36-53`). Continuous, 3-dim.

**Success.** Each channel passes its own checkpoint at its own stage of the motion: S0 at `push_end`, S1 at `leg_3_end`, S2 at `leg_6_end` (`SENSOR_CHECKPOINTS`). A failed checkpoint triggers a retry, so an episode only succeeds once all three have passed — but at their respective moments, not simultaneously at the end. Failed episodes are deleted during collection rather than saved and filtered later.

> ⚠️ This rests on an assumption worth testing: that a channel does not re-cover once uncovered. In `run_0006` all three channels produce readings above their thresholds that later fall back below, before the final sustained transition — and that run contains no retry, so it is not retraction-induced. Whether it would cause a false checkpoint pass depends on whether an excursion coincides with a checkpoint evaluation, which has not been checked. Worth resolving before large-scale collection, since it affects the success label itself and not just a downstream metric.

**Generalization axis — appearance only.** There is one physical bottle and one cap; no variants are 3D printed. OOD is produced by changing the bottle's **visual appearance** — stickers, tape, patterns, matte vs. glossy — while the mechanism stays identical. Train on one set of appearances, evaluate OOD on unseen ones.

This is a narrower axis than the stiffness-plus-appearance split previously planned, and the paper should call it *appearance robustness* rather than generalization unqualified: cap dynamics no longer vary, so nothing here tests dynamics transfer. It is, however, tightly matched to the hypothesis — if instrumentation supervision really teaches the encoder to attend to cap state rather than incidental visual structure, it should be measurably less disturbed by appearance changes than the from-scratch variant is. Make the shift substantial (colour, pattern, coverage), not a single small sticker.

> ⚠️ **Two conditions this axis depends on. Verify both before collecting.**
> 1. **Stickers must stay clear of the cap's light path.** The phototransistors measure light inside the cap. If a sticker changes what reaches them, `PER_CHANNEL_THRESHOLDS` no longer holds and the *success criterion itself* differs between ID and OOD — the two conditions would be scored with different rulers. Keep appearance changes on the bottle body, and re-read the sensor on a fully-open and fully-closed cap for every sticker configuration to confirm the voltages have not moved.
> 2. **The appearance change must be visible to the wrist camera.** The wrist cam looks down at the cap; if the bottle body is mostly out of frame, ID and OOD observations are identical, there is no distribution shift, and OOD success trivially equals ID success. Check recorded `wrist_image` frames for how much body is in view before committing to sticker placement.

*Optional second axis, free:* hold out a region of the right-arm pose space instead of sampling OOD poses from the training distribution. No hardware needed. Do not split the 10 OOD rollouts across two axes — too thin. Appearance is primary; keep pose-holdout in reserve.

### Hardware and setup

- [x] ~~Remove the scene camera before recording a single episode.~~ **Done** — publisher, subscriber, factory method, topic constants and the `scene_image` / `scene_image_original` obs keys are gone; the ZED setup is recoverable from git if a later task needs an overhead camera. Wrist camera only.
- [x] ~~Wire audio in before collection.~~ **Done** — `SpectrogramSubscriberKaldi` behind a new `with_spectogram` flag on `UR5eStation`, emitting `spectogram_image` and `spectogram_values`; `collect_data_bottle_opening.py` passes `with_spectogram=True`. Both this and the scene-camera removal change the recorded obs keys, which define the LeRobot feature schema (`dataset_recorder.py:163-188`), so neither is recoverable after episodes exist.

**Modalities for the pilot:** wrist image, audio (AST), and `observation.state` = TCP pose (6) ⊕ internal FT (6). FT rides inside the state vector because `observation.state` is the only state key the policy consumes — any other `observation.*` vector is typed but never reaches `global_cond`.
- [x] ~~Verify the wrist RealSense starts at 720p, not the 480p fallback.~~ **Done** — the fallback is removed; `create_wrist_camera` requests 720p and raises if that profile will not start (§4.6). Confirm the hardware actually supports it on the first run, since the D405 note in `ipc_camera.py:248` suggests it may not.
- [ ] Fix the appearance set: how many sticker configurations, and which are train vs. OOD. No printing needed.
- [ ] Verify both conditions in the generalization-axis warning above — sensor voltages unchanged by stickers, and stickers visible in the wrist frame.
- [ ] Calibrate `PER_CHANNEL_THRESHOLDS` once for this cap and re-confirm after the stickers go on (`bottle_sensor.py:14-19`). With a single cap there is no per-variant recalibration — but there *is* a per-appearance sanity check, per the warning above.

### Data collection

- [x] ~~Fix the success-flag bug.~~ **Done** (§4.1). Episodes recorded *before* the fix are all labelled `success=False` regardless of outcome — discard or re-label them from the sensor logs before they enter any dataset.
- [x] ~~Use a different RNG seed for val/test poses.~~ **Done** (§4.3) — per-split offsets.
- [ ] Confirm the pose distribution for OOD rollouts matches training — only the appearance changes.
- [ ] Run the N-finding loop: batches of 20, retrain from-scratch, 20 rollouts, stop at 90% or plateau, cap 100.

### Screening

- [ ] **Privilege test** (mandatory). Proprioception MLP → 3-channel sensor. Expect it to fail the gate (cap rotation depends on grip slip and thread engagement, not just wrist angle) — but confirm it, because a scripted motion makes proprioception unusually informative.
- [ ] **Modality suitability** (mandatory here — design A needs it to pick which encoder to pretrain). Wrist image, audio (AST), FT. Record all scores and fix the threshold *before* looking at them.
- [ ] Both tests run from `ur5station/screen_instrumentation.py`, which builds the *policy's own* encoder classes from the arm's config, so the resulting weights load into the policy with `strict=True` and no key remapping. Audio pretraining for design A proper is `ur5station/train_ast_bottle.py`.

### Training and the mechanism comparison

- [x] ~~Confirm design C needs no collection-code change.~~ **Confirmed and implemented.** `ur5station/prepare_datasets_bottle.py` emits a 12-dim `action` = `[action(9), bottle_sensor(3)]`; lerobot reads the denoised width from `config.action_feature.shape[0]`, so the U-Net, sampling prior and MIN_MAX normalizer all widen with no policy change. Verified: both widths build, train and sample, +10,755 params (0.004%).
- [x] ~~Add the inference slice.~~ **Done** — `LerobotAgent(n_env_action_dims=9)` truncates after the postprocessor. Pass `9` for **all four arms** (a no-op for three of them) so the eval path is byte-identical across arms.
- [ ] Build the 8 prepared datasets (`prepare_datasets_bottle.py`) once collection and success-filtering are complete.
- [ ] Generate the 16 configs (`lerobot_train/bottle/generate_configs.py`) — it asserts at build time that the arms differ only in intended keys. Requires the real `audio_norm_mean/std` from pretraining; it has no safe default.
- [ ] Train 4 arms × 4 data levels = 16 runs, 100K steps each, final checkpoint.
- [ ] 16 configs × 20 rollouts = **320 rollouts** for the pilot.
- [ ] Decide the mechanism from Table 2, then carry the winner forward.

## Task 2 — Rubber plug insertion

**Setup.** Six plug sizes, all black, cluttered in a box on an instrumented metal sheet. Robot picks a plug and seats it in the matching hole.
**Instrumentation.** Button-like contacts in the sheet — binary circuit closure.
**Success.** Circuit closed.
**Generalization axis.** Plug size: train 1/3/5, OOD 2/4/6.

- [ ] Build the instrumented sheet — contacts around each hole plus wiring. Still blocked on the promotor for the physical sheet.
- [ ] Confirm the cluttered-box protocol: all six sizes in the box simultaneously (industry-realistic), or one size presented per trial?
- [ ] Confirm box dimensions and whether plugs are reshuffled between trials.
- [ ] Privilege test (mandatory): expect a clear pass — the gripper can be at the correct pose with the plug unseated, so proprioception cannot predict seating.
- [ ] Modality suitability (mandatory if the pilot selects design A; diagnostic if it selects C): audio is the hypothesis, via the seating click. Use **balanced accuracy or AUC**, not raw accuracy — the signal is near-zero for most of every episode.
- [ ] Script the insertion with circuit closure as the episode-termination signal.

## Task 3 — Petri dish rolling

**Setup.** Press an agar petri dish in a rolling motion over a target. Lid is already off — the task is navigate + roll, two stages, not four.
**Instrumentation.** Load cell under the sampling surface, continuous.
**Success.** Contact force within [F_min, F_max] for a sufficient duration.
**Generalization axis.** Surface material: train on tabletop / metal table / white shelf wood, OOD on wood board with holes / plastic plate / glass.

- [ ] Determine [F_min, F_max] empirically from demonstrations (mean ± 1 SD of scripted force).
- [ ] **Confirm the force tolerance is genuinely narrow.** If any force in a wide band succeeds, the task is too easy to be worth imitation learning. Open since 30 June, still unanswered, and it decides whether this task ships.
- [ ] Privilege test (mandatory): expect a pass — contact force is not in joint angles.
- [ ] Modality suitability (mandatory if the pilot selects design A; diagnostic if it selects C — but run it here regardless, it answers a question the paper wants). Audio is the hypothesis: friction and contact sound. This is where the **visually-identical-materials** question gets answered: white painted wood and a white plastic plate look the same to the camera but differ acoustically (wood duller, 200-800 Hz; plastic sharper, >1 kHz). If audio can predict force with visual input held constant, that is a result worth reporting on its own.
- [ ] Coordinate input: 3D point + surface normal, passed to the policy.
- [ ] Optional side experiment: FT encoder → predict the external load cell. Standalone, not part of the variant comparison (see the parity caveat in §1.4).

## Task 4 — Wiping (conditional)

**Runs only if the force reformulation holds.** As previously specified — an overhead camera reporting the next grid index in a snake sequence — this task fails the privilege test by construction: with a non-revisiting path the index is a deterministic function of end-effector position, and you confirmed the wiping leaves no visible trace. There is nothing privileged for a camera to sense.

- [ ] Re-specify the instrumentation as **contact force through the cloth** (load cell), which is not in proprioception and *does* vary with surface material — unlike the coverage geometry, which is identical over glass and wood.
- [ ] Keep grid coverage as the **success criterion only** — computed offline by the overhead camera after the episode. No memory required in the policy for that, and it was never the problem.
- [ ] Run the privilege test on the force signal before building anything — the coverage version already failed it, so this is the gate that decides whether the task exists.
- [ ] Generalization axis: same six surfaces as the petri dish.
- [ ] Cloth extraction from the box is part of the demonstration and learned by the policy.

## Out of scope

**Ziplock bag.** Deferred. The promotor's suggestion — disconnected square contact pads, more pads bridged = more closed — is recorded here so it is not lost, but it is not in this paper. Bimanual, deformable, and needs a framework extension.

**Casting pieces (Company ABC).** Deferred. Precision seating with a distance sensor is a different problem from trajectory learning and dilutes the narrative.

---

# Part 3 — Open items for the promotor

1. ~~**FT parity.** No generic pretrained weights exist for a force-torque MLP.~~ **Largely resolved by implementation.** FT is concatenated into `observation.state` rather than given its own encoder, because `observation.state` is the only state key the policy consumes. So there is no FT encoder to initialize and no parity asymmetry in the variant comparison. What remains is a reporting point rather than a design question: FT is not separately *encoded*, so per-modality attribution for it comes from the offline screening runs, not from the policy. Still worth confirming the promotor is happy with that framing.
2. ~~**Bottle cap mechanism.** Who designs the spring-based 3D print?~~ **Closed** — no variants are printed. One bottle, one cap; generalization comes from appearance changes instead. Note the cost: the axis no longer tests dynamics transfer, only appearance robustness.
3. **Instrumented metal sheet.** Need the physical sheet and the wiring worked out. Blocks task 2 entirely.
4. **Plugs per trial.** All six sizes cluttered in the box at once, or one per trial?
5. **Wiping force tolerance.** Is there a real, narrow tolerance? If not, drop the task.
6. **Relationship to the colleague's workshop paper.** The mechanism (predicting the instrumentation alongside the action) is already published there, at 100% data on one task and one seed. This paper's contribution is therefore the **data-efficiency curves** and the **generalization split** — the privilege test is a supporting methodological note, not a claim. Confirm that framing and confirm authorship overlap.
7. **Task count.** Bottle + plugs + petri dish is three. Pooled statistics gets meaningfully stronger with a fourth (n=30 → n=40 per cell). Is wiping worth rescuing for that reason alone, or is three enough?

---

# Part 4 — Code issues found

Found while reading the bottle collection path. The first one is a correctness bug that will silently corrupt the dataset.

### 4.1 `episode_success` is never set — every episode is labelled a failure — **FIXED**

`collect_data_bottle.py:188` gated success on `event_name == "leg_5_end"`, but `SENSOR_CHECKPOINTS` defines `leg_3_end` and `leg_6_end` (`bottle_sensor.py:25-28`). `"leg_5_end"` is never generated by `f"leg_{leg_index + 2}_end"` *and* present in the dict, so the branch never fired. `episode_success` stayed `False` for every episode, and `run_opening_motion_with_retry` always returned `False`.

Consequences: every frame written with `next.success=False`, the "successful demonstrations only" filter with nothing to filter on, and the N-finding loop with no success signal. The checkpoint had been renamed 5 → 6 in the dict without updating either use. Changed to `leg_6_end` in the branch and both docstring references.

**Any episodes already recorded before this fix carry `next.success=False` regardless of outcome and cannot be filtered — re-record or re-label them from the sensor logs.**

### 4.2 Success flag lags one leg behind — open, low severity

`servo_to_waypoint(..., success=episode_success)` at `collect_data_bottle.py:176-178` passes the flag as it stood *before* the current leg ran, so the final leg's frames — the ones where the cap actually pops open — are still written with `success=False`.

Severity is low because the lift-off segment after the loop *does* receive the correct flag, so episode-level filtering ("does any frame in this episode carry `success=True`") works. It only matters if anything downstream reads `next.success` per frame. Fix by setting the flag on the episode at save time rather than per-step.

### 4.3 Train and val bottle poses are identical — **FIXED**

`collect_data_bottle_opening.py` created one `np.random.default_rng(RANDOM_SEED)` and drew from the same stream for both splits, so val's 25 poses were train's first 25. Now seeded per split via `SPLIT_SEED_OFFSETS` (`train` +0, `val` +1, `test` +2).

### 4.4 `ft` is unbound when `use_internal_ft=False`

`ur5_robot_env.py:272-275`: the `else` branch that would read `self.ft_subscriber` is commented out, but line 301 unconditionally puts `ft` in the obs dict. `NameError` on that path. Harmless today because the entrypoint passes `True`, but it will bite the moment the external FT is wired up for the petri dish.

### 4.5 Naming collision on "instrumentation"

`UR5eStation(with_instrumentation=...)` refers to the *button* subscriber from the earlier button-pressing work, while the bottle's instrumentation arrives unconditionally through `BottleStation`. The bottle entrypoint passes `with_instrumentation=False` while collecting instrumentation data. Worth renaming to `with_button` before this confuses someone reading the paper's code release.

### 4.6 RealSense 480p fallback removed — **FIXED**

`create_wrist_camera` tried 720p and silently stepped down to 480p if it failed (`ur5_robot_env.py:64`). `touch_point_detector`'s radii are calibrated on 1280×720, so a fallback would have skewed touch-point detection for an entire recording session with only a warning. It now requests 720p and raises. The runtime shape check at `collect_data_bottle.py:250` is kept as a downstream-resize guard, with its message updated.

Note `ipc_camera.py:250` has a *separate* `CameraFactory.create_camera` pinned to 480p, with a comment that the D405 does not support 1080p. It is referenced only from that file's `__main__`, so the bottle rig is unaffected — but it hints that 720p may not be available on this camera, in which case the new code will raise rather than degrade. Test once before a collection session.

### 4.7 AST position embeddings were silently randomized — **FIXED**

`DiffusionAudioEncoder` sets `hf_config.max_length = time_dimension` (298) while the AudioSet checkpoint is trained at 1024 frames. That changes the patch count, so `position_embeddings` mismatches — `(1, 1214, 768)` vs `(1, 350, 768)` — and `ignore_mismatched_sizes=True` does **not** rescale them as the old comment claimed. It re-initialized 268,800 parameters at random and then fed AudioSet-pretrained transformer blocks position encodings they had never seen. The generic arm, which has to be strong for the "is the hardware worth building" argument, was silently crippled.

Now the checkpoint is loaded at its native length and the embeddings are resized before transfer: a centre slice when shrinking, interpolation only when growing. The slice is deliberate — the embeddings encode absolute position at a fixed 10 ms patch stride, and our clips use that same stride, so a contiguous slice preserves the time scale where interpolation would compress 10 s of structure into 3 s. Matches the reference AST implementation. Verified against the real checkpoint: **198/199 tensors transfer byte-identically, `position_embeddings` is the only resized tensor, and nothing is left at random init.** `_load_pretrained_ast` now raises on any unmatched key instead of degrading quietly.

The same bug was in `train_ast_single.py:90` (with the same incorrect comment), so the button experiment's AST also pretrained from AudioSet blocks plus random position embeddings. It trained for 10 epochs afterwards, so the *output* checkpoint has learned embeddings — but it started from a worse initialization than intended. `train_ast_bottle.py` inherits the fix by building the policy's encoder class.

### 4.8 The spectrogram is video-compressed before the AST sees it — open

`spectogram_values` is a 3-D array, so `dataset_recorder.py:171-177` classifies it as a **video** feature. The mel spectrogram the AST consumes is therefore H.264-encoded and float-quantized to uint8 — lossy compression applied to one of the modalities under evaluation.

Pre-existing (the button experiment's results went through it too), so it was left alone rather than changed unilaterally, but it deserves a deliberate decision before the audio arms are taken seriously. The lossless option is storing the spectrogram flattened as a 1-D float32 vector, which makes it a state feature; that would need a small change to the fork's audio branch to reshape on the way in.
