# Instrumented Imitation Learning — Experiment Protocol

**Status:** supersedes `Task_Specifications.md` (8 July 2026)
**Date:** 12 August 2026 · **Revised:** 2 October 2026 on branch `ral_simplified` (see the revision block below)
**Target:** RA-L
**Structure:** Part 1 is the methodology shared by all tasks. Part 2 is a per-task checklist. Part 3 lists what still needs the promotor. Part 4 lists code issues found along the way — items marked **FIXED** are applied and committed; the rest are open.

## Revision — 2 October 2026: simplified arm set (`ral_simplified`)

The four-arm × four-level matrix (16 runs, 320+ rollouts) is replaced by a **two-arm × four-level
curve plus a one-time initialization-verification cell**:

| Arm | Role | Encoder init | Denoised vector |
|---|---|---|---|
| `generic` | **control** | ImageNet/AudioSet | action(9) |
| `generic_c` | **treatment** | ImageNet/AudioSet | action(9) ⊕ instrumentation(3) |
| ~~`from_scratch`~~ | ~~init verification, 100% level only~~ **dropped 2026-10-14, not run** | random | action(9) |

Rationale (design-A/B mechanism comparison retired; the reasoning lives in §1.1–§1.3):

- **Single factor.** The workshop paper's C arm differed from its baselines in *two* things at
  once — random init AND auxiliary channels. With both arms on the standard pretrained init,
  `generic_c − generic` isolates sensor supervision exactly; nothing else differs.
- **Design A (instrumentation-pretrained encoders) is retired** — the A-vs-C comparison was a
  replication of the workshop mechanism, not the headline, and its cost was four extra trainings,
  per-level encoder pretraining, and a 160-rollout mechanism table.
- **A ManiWAV-style input-fusion transformer (image × audio cross-attention) was considered as
  the default architecture and declined** — explicitly *not* on invariant grounds (fusing two
  deploy-available modalities creates no sensor reliance); on keeping-the-control-off-the-shelf
  grounds. Deferred to a possible follow-up ablation. Full reasoning: §1.2. **Update 2026-10-15:** that ablation is
  now running at 100% only (`generic_fusion_100`), in parallel with the concat curve; see §1.2.
- **Pretrained init is the assumed starting point, and the assumption is checked, not trusted.**
  Both arms use it because it is the practitioner standard — the original Diffusion Policy code
  initializes its ResNet-18 with ImageNet weights, and lerobot defaults ACT's backbone to
  `ResNet18_Weights.IMAGENET1K_V1` (its DiffusionPolicy dataclass defaults to `None`, so this
  repo's configs set it explicitly, as lerobot's own ACT configs do) — which makes the control as
  strong as off-the-shelf practice without us tuning it. But a single from-scratch run at 100%
  verifies the assumption, because for *contact audio* it is genuinely unproven (ManiWAV preferred
  from-scratch ASTs on contact-rich audio, albeit confounded with architecture).
  **Not run (decision 2026-10-14):** the cell below is dropped and the init assumption stays
  unverified; the paper must not claim it.
  Pre-registered one-sided rule: `from_scratch` is **dropped from the design — reported as the
  appendix init-verification row — only if it loses to `generic` at 100% ID** (strictly fewer
  successes). A **tie or a win promotes it to a live third arm** (its remaining levels get
  generated and it joins the curve budget): a tie means pretrained init was never shown to help
  this task family, a win means it actively misfits contact audio — either is a finding in its own
  right, since the whole design leans on that assumption.
- **Rebaseline.** The cap-sensor recalibration is **done (2026-10-07)**: the opening motion was
  re-tuned leg by leg, thresholds and `CALIBRATED_RANGE` were re-derived from the calibration
  batch (run_0013/0018/0020–0023) with `bottle_experiment/derive_thresholds.py`, and
  `SENSOR_CHECKPOINTS` was re-mapped to where the channels actually cross (§1.7); both were
  re-derived once more on 2026-10-09 for the 6-leg motion (3.20/3.12/3.10 V, S1 at `leg_3_end`,
  S2 at `leg_7_end`, S0 logged only). **The recollection is done** (90 expert episodes
  2026-10-09/10, 20 recovery 2026-10-13/14) and **N = 70 = 50 expert + 20 recovery** (Task 1,
  N-finding log). The screening scores were computed on the pre-revision 51-episode set and have
  not been re-run on the final data. The evaluation protocol also changed: the
  ID/OOD sticker confirmation prompt is removed (the condition is a recorded label, not a
  confirmed fact), and the operator can stop a rollout mid-run with Enter and record a visual
  verdict (see §1.8).

**Implementation** (the pilot's tooling; the curve campaign's later changes — recovery collector, relative actions, audio fix, `plot_curves.py` — are not all committed yet):

| Script | Does |
|---|---|
| `ur5station/bottle/collect_data_bottle_opening.py` | collection entrypoint — wrist camera, audio, cap sensor, per-split pose seeds |
| `ur5station/bottle/prepare_datasets_bottle.py` | the 8 prepared datasets: {9,12}-dim action × 4 data levels |
| `ur5station/bottle/screen_instrumentation.py` | privilege gate (mandatory) + per-modality suitability (§1.4, now diagnostic-only) |
| `ur5station/bottle/generate_configs.py` | the 8 curve training configs, with a build-time arm-parity assertion |
| `agents/lerobot_agent.py` | `n_env_action_dims` — slices `generic_c`'s auxiliary channels off before the robot |

`train_ast_bottle.py` (design-A audio pretraining) and the `*_encoder_init_checkpoint` config
fields are retired with design A; they stay on disk, unused, for the button-experiment lineage.
Fork changes live in `lerobot` at `dd8ad224`: the AST position-embedding fix (§4.7) and the
retired checkpoint-init fields. The mechanism needs no fork change at all — lerobot derives the
denoised width from the dataset.

---

## What changed since `Task_Specifications.md`

| Item | Was | Now | Why |
|---|---|---|---|
| Mechanism | Encoder pretraining only | ~~Pretraining vs. auxiliary prediction, decided by the bottle pilot~~ → **auxiliary prediction only** (revised 2026-10; the A-vs-C comparison was retired — see revision block) | The workshop result plus the single-factor argument make C the chosen mechanism; re-deciding it cost four trainings and 160 rollouts |
| Pilot task | Plugs | **Bottle** | Data collection is nearly built (`collect_data_bottle.py`) |
| Modality choice | Argmax over image vs. audio | **Proprioception privilege gate (mandatory)** + threshold over all modalities (**diagnostic-only since 2026-10**) | Argmax discards a modality that works; the privilege gate tests whether the instrumentation is privileged at all; with design A retired nothing branches on the modality score |
| Statistics | Per-task, n=10 per cell | **Pooled across tasks, stratified by task** | n=10 cannot support the primary claim |
| Wiping instrumentation | Vision grid coverage | **Force (load cell)** — coverage demoted to success criterion | Coverage index is redundant with proprioception under a non-revisiting snake path |
| Bottle generalization axis | 3D-printed variants (cap stiffness + appearance) | **Appearance only** — one bottle, one cap, different stickers | No printing capacity for variants; costs the dynamics-transfer half of the axis |
| Petri dish | 4 stages incl. lid open/close | **Lid pre-removed** — navigate + roll only | Confirmed: lid is off before the episode |
| Casting pieces, ziplock | Task 5 / deferred | **Both out of scope** | — |
| Instrumentation normalization | Sensor hardware range | **Calibrated covered/uncovered range** (see §1.7) | Hardware range compresses the useful signal to a fraction of its span |
| Design A starting point | Random init | ~~Generic (ImageNet/AudioSet), then instrumentation~~ **Design A retired 2026-10** (see revision block) | The instrumentation-pretraining stage cost the most in the matrix and bought a replication, not the headline |
| Encoder input normalization | Not specified | **Per arm, matched to each encoder's initialization** (see §1.7) | A pretrained encoder is (weights, expected input distribution); splitting the two handicaps the control arm |
| Step budget | Equalize total optimizer steps across arms | **Fixed 150K everywhere** (100K until 2026-10-14), pretraining cost reported in text | The equalization is approximate anyway (encoder-only steps) and un-fixes the clean budget |

---

# Part 1 — General methodology

## 1.1 Research question

Does task-specific instrumentation — privileged sensor signals available during training but not at inference — improve the data efficiency of imitation learning?

**Hypothesis.** Supervising a policy with the instrumentation signal acts as an inductive bias, teaching it to attend to task-relevant features. The same success rate is reached with fewer demonstrations than a matched control — identical architecture, identical initialization, no instrumentation — achieves.

**Contribution boundary.** The colleague's workshop paper establishes that auxiliary instrumentation prediction helps at 100% data on one task, one seed — but their treated arm differed from its baselines in two factors at once (random initialization *and* the auxiliary channels). This paper (1) re-runs the mechanism **de-confounded**: both arms start from the same standard pretrained initialization, so the gap is attributable to the instrumentation alone, and (2) makes the headline the **data-efficiency curves** — how the benefit scales as demonstrations are removed — plus the **appearance-generalization split**, whether it survives out of distribution. The privilege test in §1.4 is a supporting methodological note: worth a subsection, not a claim.

## 1.2 The mechanism: auxiliary prediction channels

**Design C — auxiliary prediction channels. THE RUN MECHANISM.** Append the instrumentation to the denoised output vector. Diffusion Policy predicts `[action (9), instrumentation (k)]` over the horizon; the instrumentation prediction is discarded at inference. Implemented entirely at the dataset level — a 12-dim `action` feature — because lerobot derives the denoised width from `config.action_feature.shape[0]`.

The instrumentation is **never a policy input**: it appears only as extra dimensions the diffusion head denoises and inference throws away, so the deployed policy needs no sensor hardware. This is the paper's "no reliance" property, structural rather than aspirational — a policy cannot shortcut through a modality it cannot read at test time, and the auxiliary loss is the only path by which the signal can shape behaviour. It is also why C was preferred over A on principle, before any comparison was run: C is action-conditioned (the model learns what its chosen action chunk will *do* to the sensor, not just what the sensor reads now), it shapes the whole network rather than the encoder alone, and it is single-stage — which removes the "the treatment saw the data twice" objection entirely.

**A ManiWAV-style input-fusion transformer is not part of this design — and not because of the invariant.** Fusing the image and audio encoder features with cross-attention on the *input* side (as ManiWAV does before its action head) would involve no privileged signal at deployment: wrist image and microphone audio are both physically present on the rig, so the never-an-input rule is not touched — that rule binds only the cap sensor. It is set aside as a design-time change on three grounds, none principled, all about the comparison's value: (1) the control earns its strength from being off-the-shelf; an attention fusion block this project picked and tuned puts every measured gap on top of bespoke machinery, and "why not vanilla DP?" gets harder to answer; (2) the fusion architecture is shared by both arms, so it is a whole-network change riding on the critical path of a rebaseline — new fork code, hyperparameters untested against the fixed 100K budget, and an evidence base (their fusion ablation: wiping only, from-scratch AST) that would itself need real-robot validation with rollouts the budget reserves for the treatment comparison; (3) ManiWAV's own numbers do not transfer cleanly enough to justify (1)+(2) up front. If the ID curves suggest the audio stream is underused under plain concatenation into `global_cond`, a concat-vs-cross-attention ablation — both arms, same episodes, no new collection — is a follow-up study, not a mid-campaign edit.

**Update 2026-10-15: the ablation is run, at 100% only, alongside the curve.**
- *Why now:* the audio-sensitivity test showed every pre-fix checkpoint ignored audio. The cause was the normalization bug (§1.7), not concatenation. But whether concatenation uses the *fixed* audio is unknown.
- *The arm:* `generic_fusion_100` is `generic`@100% with `policy.fusion_mode = "modality_attention"` and nothing else changed. The fork reproduces ManiWAV's `ManiWAVObsEncoder` modality-attention:
  - one token for the image features of both obs steps, and one token for the audio features linearly projected to the same width (1024);
  - one `TransformerEncoderLayer` over the two tokens (8 heads, FFN 2048, dropout 0);
  - the two outputs concatenated and projected to 512, then the state appended.
- *The curve stays on concat.*
- *Decision rule:* the offline audio-sensitivity test on `generic_100` and `generic_fusion_100`.
  - If concat uses audio, fusion is an appendix ablation.
  - If concat still ignores audio and fusion does not, switching the curve to fusion is a justified, evidence-based change.
- *Result 2026-10-16: concat uses audio, so the curve stays on concat and fusion is an appendix ablation.*
  - The test used 590 samples (train episodes, held-out episodes and eval rollouts), 4 matched noise seeds each, and compared the translation of the executed chunk.
  - Swapping in another episode's audio moves `generic_100` by 5.7× its seed-to-seed noise floor (1.9 mm) and `generic_fusion_100` by 8.4× (3.0 mm). Every pre-fix checkpoint scored ≤ 0.01×.
  - The effect peaks during twisting: 11× for concat, 14× for fusion.
  - It is the same on held-out episodes and eval rollouts as on training episodes, so it is not memorization.
  - Images still dominate both models: swapping images gives about 30×. The audio effect is 18% (concat) and 29% (fusion) of the image effect.
  - Robot success rates decide whether fusion's larger audio use matters.

~~**Design A — encoder pretraining.** Take a generic-pretrained perception encoder (ImageNet/AudioSet), finetune it to predict the instrumentation signal, then use those weights to initialize the policy encoder and finetune everything.~~ **Retired 2026-10.** The A-vs-C comparison was a controlled replication of the workshop mechanism, not the headline; running A cost four extra trainings, per-level encoder pretraining, and the modality-suitability gate that only A needed. The machinery (`train_ast_bottle.py`, the `*_encoder_init_checkpoint` config fields, the fork's strict-load code) stays on disk in case the from-scratch verification cell below surprises us in a direction that revives the question.

> An intermediate design B (auxiliary head hanging off the encoder, predicting the instrumentation at the current timestep) is **not** run. C subsumes what it was for, without the loss-weighting problem.

## 1.3 Variants

Two arms on every task — control and treatment. (A verification cell for the pilot was planned and dropped on 2026-10-14; see below.)

| Variant | Encoder init | Output vector | Role |
|---|---|---|---|
| **`generic`** | ImageNet / AudioSet | action only | control |
| **`generic_c`** | ImageNet / AudioSet | action + instrumentation | treatment; the *only* difference from the control is the auxiliary channels |
| ~~**`from_scratch`**~~ | random | action only | ~~init verification, pilot 100% level only~~ **dropped 2026-10-14, not run** |

**Why both arms start pretrained.** ImageNet/AudioSet initialization is the practitioner standard — the original Diffusion Policy code initializes its ResNet-18 with ImageNet weights, and lerobot defaults ACT's backbone to `ResNet18_Weights.IMAGENET1K_V1` (its DiffusionPolicy dataclass defaults to `None`; lerobot's own configs and this repo's pass it explicitly) — so it is where any practitioner starts, and it makes the control as strong as free practice allows. It also collapses the design to a single manipulated factor: `generic` and `generic_c` share architecture, init, input normalization, hyperparameters, and step budget, differing only in the denoised width.

**(Dropped 2026-10-14: the verification cell is not run, so the init assumption stays unchecked and the paper must not claim it. Rationale kept below for the record.)** **Why the assumption is checked, not trusted.** For *contact audio* specifically, "pretrained is free and always better" is unproven: ManiWAV found a from-scratch AST beat AudioSet-based audio encoders on contact-rich tasks — though their comparison confounds architecture with initialization (scratch transformer vs AudioSet-pretrained CNN), which is exactly the confound this design's verification cell avoids: same AST, same everything else, only init differs. The pre-registered one-sided rule: `from_scratch` is trained and evaluated once at the pilot's 100% level, and **dropped — reported as the appendix init-verification row — only if it loses to `generic` at 100% ID**. A **tie or a win promotes random init to a live third arm** (remaining levels generated, full curve budget): a tie means the initialization the whole design leans on was never shown to help here; a win means AudioSet init actively misfits contact audio — each a finding in its own right.

All encoders are fine-tuned during policy training in every variant. Same architecture, same hyperparameters, same step budget.

## 1.4 Screening — one mandatory gate, one diagnostic

Two cheap offline tests, neither needing robot time. They have **different statuses**, and the difference matters: one decides whether a task happens at all, the other explains *why* a task works — since design A was retired, nothing branches on the second score anymore.

### Step 1 — Privilege test — **mandatory, every task, before anything else**

Train a small MLP on **proprioception alone** (joints, TCP pose, gripper state) to predict the instrumentation signal on a held-out validation split.

> If proprioception predicts the instrumentation well, the robot already knows it. The signal is not privileged, the task is unsuitable, and neither mechanism can help — there is nothing to supervise on that the policy does not already observe.

This is a go/no-go on the task itself and is completely independent of which mechanism wins in §1.3. It is what disqualifies wiping-as-coverage: under a non-revisiting snake path the grid index is a deterministic function of end-effector position, so the MLP scores near-perfectly and the "instrumentation" is revealed as a redundant, noisier proprioception sensor.

Run it on every task, including any new task proposed later, before building hardware.

**Note 2026-10-19 (from the bottle pilot): vision-predictable instrumentation is NOT disqualifying.**
- On the bottle, the wrist image predicts the cap sensor with R² 0.79 (Step 2 table, Task 1). The trained `generic_c` predicts it on unseen episodes with about 97% open/closed accuracy.
- That does not contradict the hypothesis. The claim is *faster* learning: an auxiliary target has to be learnable from the inputs to shape the encoder at all, so a target vision can predict is what the mechanism needs.
- What would make a task unable to show a benefit is narrower: the **un-instrumented** policy already learning the cap state from the action loss alone at low data.
- Check that offline with a linear probe on the frozen `generic` vs `generic_c` encoder features at each data level, before calling a task unsuitable.

### Step 2 — Modality suitability — **diagnostic (was pilot-mandatory under design A)**

For each available modality — image (wrist camera), audio (AST spectrogram), force-torque (MLP over the 6-dim internal FT) — train an encoder to predict the instrumentation on the same 80/20 split.

~~**Why it is mandatory for the bottle pilot.** Design A requires choosing *which* encoder to pretrain. That choice must be made by a pre-registered rule rather than by intuition, or the one comparison the paper hangs on is open to the cherry-picking objection. The rule: every modality clearing the bar is pretrained; those below it are not — a threshold, not an argmax, so a modality that works is not discarded merely because another works slightly better.~~

**Superseded 2026-10.** With design A retired there is no encoder-selection decision left to guard; the test keeps its metric discipline either way. Its former pre-registered rule (a threshold over each modality's margin above the trivial baseline, never an argmax) is retained below because the reported scores still need to be interpretable.

The bar is **relative to a trivial baseline**, never an absolute percentage:

| Signal type | Metric | Bar |
|---|---|---|
| Binary (plug seated, cap open) | balanced accuracy or AUC | pre-registered margin above chance |
| Continuous (force, phototransistor) | R² | pre-registered margin above a mean-predictor |

A flat 70% does not travel across signal types: binary accuracy floors at 50%, R² floors at 0, a 25-way classification floors at 4%. Worse, the plug is unseated for most of every episode, so a constant "not seated" predictor may already clear 70% having learned nothing. Balanced accuracy and AUC are immune to that.

Read every score as *how much this modality adds over proprioception*, using step 1 as the floor.

**Why it is skippable under pressure (status since 2026-10).** Under the run mechanism there is no encoder to select — the instrumentation is predicted from the fused representation of every modality present, and both arms simply load ImageNet and AudioSet for all of them. No branch depends on the score, so the test is not a gate. It remains worth running as a **results-section diagnostic**: it explains *why* the method works on a given task ("audio carries the seating click, image does not") and tells anyone reproducing the setup which sensors they actually need. Cheap, no robot time, but skippable under schedule pressure — the cost is an explanation, not a decision.

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
2. Train the **control** policy (`generic`) on everything collected so far.
3. Evaluate with 20 rollouts.
4. Repeat until success ≥ 90%, or plateau (≤ 5% improvement over two consecutive batches), or a cap of 100 episodes.
5. N is then fixed for that task.

Data reduction levels: 100%, 75%, 50%, 25% of N. Uniform prefix after a fixed-seed shuffle. **Both pretraining and fine-tuning use the same reduced subset** — at 25%, everything sees only 25%. That is the honest data-efficiency test.

Report absolute episode counts alongside percentages ("25%, N=8"), since N will differ across tasks.

## 1.7 Training

- **Diffusion Policy** via lerobot, 10 Hz. Action representation: see below.
- **Fixed 150K steps for every variant** (raised from 100K on 2026-10-14, together with the relative-action switch below, before any curve run). The exact budget does not matter — what matters is that it is equal. A result that holds under an unoptimized-but-equal budget is a stronger result, not a weaker one.
- **Take the final checkpoint.** Not "select by rollout success" — that would mean real-robot rollouts on multiple checkpoints per config, silently multiplying the rollout budget. Fixed budget, final checkpoint, no selection.
- ~~**Pretraining (design A only):** ...~~ / ~~**Design A starts from generic weights:** ...~~ / ~~**No step equalization:** ...~~ — the three design-A training rules (per-modality pretraining LRs, the two-stage starting point, the step-equalization debate) are retired with the arm; the single-stage treatment needs none of them. The whole design, `generic` and `generic_c` alike, is one 150K-step run.

**Action representation: UMI-style relative poses (since 2026-10-14; tool-frame deltas before).**

*Decision, 2026-10-14, before any curve run.* Every action of a chunk is the absolute target pose `[xyz, rot6d]` expressed **in the frame of the end-effector pose at the latest observation** — `T_rel = T_ref⁻¹ · T_target`, UMI (Chi et al., 2024). This is the "principled fix" the delta limitation below names: all `horizon` actions share one observed reference, so errors do not accumulate along the chunk, and the representation stays equivariant to where the bottle is presented (it is relative to the tool, not the base). Implementation: datasets built with `prepare_datasets_bottle --relative-actions` store absolute targets plus **relative** action statistics (`action_representation.json`); `SE3RelativeActionsProcessorStep` (lerobot fork, `processor/se3_relative_action_processor.py`) converts to relative before normalization when `policy.use_relative_actions` is set (generate_configs sets it from the dataset); at inference `LerobotAgent` converts each chunk back with the pose cached when the chunk was predicted — a postprocessor cannot, because DiffusionPolicy pops queued actions one at a time. The sensor tail of `generic_c` passes through unchanged. Reasons 2–4 against lerobot's processors below are answered by this custom step (geometric, not elementwise; diffusion pre-processor wired; eval path converted in the agent); reason 1 still holds — the representation is identical across arms, so it is a nuisance factor for the instrumentation comparison. It was adopted together with the recovery episodes (§1.6 log), so the two contributions to the N-finding jump are not separated; the `generic_90ep_recovery` (delta) run exists to separate them and is reported as a side note, not a design input.

*The previous choice, tool-frame deltas, kept for the record:*

The policy predicts `[delta_xyz_tool(3), rot6d(R_delta)(6)]` — a translation offset in the **tool** frame and a relative rotation `R_delta = R_current^T · R_target`, applied at execution to the robot's live pose. Considered and rejected: absolute joint space (lerobot's default) and lerobot's `RelativeActionsProcessorStep` / `AbsoluteActionsProcessorStep`.

*The defense.* Tool-frame actions combined with a **wrist-mounted** camera make the policy equivariant to where the non-dominant arm presents the bottle: move the whole scene rigidly and both the correct action and the observed image are unchanged. The non-dominant arm presents the bottle at ~100 different poses, and the policy sees each as the same problem rather than a hundred separate ones. On a data-efficiency paper that equivariance is doing real work, which is also why **absolute joint space is the worse option here** — it would turn each presentation pose into a distinct configuration with no sharing between them.

*Why not lerobot's relative-action processors.* Four reasons, in order of weight:
1. It is orthogonal to the paper's claim. Action representation is a nuisance factor held identical across all arms, so it moves absolute success rates but cannot affect the instrumentation comparison.
2. It would rewrite the eval path — the policy would emit absolute poses instead of deltas — which is the riskiest code to change immediately before collection.
3. Relative-to-chunk-start **cannot be precomputed per frame** (frame *t*'s action appears in up to `horizon` chunks with different reference poses), so it has to be a processor step, and `make_diffusion_pre_post_processors` has none — the machinery is wired for the pi family only.
4. `to_relative_actions` is elementwise subtraction, which cannot compose rotations: the rot6d dims would get a linear difference rather than a geometric delta. Invertible and therefore lossless, but not what the representation is supposed to mean.

*Proprioception stays in `observation.state`.* It entered this protocol as the §1.4 screening gate, not as a policy input — the policy input is a separate decision, and it is to keep it: with `n_obs_steps = 2`, pose[t-1] and pose[t] give velocity, which is not cleanly recoverable from two wrist frames and matters for a contact task; the petri-dish task feeds the policy base-frame target coordinates, so stripping pose here would make the pilot structurally different from the tasks it is pooled with; and it is identical across all arms, so it cannot affect the comparison either way.

*Limitation to state in the paper.* This is UMI's "delta" category, and its objection applies within a chunk: action *k* is an offset from pose[t+k], which the policy never observes, so later actions assume the earlier ones executed as predicted. Two things blunt it — each action is applied to the robot's **live** pose rather than to an integrated prediction, and at `n_action_steps = 8` / 10 Hz the accumulation window is 0.8 s. The principled fix, if a reviewer presses, is to express all `horizon` actions as tool-frame offsets from the *chunk-start* pose: UMI-correct and equivariance-preserving, but it needs a custom processor step, since neither a dataset transform nor lerobot's elementwise version can do it.

*Revisitable.* Absolute target poses are recoverable from what is recorded — `policy_action_to_tcp_pose(robot_pose, action)` — so a relative-action arm can be run later on the same episodes as a clean A/B, without recollecting.

**Normalization of the instrumentation signal.**

*Treatment (`generic_c`):* normalize the instrumentation channels with the **same normalizer lerobot applies to the action dims** (dataset mean/std). Mixed scales inside one denoised vector give the network badly conditioned inputs even though the loss is fine.

*Measured operating ranges* (formerly the design-A target normalization; they and `PER_CHANNEL_THRESHOLDS` were re-derived together on 2026-10-07 for the current sensor layout and motion): against a 0–3.3 V ADC span, the bottle sensor only ever traversed S0 3.026–3.252 V, S1 2.633–3.290 V, S2 2.101–3.297 V (global min/max over `sensor_logs/run_{0013,0018,0020,0021,0022,0023}.json` — the re-tuned motion on the "test" split poses; `derive_thresholds.py` reports these as the paste candidate). The superseded pre-retune values were S0 2.96–3.25 / S1 2.48–3.29 / S2 2.08–3.25 (`run_{0003,0005,0006}`, previous layout). Dividing by 3.3 V would compress the entire useful signal into a fraction of the range. Ranges must come from separate calibration runs rather than the demonstration set, so they are fixed before training and identical across all reduction levels — no leakage. **Re-derive alongside `PER_CHANNEL_THRESHOLDS` again whenever the cap, mounting or motion geometry changes.**

**Auxiliary loss weighting (the treatment) is nearly free.** With `prediction_type=epsilon` every channel's regression target is the sampled noise ε ~ N(0, I), so all channels sit on the same loss scale regardless of what the underlying quantity is. With mean reduction over 12 channels, the 3 instrumentation dims take 3/12 = 25% of the objective automatically. No λ sweep, no gradient-norm matching. Just be deliberate that channel count sets the weight: three phototransistors give the instrumentation 25%, one gives it 10%.

Accept and state one consequence: the *action* term is correspondingly scaled 9/12 = 0.75× relative to the action-only arms. That is inherent to the design rather than a bug, it is what the colleague's workshop result already did, and isolating it would need a non-standard loss patch.

**Normalization of the encoder *inputs* is matched to each encoder's initialization.** Distinct from the instrumentation-target normalization above, and it is not a parity violation — a pretrained encoder is (weights, expected input distribution), and splitting the two handicaps it. Mis-normalizing the generic arm to keep configs superficially identical would weaken exactly the control that has to be strong for the "is the custom hardware worth building" argument.

| Arm | image (`dataset.use_imagenet_stats`) | audio (`audio_norm_mean/std`) |
|---|---|---|
| generic **and generic_c** | **ImageNet (`true`)** | **per-level dataset-computed** (since 2026-10-15; AudioSet's −4.2677393 / 4.5689974 before — see below) |
| ~~from_scratch (verification cell)~~ dropped 2026-10-14 | dataset (`false`) | per-level dataset-computed |

**Audio-normalization bug, found and fixed 2026-10-15. Every checkpoint trained before this date ignores audio.**

*Cause*
- The AudioSet constants apply to *raw* Kaldi log-fbank values.
- `SpectrogramSubscriberKaldi.get_spectogram` (`hardware/ipc_mic.py`) stores spectrograms as `(fbank + 20) / 60`. The dataset mean and std of these values are 0.661 / 0.062.
- Applying the raw constants to the rescaled values gave the AST an almost constant input (0.540 ± 0.007, about 1/70 of its pretraining variation).
- Because the task gradient on the AST was then almost zero, lerobot's coupled `adam` weight decay drove its weights to about 0: median |w| was 1e-38, and more than 99.8 % were below 1e-3.

*Evidence*
- An offline sensitivity test on 11 checkpoints and 590 samples changed the audio input in five ways: replaced with silence, replaced with the dataset-mean spectrogram, swapped with another episode's, swapped with another moment's, and shifted by 0.5 s.
- None of these changed the actions by more than 0.02× the seed-noise floor, and none flipped a motion decision.
- Swapping the image moved the actions 18–33× the noise floor and flipped 40 % of decisions.

*Fix (final, 2026-10-15)*
- **Audio: per-level dataset stats in every arm** (`audio_stats.json`, about 0.661 / 0.062). With the encoder's `(x − mean) / (2·std)` this reproduces AudioSet's *normalized* input distribution (mean 0, std 0.5). That is the AST authors' recipe for fine-tuning on a new dataset.
- **Why not AudioSet's constants mapped to the stored scale** (0.2622 / 0.07615, tried first): the microphone waveform is in ADC units, not [−1, 1], so our raw fbank sits about +24 above AudioSet's mean. Even correctly mapped, the AST input would be centred at about +2.6 instead of 0.
- **Leak and parity:** both curve arms of a level share the stats, so the instrumentation comparison is untouched. Each level's stats come only from its own training episodes, so nothing leaks. Paper wording: ImageNet stats for images; dataset stats for audio, because the spectrogram's scale differs from AudioSet's.
- **Optimizer: AdamW, weight decay 1e-6** (decoupled), as in Diffusion Policy and ManiWAV. `use_policy_training_preset` is false, because lerobot's diffusion preset is the coupled `adam`.

*Consequences*
- All curve runs were resubmitted (a first resubmission with the mapped AudioSet constants and weight decay 0 was cancelled after about 1 h).
- The pre-fix curve models are archived under `outputs/train/bottle/audiobug_2026-10-15/`, locally and on `/project_ghent`. The N-finding models (`generic_{50,70,90}ep_recovery*`) stay where they are, but they have the same bug.
- Every pre-fix real-robot result is a **vision + state** result.
- N = 70 was chosen with those policies. The fixed `generic`@100% run is trained on exactly the N-level set, so its ID eval re-checks the ≥ 90 % rule for the audio-conditioned policy. **Re-checked 2026-10-16:** `generic_100` 18/20 = 90%, so N = 70 stands.

`audio_norm_mean`/`audio_norm_std` default to `0.0`/`1.0` — i.e. **no normalization at all** — so every arm must set them explicitly. The random-init cell has no prior expectation, so per-level dataset stats are simply the well-conditioned default (and per level, never global — the leak rule in `generate_configs.py`).

Framing for the paper: the two curve arms are *identical in every respect except the three auxiliary channels of the denoised vector* — same initialization, so same matched input normalization, same hyperparameters, same step budget. This is exactly what the four-arm matrix could not offer: with A-vs-C (or generic-vs-C) the arms differed in initialization and normalization as well, so their gaps were not a pristine isolation of the treatment. With the `from_scratch` cell dropped, no init-dependent normalization contrast is left in the design.

## 1.8 Evaluation

- **40 real-world rollouts per curve configuration**: 20 in-distribution + 20 out-of-distribution. (The `from_scratch` verification cell's 20 ID rollouts were dropped with the cell, 2026-10-14.)
- **Success rate** is the only metric. No secondary metrics.
- **Inference settings, fixed for every arm, level and condition** (2026-10-14): `--num-inference-steps 10` (DDIM; the checkpoint's 100 takes ~440 ms per chunk on the 4090 and stalls the 10 Hz loop) and `--n-action-steps 8` (the trained value). A config evaluated with other inference settings is not comparable.
- Rollout timeout: 2× the median demonstration duration, **rounded up to the next 5 s** (**65 s** for the bottle pilot, fixed 2026-10-14 with N and used for every curve rollout; for a new task the guide's snippet recomputes it from that task's data and applies the rounding). The same timeout is used for every arm and condition; a config evaluated with a different timeout is not comparable.
- A rollout ends on **sustained sensor success** (the deciding channels S1 + S2, `OPEN_CHANNELS`, at or above their thresholds for 5 consecutive steps — `is_cap_open` uses `>=`; S0 has no usable closed/open gap and is not part of success since 2026-10-09), **timeout**, **force-abort** (any drift-corrected force axis beyond `MAX_ABS_FORCE_NEWTONS`), or an **operator Enter-stop**. **Force-abort threshold: 96 N since 2026-10-16** (was 52 N, see the note after the 50 ep detail in §1.6). An Enter-stopped rollout is scored by the operator's visual verdict — the episode's `next.success` and the results-row `success` take that answer, because a visibly-open cap whose third sensor channel never clears is a real success the sensors miss. The verdict's source is recorded in the results rows (`success_source: "operator" | "sensors"`; the eval dataset carries only the resulting `next.success`, not the source). The verdict never overrides a sensor-determined outcome. Every rollout end must be reported with its outcome class and source.
- The ID/OOD condition is **declared by the operator's `--condition` flag, not verified by the script** (the confirmation prompt was removed 2026-10; the label is what the sticker state was at the operator's hand, and mislabeling is an operating error to avoid, not a check to automate).
- OOD varies **only** the generalization axis. Every other condition — including the bottle-holding arm's pose distribution — is drawn from the training distribution, or an OOD failure cannot be attributed to the axis under test.

**Statistics: pool across tasks, stratify by task.**

Twenty rollouts of one trained policy are twenty samples of *that policy*, not of *the method* — the training run is the real experimental unit. Pooling across tasks supplies genuinely independent replicates, so this is a correctness improvement as well as a power one.

Use **Cochran–Mantel–Haenszel** for pairwise variant comparisons at each data level, stratified by task, or a GLMM with variant and data level as fixed effects and task as a random effect. Do not simply concatenate the 2×2 tables — that invites Simpson's paradox if one task's effect runs the other way.

| Rollouts per cell | 95% CI half-width at p≈0.5 | 50% vs 80% detectable? |
|---|---|---|
| 20 (one task) | ±22 pp | borderline, p ≈ 0.04 |
| 60 (3 tasks) | ±13 pp | yes |
| 100 (5 tasks) | ±10 pp | comfortably |

**Pre-register the pooled analysis as primary and per-task curves as descriptive.** Deciding to pool after seeing per-task results is the thing reviewers punish.

## 1.9 Reporting

Two distinct claims from the same rollouts. Never conflate them:

- **Figure A — data efficiency:** success rate vs. % training data, ID rollouts only, pooled across tasks, two curves (control, treatment).
- **Figure B — generalization:** ID vs. OOD success at each data level, per arm.
- **Table 1 — screening:** per task, the proprioception privilege-test score for every task attempted (including those that failed the gate), plus modality suitability scores wherever they were run — diagnostic on every task since 2026-10.
- ~~**Table 2 — init verification (pilot only)**~~ — dropped with the `from_scratch` cell (2026-10-14).

---

# Part 2 — Per-task checklists

## Task 1 — Bottle opening (pilot)

The task that defines the design: the control and treatment arms run at all four data levels here
(the `from_scratch` init-verification cell planned for this task was dropped on 2026-10-14). The
remaining tasks inherit the two-arm curve design. **Status 2026-10-20:** ID curves done, OOD sticker
nearly done; the complete results record is `Bottle_Pilot_Results.md`.

**Setup.** Left UR5 + Schunk gripper opens a flick-switch cap; right UR5 holds the bottle at a different pose each episode. Cap pose is recomputed live from the right arm's TCP (`bottle_station_env.py:47-50`), which is what makes the opening motion scriptable.

**Instrumentation.** Three phototransistor channels inside the cap, published over DDS on topic `Bottle`, read into `obs["bottle_sensor"]` (`bottle_sensor.py:36-53`). Continuous, 3-dim.

**Success.** **Current (since 2026-10-09, 6-leg retuned motion):** only S1 and S2 decide "open" (`OPEN_CHANNELS`); S1 is gated at `leg_3_end` and S2 at `leg_7_end`, the end of the last leg, because S2 jumps 1–4 s after `leg_5_end`. S0 has no usable closed/open gap, crosses back and forth, always opens first, and is logged only. Thresholds 3.20/3.12/3.10 V (`hardware/bottle_sensor.py` is the provenance). *History, superseded:* each channel passed its own checkpoint at its own stage of the motion: S0 at `leg_2_end`, S1 at `leg_3_end`, S2 at `leg_5_end` (`SENSOR_CHECKPOINTS`, mapped 2026-10-07 to where the channels actually cross — the old `push_end` gate failed 2 of 4 clean runs because S0 pops ±0.2 s around that event; S2 gates after legs 4–5 because the first collected episode showed a pose where leg 3 opened S1 but left S2 pinned at 2.72 V across four depth-stamped replays, a rotation shortfall only further legs can supply — and with `leg_3_end` as the last gate those legs never executed). A failed checkpoint triggers a retry, so an episode only succeeds once all three have passed — but at their respective moments, not simultaneously at the end. Once every channel holds open for a 0.5 s dwell, the remaining legs are skipped and the gripper lifts 5 cm (`STOP_LEGS_WHEN_OPEN`) — on poses that open by leg 3, legs 4–6 are post-open travel that could press a loose cap back down; on poses that need the extra turn, the dwell lets them end the moment the last tab clears. Failed episodes are deleted during collection rather than saved and filtered later.

> ⚠️ **Re-cover, measured 2026-10-07.** The `run_0006` transient is systematic under the current mounting, not a one-off: the calibration batch shows pre-pop light leaks (excursions to the open level before the tab yields) at up to ~2% of covered samples on some poses. They are absorbed by construction — thresholds come from percentile bounds (a leak never sets a threshold), checkpoint reads are 0.3 s debounced, success needs 5 sustained steps, and the stop-open rule needs a 0.5 s dwell. But the *ordering* is pose-dependent (run_0013's pose crossed S1 before leg 2 even ended), and some poses need more turn than leg 3 delivers: test pose 1 never opened in 3/3 batch attempts, and the same signature appeared in collection on train pose 3 (S1 open, S2 pinned 2.72 V through four leg-3 replays). That is what moved S2's gate to `leg_5_end` — the retry now redoes the legs that actually turn the cap, each 3 mm deeper; a pose that still fails after `MAX_MOTION_RETRIES` is a reproducible capability failure, not noise, and a candidate for the feasibility blacklist (with a written reason) if the deeper-and-further sweeps keep missing it.

**Generalization axis — appearance only.** There is one physical bottle and one cap; no variants are 3D printed. OOD is produced by changing the bottle's **visual appearance** — stickers, tape, patterns, matte vs. glossy — while the mechanism stays identical. Train on one set of appearances, evaluate OOD on unseen ones.

This is a narrower axis than the stiffness-plus-appearance split previously planned, and the paper should call it *appearance robustness* rather than generalization unqualified: cap dynamics no longer vary, so nothing here tests dynamics transfer. It is, however, tightly matched to the hypothesis — if instrumentation supervision really teaches the encoder to attend to cap state rather than incidental visual structure, it should be measurably less disturbed by appearance changes than the control (`generic`) is. Make the shift substantial (colour, pattern, coverage), not a single small sticker.

> ⚠️ **Two conditions this axis depends on. Verify both before collecting.**
> 1. **Stickers must stay clear of the cap's light path.** The phototransistors measure light inside the cap. If a sticker changes what reaches them, `PER_CHANNEL_THRESHOLDS` no longer holds and the *success criterion itself* differs between ID and OOD — the two conditions would be scored with different rulers. Keep appearance changes on the bottle body, and re-read the sensor on a fully-open and fully-closed cap for every sticker configuration to confirm the voltages have not moved.
> 2. **The appearance change must be visible to the wrist camera.** The wrist cam looks down at the cap; if the bottle body is mostly out of frame, ID and OOD observations are identical, there is no distribution shift, and OOD success trivially equals ID success. Check recorded `wrist_image` frames for how much body is in view before committing to sticker placement.

*Optional second axis, free:* hold out a region of the right-arm pose space instead of sampling OOD poses from the training distribution. No hardware needed. Do not split the 20 OOD rollouts across two axes — too thin. Appearance is primary; keep pose-holdout in reserve.

### Hardware and setup

- [x] ~~Remove the scene camera before recording a single episode.~~ **Done** — publisher, subscriber, factory method, topic constants and the `scene_image` / `scene_image_original` obs keys are gone; the ZED setup is recoverable from git if a later task needs an overhead camera. Wrist camera only.
- [x] ~~Wire audio in before collection.~~ **Done** — `SpectrogramSubscriberKaldi` behind a new `with_spectogram` flag on `UR5eStation`, emitting `spectogram_image` and `spectogram_values`; `collect_data_bottle_opening.py` passes `with_spectogram=True`. Both this and the scene-camera removal change the recorded obs keys, which define the LeRobot feature schema (`dataset_recorder.py:163-188`), so neither is recoverable after episodes exist.

**Modalities for the pilot:** wrist image, audio (AST), and `observation.state` = TCP pose (6) ⊕ drift-corrected internal FT (6). FT rides inside the state vector because `observation.state` is the only state key the policy consumes — any other `observation.*` vector is typed but never reaches `global_cond`.

**FT drift correction.** The internal FT sensor drifts thermally over tens of minutes. Uncorrected this is not merely noise: episodes recorded close together share an offset, and since appearance conditions are collected in blocks, the offset becomes a shortcut feature identifying the condition — inflating ID success and collapsing OOD. `capture_ft_bias()` averages the reading at the **fixed home joint configuration** at the start of every episode, before the right arm moves the bottle. The home pose is the only configuration identical across episodes, so the payload's gravity contribution is constant there and any episode-to-episode change is drift; a baseline taken at the per-episode hover pose would instead fold a *varying* gravity term into the zero.

The raw `ft` and the per-episode `ft_bias` are both recorded; the subtraction happens in `prepare_datasets_bottle.py`, so the correction stays inspectable and redoable and the raw signal is never lost. An all-zero `ft_bias` degrades to a no-op, so episodes predating this remain usable.

> This cancels drift, not gravity: subtracting a home-pose baseline elsewhere leaves `gravity_at_pose − gravity_at_home`. UR documents `actual_TCP_force` as payload-compensated, which would make that residual negligible — but only if the payload mass and CoG are configured in the controller. **Verify before collecting:** move through the task's orientations with nothing in contact and confirm the reading stays put. If it swings by newtons, fix the payload configuration first; no baseline can rescue it.
- [x] ~~Verify the wrist RealSense starts at 720p, not the 480p fallback.~~ **Done** — the fallback is removed; `create_wrist_camera` requests 720p and raises if that profile will not start (§4.6). Confirm the hardware actually supports it on the first run, since the D405 note in `ipc_camera.py:248` suggests it may not.
- [ ] Fix the appearance set: how many sticker configurations, and which are train vs. OOD. No printing needed.
- [ ] Verify both conditions in the generalization-axis warning above — sensor voltages unchanged by stickers, and stickers visible in the wrist frame.
- [x] ~~Recalibrate `PER_CHANNEL_THRESHOLDS` for the current sensor layout.~~ **Done (2026-10-07).** The motion was re-tuned, then thresholds and `CALIBRATED_RANGE` were re-derived together from the calibration batch (`sensor_logs/run_{0013,0018,0020,0021,0022,0023}.json`, the re-tuned motion on the "test" split poses) via `derive_thresholds.py`. Values then: 3.16 / 3.12 / 3.08 V, `SENSOR_CHECKPOINTS` `leg_2_end`→S0, `leg_3_end`→S1, `leg_5_end`→S2. **Re-derived again 2026-10-09** for the 6-leg motion, from the fully-closed runs `run_0024, 0026–0028, 0030, 0032–0038`: **3.20 / 3.12 / 3.10 V**, `CALIBRATED_RANGE` as 1st/99th percentiles, `OPEN_CHANNELS` = S1 + S2, `SENSOR_CHECKPOINTS` `leg_3_end`→S1, `leg_7_end`→S2. All 110 training episodes were collected under these. **Still unrecorded: a sensor re-read with the sticker on** (the pilot's OOD sticker sits on the lid, so this matters more than for a body sticker). Nothing scored or labelled with the pre-2026-10-07 constants is trustworthy.
- [ ] With a single cap there is no per-variant recalibration — but there *is* a per-appearance sanity check, per the warning above.

### Data collection

- [x] ~~Fix the success-flag bug.~~ **Done** (§4.1). Episodes recorded *before* the fix are all labelled `success=False` regardless of outcome — discard or re-label them from the sensor logs before they enter any dataset.
- [x] ~~Use a different RNG seed for val/test poses.~~ **Done** (§4.3) — per-split offsets.
- [x] **Recollect the train split — done 2026-10-09/10:** 90 successful expert episodes under the recalibrated constants (raw root `bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train`), plus 20 recovery episodes 2026-10-13/14 (below). The old 51 episodes are retired; the screening scores below still come from them.
- [ ] Confirm the pose distribution for OOD rollouts matches training — only the appearance changes.
- [x] **Done 2026-10-14: N = 70** (log below). Run the N-finding loop (§1.6): batches of 20, retrain the **control** (`generic` @ 100% of what exists so far), 20 rollouts, stop at 90% or plateau, cap 100.

  **N-finding log.** 90 successful episodes were collected in one go (2026-10-09/10, raw root `bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train`) and are evaluated as nested prefixes of the fixed-seed shuffle (`EPISODE_ORDER_SEED` 20260812, `prepared/episode_order.json`) instead of batch by batch: 50 → 70 → 90 episodes. Each level is `generic` trained from scratch on `prepared/bottle_9d_<n>ep` (100K steps, batch 32, final checkpoint), evaluated with 20 ID rollouts on the test split. Rollout timeout **60 s** (median demo 29.4 s → 2×29.4 = 58.8 → 60), the same for every level.

  | Level | Checkpoint | Rollouts | Success | 95% CI (Wilson) | Date |
  |---|---|---|---|---|---|
  | 50 ep | `outputs/train/bottle/generic_50ep/checkpoints/100000/pretrained_model` | 20 | **12/20 = 60%** | 40–77% | 2026-10-12 |
  | 70 ep | `outputs/train/bottle/generic_70ep/...` | — | — | — | trained, not evaluated (superseded by the recovery round below) |
  | 90 ep | `outputs/train/bottle/generic_90ep/checkpoints/100000/pretrained_model` | 20 | **6/20 = 30%** | 15–52% | 2026-10-13 |

  50 ep detail (`outputs/eval/bottle_results.json`, dataset `datasets/bottle_experiment/eval/eval_bottle_generic_50ep_id`, 20 episodes):
  - All 12 successes were called by the sensors (S1+S2 sustained 0.5 s), in 132–150 steps (19–22 s, faster than the 29.4 s median demo).
  - All 8 failures were operator Enter-stops scored "not open": 7 were stopped early, at 66–91 steps (10–13 s), and one at 164 steps (24 s). No timeouts and no force aborts. Failed pose indices (list position): 1, 3, 5, 6, 10, 11, 17, 19. The failure mode was not recorded per rollout -- note it in the results row next time.
  - Rollout 6 (pose 5) ended in a left-arm protective stop at ~step 80, before the operator stop; the crash in the retreat lost its results row, which was re-added by hand from the log (marked with a `note`). `eval_bottle` now ends a rollout as `robot_stopped` when the UR control script dies and writes the results row before the retreat.
  - Test-split poses **9, 10, 19, 23** were blacklisted during this eval (`POSE_BLACKLIST["test"] = [0, 9, 10, 19, 23]`, `splits.py`): each was the pose at the rollout the operator stopped and restarted on (9: left-arm protective stop 4.4 s in; the others rig problems of the pose); those attempts were not saved. The 20 evaluated poses are therefore indices 1–8, 11–18, 20, 21, 22, 24 -- verified by matching each episode's start pose to the pose cache -- and the same list applies to every later arm and level. (Blacklist entries are pose indices, not rollout numbers; `eval_bottle` now prints the index per rollout. The list briefly held 17 and 20 -- rollout numbers entered by mistake -- and was corrected the same day.) The results rows' `pose_index` field is the list position, not this index.
  - Eval conditions changed on 2026-10-12, before these 20 rollouts: right-gripper re-grasp sequence per rollout (open 0.028 m → operator places the closed bottle → close 0.023 m; repeatability not yet measured), force abort at `MAX_ABS_FORCE_NEWTONS` = 52 N (2× the demos' mean per-episode peak, 26.1 N), and the UR FT sensor re-zeroed at home before each bias capture (its offset had drifted to ~40 N, which force-aborted a first attempt in free air; that attempt was discarded). Keep these identical for 70 ep / 90 ep and every curve config.
  - **Force-abort threshold raised to 96 N on 2026-10-16, before the curve evaluation; it applies to every curve and ablation model.**
    - *Old rule:* 52 N, 2× the 90 expert demos' mean per-episode peak.
    - *New rule:* 96 N, 2× the 95th-percentile per-episode peak of the 110 training episodes (expert + recovery): 48.0 N. Their median is 22.5, mean 26.8, max 140.8 N.
    - *Why:* 52 N cut at demo-level pushing. 4 of the 110 successful training episodes exceed it (2 expert, 2 recovery), so it scored how hard a policy pushes rather than whether it fails. At 96 N the abort is a safety stop, inside what the rig took during the demos (96 and 141 N), and success is decided by the sensors and the timeout.
    - *Trigger, stated openly:* `generic_100` (post audio fix) force-aborted 2 of its first 3 rollouts at 52 N. One was a z-spike 2 → 42 N in about 0.3 s; the other held 49–51 N in z for 6+ steps while twisting.
    - *Handling of those 3 rollouts:* all 3 were discarded, the success included. The results rows are in `outputs/eval/bottle_results.backup_2026-10-16.json`, the dataset in `datasets/bottle_experiment/eval_discarded/eval_bottle_generic_100_ddim10_a8_id_52N_2026-10-16`. `generic_100` is re-evaluated from scratch under 96 N.
    - *No earlier result changes:* no completed rollout before this had a force abort. The two recovery models peaked at ≤ 44 N in all 40 rollouts.
    - *Secondary metric:* every eval dataset logs the force trace, so report per model the rate of rollouts that would have aborted at 52 N, alongside the success rate.

  Decision: 60% < 90% -> not enough at 50 episodes; evaluate `generic_70ep` next (then 90 ep). Plateau rule (§1.6) applies across 50 → 70 → 90; the cap is 100.

  **90 ep (2026-10-13): 6/20 = 30%** (`generic_90ep`, 100K steps, delta actions) -- worse than 50 ep. Failure analysis of every rollout so far: the failures make first contact too deep (median 129.8 mm below hover vs 125.4 mm for successes and 126 mm in every demo), jam at 20-38 N with no lid motion, and never recover -- the demos contain no recovery. The scripted demonstrations always grip at the right depth, so the policy never saw what to do after a bad first contact.

  **Recovery episodes (2026-10-13).** 20 recovery episodes were recorded with `ur5station/bottle/collect_recovery_bottle.py` on train-split poses (raw root `.../expert/bottle_opening_train_recovery`): a too-deep first contact (0-10 scripted, 11-19 driven by the `generic_90ep` policy itself at 10 DDIM steps / 1 action step) followed by a scripted recovery -- lift, back over the grip point, opening motion with the grip point raised 3 mm (`RECOVERY_RAISE_M`). Note the small leak this implies: episodes 11-19 start with behaviour of a policy trained on 90 demos, and they are part of every level below.

  **Second round -- control = relative actions + recovery episodes, 150K steps**, evaluated at 10 DDIM steps / 8 action steps, 65 s timeout, 20 ID rollouts on the same test poses:

  | Level | Checkpoint | Rollouts | Success | 95% CI (Wilson) | Date |
  |---|---|---|---|---|---|
  | 50 ep + 20 recovery | `outputs/train/bottle/generic_50ep_recovery_rel/checkpoints/150000/pretrained_model` | 20 | **19/20 = 95%** | 76–99% | 2026-10-14 |
  | 90 ep + 20 recovery | `outputs/train/bottle/generic_90ep_recovery_rel/checkpoints/150000/pretrained_model` | 20 | **19/20 = 95%** | 76–99% | 2026-10-14 |
  | 70 ep + 20 recovery | `generic_70ep_recovery_rel` (GPULab) | — | — | — | trained, not evaluated (not needed: 50+20 already reached 95%) |

  Both remaining failures (50+20 pose 11, 90+20 pose 19) are the same mode: S1 opened, S2 stayed at ~2.5 V, then the contact force dropped to 1–3 N and the arm drifted (~1.6 m path vs ~0.4 m in a success) until the timeout. The too-deep failure is gone: first contact is 2–4 mm shallower than the old policies', and the 3 deep contacts left (> 128 mm) all recovered. Behaviour differs by data volume at equal success: 90+20 lifts and re-grips before the lid moves in 10/20 rollouts (50+20: 2/20, Fisher p = 0.014) with lower first-contact force (26 vs 35 N, Mann-Whitney p = 0.004), and is ~2 s slower.

  **Decision (2026-10-14): N = 70** = the 50-expert-episode set + the 20 recovery episodes -- the smallest level at ≥ 90% (§1.6). The control is fixed as relative actions, 150K steps, 10 DDIM steps / 8 action steps, 65 s timeout. The 70 episodes are pooled, shuffled once with `EPISODE_ORDER_SEED`, and cut into the nested levels 100/75/50/25% = **70/52/35/18 episodes** (recovery episodes 20/13/9/4) by `prepare_datasets_bottle --pool-count 50 --extra-raw-root .../bottle_opening_train_recovery --relative-actions` (manifest `prepared/episode_order_pool50.json`). The recovery episodes are spread over the levels like any other episode, so "at 25%, everything sees only 25%" still holds. The 100% level is exactly the `generic_50ep_recovery_rel` training set.

### Screening

- [x] **Privilege test** (mandatory). Proprioception MLP → 3-channel sensor. Expect it to fail the gate (cap rotation depends on grip slip and thread engagement, not just wrist angle) — but confirm it, because a scripted motion makes proprioception unusually informative. Run once (see table below): a moderate positive signal, neither a clean pass nor a clean fail against any fixed threshold.
- [x] **Modality suitability** (**diagnostic since 2026-10** — it was mandatory only because design A needed it to pick which encoder to pretrain; with A retired, nothing branches on these scores). Wrist image, audio (AST), FT. Record all scores and fix the threshold *before* looking at them. All three run once (see table below); re-run on the recollected data before reporting them.
- [x] Both tests run from `ur5station/bottle/screen_instrumentation.py`, which builds the *policy's own* encoder classes from the arm's config, so the resulting weights load into the policy with `strict=True` and no key remapping. ~~Audio pretraining for design A proper is `ur5station/bottle/train_ast_bottle.py`~~ — that script is retired with design A (kept on disk for the button lineage).

**Preliminary results** (`datasets/bottle_experiment/prepared/bottle_9d_100`, 51 episodes, one run each, 10 epochs, R² against a mean-predictor on held-out validation). **Provisional twice over:** one run per modality, and computed on the pre-recollection dataset — extend with more seeds and re-run on the new data before treating any of this as load-bearing.

| Modality | r2_mean | S0 | S1 | S2 | Notes |
|---|---|---|---|---|---|
| Proprioception (privilege gate) | 0.182 | 0.146 | 0.227 | 0.172 | Above the R²=0 floor, well short of "near-perfect" |
| Image (wrist camera) | 0.787 | 0.918 | 0.852 | 0.592 | Clearly the strongest of the four |
| Audio (AST spectrogram) | 0.606 | 0.666 | 0.675 | 0.476 | Second-strongest. Getting here needed a fix: the first attempts crashed identically (`RuntimeError: Could not push packet to decoder`) decoding the small (128×298) `spectogram_values` AV1 stream, reproducible at both `num_workers=4` and `num_workers=1` — so not a cross-worker race, but torchcodec breaking in any forked worker process. Video confirmed not corrupted (a full sequential ffmpeg decode and a single-process frame-by-frame torchcodec decode of all 16,405 frames both passed cleanly). Fixed by setting `video_backend="pyav"` for the audio case only (`screen_instrumentation.py`), keeping `num_workers=4`. |
| FT (internal force-torque) | 0.096 | 0.085 | 0.067 | 0.135 | Weakest, close to the mean-predictor floor |

### Training and the two-arm curves

- [x] ~~Confirm design C needs no collection-code change.~~ **Confirmed and implemented.** `ur5station/bottle/prepare_datasets_bottle.py` emits a 12-dim `action` = `[action(9), bottle_sensor(3)]`; lerobot reads the denoised width from `config.action_feature.shape[0]`, so the U-Net, sampling prior and MIN_MAX normalizer all widen with no policy change. Verified: both widths build, train and sample, +10,755 params (0.004%).
- [x] ~~Add the inference slice.~~ **Done** — `LerobotAgent(n_env_action_dims=9)` truncates after the postprocessor. Pass `9` for **all arms** (a no-op for all but `generic_c`) so the eval path is byte-identical across arms.
- [x] Build the 8 prepared datasets (`prepare_datasets_bottle --pool-count 50 --extra-raw-root .../bottle_opening_train_recovery --relative-actions`): N = 70 = 50 expert + 20 recovery, levels 70/52/35/18. Built without `--ft-orientation-correction`.
- [x] Generate the 8 curve configs (`bottle/generate_configs.py`, default `--arms generic,generic_c`) — it asserts at build time that the arms differ only in intended keys, at every level. `audio_norm_mean/std` are **per level in both arms** since 2026-10-15 (that level's `audio_stats.json`, written by `prepare_datasets_bottle`, computed over exactly the episodes the level trains on — a global value would leak the 100% run's statistics into smaller levels); AudioSet's constants do not fit our spectrogram scale (§1.7). The design-A inputs (`--design-a-modalities`, `--pretrain-dir`) are gone with the arm.
- [x] Train 2 arms × 4 data levels = **8 curve runs**, 150K steps each, final checkpoint (done on GPULab; checkpoints `outputs/train/bottle/<arm>_<level>/checkpoints/150000/pretrained_model`, HF `ramen-noodels/bottle_<arm>_<level>`). `generic`@100% is the already-trained `generic_50ep_recovery_rel` (identical episodes, config and budget; already evaluated ID 19/20). ~~Plus the single `from_scratch`@100% verification run~~ -- **dropped 2026-10-14**: the init-verification cell is not run, and the paper does not claim that AudioSet initialization helps over random init.
- [ ] 8 curve configs × (20 ID + 20 OOD) = **320 rollouts** for the pilot (the verification cell's 20 ID are dropped with it). Status 2026-10-20: all 160 ID done; sticker OOD done for `generic_100` (17/20), `generic_c_100` (14/20), `generic_50` (13/20); `generic_c_50` to run; 75/25 none yet.
- **Results 2026-10-19: ID curves, all 160 rollouts** (20 per configuration, the same 20 test poses in the same order for both arms, 96 N abort; `plot_curves.py`, figures in `outputs/figures/bottle_{curve,outcomes}_id.{png,pdf}`).

  | Level (episodes) | `generic` | `generic_c` | Pose-by-pose: only control / only instrumented succeeded | McNemar p |
  |---|---|---|---|---|
  | 25% (18) | 6/20 = 30% | 5/20 = 25% | 3 / 2 | 1.00 |
  | 50% (35) | 13/20 = 65% | 12/20 = 60% | 5 / 4 | 1.00 |
  | 75% (52) | 17/20 = 85% | 15/20 = 75% | 3 / 1 | 0.62 |
  | 100% (70) | 18/20 = 90% | 18/20 = 90% | 1 / 1 | 1.00 |

  - Cochran–Mantel–Haenszel over the four levels: p = 0.57, common odds ratio 1.34 in favour of the control. This single-task test is exploratory; the pre-registered primary analysis is the pooled one (§1.8).
  - **No benefit of instrumentation on ID.**
  - OOD (Figure B), sticker on the lid, 2026-10-20: at 100%, `generic` 17/20 vs `generic_c` 14/20 (pose-by-pose 4 / 1, McNemar p = 0.38), arms alternated in one session under training-range lighting; no robustness advantage for instrumentation. 50% in progress. Details: `Bottle_Pilot_Results.md` §4.4–4.5.
  - **Lighting confound.**
    - The eval camera runs auto-exposure, which cannot compensate strong daylight.
    - All four `generic` sessions and `generic_c_100` ran bright: mean wrist-image brightness 0.53–0.57, 15–25% saturated pixels.
    - `generic_c` 25/50/75 ran on 10-19 after about 14:15 at 0.39–0.40, close to the training data's 0.38 ± 0.06.
    - The confound probably handicapped the control, which still matched or beat the instrumented arm, but the protocol requires identical conditions.
    - **Before reporting:** re-run 50% for both arms (40 rollouts) under controlled lighting, alternating the arms pose by pose instead of evaluating one arm per session.
    - `eval_bottle` now logs `hover_brightness` per rollout and warns outside the training range.
  - **Failures are late, not at the grip.**
    - First-contact depth is the same in failures and successes: 124.0 vs 124.1 mm below hover (p = 0.55), against 125.9 mm in the demos. The too-deep contact that motivated the recovery data is gone.
    - 31 of the 56 failures open S1 and lift S2 to about 2.97 V (threshold 3.10) but never clear it. After S1 opens they keep moving with lower contact force (3.7 vs 5.4 N): the gripper loses the lid edge.
    - 12 failures open neither deciding channel despite some lid movement; 13 never move the lid at all (9 of them force aborts).
  - **Pose list position 0 fails for all 8 curve models.** It succeeded for all 4 earlier models.
    - The start pose is in distribution (18 mm from the nearest training episode).
    - Contact is normal, but after contact 6 of 8 models skip the sideways push and press down until the abort.
    - An offline swap test points to the image: another rollout's image restores the push, audio partly, state not at all.
    - Pose 0 is always the first rollout of a session. **Open:** run it as the third rollout once, to separate pose from session start.
  - **The successful rollouts push harder than the demos:** median peak |Fz| 27 N (`generic`) and 31 N (`generic_c`) vs 12 N, rising with data level. `generic_c` also has a higher force × contact time (145 vs 101 N·s, p < 0.001).
  - **The cap sensor is clean but carries little the image does not.**
    - Noise is about 0.003 V against closed→open steps of 0.17 / 0.61 / 1.11 V.
    - 7 of the 90 expert episodes start with S2 already reading open (11, 14, 55–58, 73): wrong channel-3 labels.
    - About 26% of frames are mid-transition.
    - See the §1.4 note: vision-predictability is not disqualifying; the relevant check is whether `generic` learns the cap state without supervision (linear probe).
  - **The expert demos' FT signal is corrupted.**
    - The expert collector captured the home bias without `zeroFtSensor()` (eval and the recovery collector did re-zero).
    - The sensor-frame offset therefore leaves 6.7 N median, up to 17 N, in free air in all 90 expert demos, and it varies with the arm's pose. The recovery demos and the eval rollouts read about 2 N.
    - The policies essentially ignore the state input: swapping it moves the action by 0.3–0.4× the run-to-run noise.
    - **Fixed 2026-10-19:** `collect_data_bottle` now re-zeroes before the bias.
    - `prepare_datasets_bottle --ft-orientation-correction` repairs existing expert recordings. It fits raw(t) = R_t·o + c per episode on the first 30 free-air frames, which brings rotated held-out free-air frames from 9.1 to 1.2 N median.
    - It is opt-in. The curve datasets were built without it, and rebuilding them would mean retraining.
- [x] ~~Apply the init-verification rule (Table 2)~~ -- dropped with the verification cell (2026-10-14). Analyze the two-arm curves: Figure A (ID, pooled) and Figure B (ID vs OOD per arm). Original item: `from_scratch`@100% ID vs `generic`@100% ID — a **loss** (strictly fewer successes) files it as the appendix verification row; a **tie or win** promotes it to a live arm (generate its remaining levels, add its curve). Then analyze the two-arm curves: Figure A (ID, pooled) and Figure B (ID vs OOD per arm).

## Task 2 — Rubber plug insertion

**Setup.** Six plug sizes, all black, cluttered in a box on an instrumented metal sheet. Robot picks a plug and seats it in the matching hole.
**Instrumentation.** Button-like contacts in the sheet — binary circuit closure.
**Success.** Circuit closed.
**Generalization axis.** Plug size: train 1/3/5, OOD 2/4/6.

- [ ] Build the instrumented sheet — contacts around each hole plus wiring. Still blocked on the promotor for the physical sheet.
- [ ] Confirm the cluttered-box protocol: all six sizes in the box simultaneously (industry-realistic), or one size presented per trial?
- [ ] Confirm box dimensions and whether plugs are reshuffled between trials.
- [ ] Privilege test (mandatory): expect a clear pass — the gripper can be at the correct pose with the plug unseated, so proprioception cannot predict seating.
- [ ] Modality suitability (diagnostic on every task since 2026-10 — design A, the only thing that ever branched on it, is retired): audio is the hypothesis, via the seating click. Use **balanced accuracy or AUC**, not raw accuracy — the signal is near-zero for most of every episode.
- [ ] Script the insertion with circuit closure as the episode-termination signal.

## Task 3 — Petri dish rolling

**Setup.** Press an agar petri dish in a rolling motion over a target. Lid is already off — the task is navigate + roll, two stages, not four.
**Instrumentation.** Load cell under the sampling surface, continuous.
**Success.** Contact force within [F_min, F_max] for a sufficient duration.
**Generalization axis.** Surface material: train on tabletop / metal table / white shelf wood, OOD on wood board with holes / plastic plate / glass.

- [ ] Determine [F_min, F_max] empirically from demonstrations (mean ± 1 SD of scripted force).
- [ ] **Confirm the force tolerance is genuinely narrow.** If any force in a wide band succeeds, the task is too easy to be worth imitation learning. Open since 30 June, still unanswered, and it decides whether this task ships.
- [ ] Privilege test (mandatory): expect a pass — contact force is not in joint angles.
- [ ] Modality suitability (diagnostic — see Task 2; run it here regardless, it answers a question the paper wants). Audio is the hypothesis: friction and contact sound. This is where the **visually-identical-materials** question gets answered: white painted wood and a white plastic plate look the same to the camera but differ acoustically (wood duller, 200-800 Hz; plastic sharper, >1 kHz). If audio can predict force with visual input held constant, that is a result worth reporting on its own.
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

**This recurred on 2026-10-07 and was fixed structurally.** The checkpoint re-map (S0→`leg_2_end`, S1+S2→`leg_3_end`, `leg_6_end` dropped — §1.7) meant the then-current `event_name == "leg_6_end"` literal could never fire again: every collected episode would have been silently recorded a failure, with no error. `run_opening_motion_with_retry` now locates the final gate **by position** (`gated_indices[-1]`, the last checkpoint-bearing segment), so it survives any future re-map without a code change — the same day's second re-map (S2→`leg_5_end`, §1.7) already proved it: absorbed with no edit to the retry logic. The same pass made `_build_motion_segments` name the push checkpoint only when `push_end` is actually gated — a segment that names an ungated event was otherwise counted by the recovery cascade, forcing a full restart instead of resuming from the last leg that passed.

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

`spectogram_values` is a 3-D array, so `dataset_recorder.py:171-177` classifies it as a **video** feature. The mel spectrogram the AST consumes is therefore AV1-encoded (confirmed: PyAV decodes the recorded stream with `libdav1d`) and float-quantized to uint8 — lossy compression applied to one of the modalities under evaluation.

Pre-existing (the button experiment's results went through it too), so it was left alone rather than changed unilaterally, but it deserves a deliberate decision before the audio arms are taken seriously. The lossless option is storing the spectrogram flattened as a 1-D float32 vector, which makes it a state feature; that would need a small change to the fork's audio branch to reshape on the way in.
