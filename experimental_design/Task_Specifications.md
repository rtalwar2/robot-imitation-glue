# Instrumented Imitation Learning — Task Specifications

**Prepared for:** Promotor meeting (follow-up)
**Date:** 8 July 2026
**Status:** Task descriptions formalized; open items per task flagged with 🔶

---

## Shared Experimental Protocol

All five tasks share the same core methodology. This section is referenced by each task one-pager.

### Research hypothesis

Supervised pretraining of a perception encoder on task-specific instrumentation signals (privileged, training-only) acts as an inductive bias, yielding better data efficiency during policy fine-tuning than random initialization or generic pretraining (ImageNet/AudioSet).

### Three-variant comparison

| Variant | Encoder initialization | Purpose |
|---|---|---|
| **From scratch** | Random weights | Lower baseline |
| **Instrumentation-pretrained** | Encoder pretrained to predict instrumentation signal | Proposed method |
| **Generic-pretrained** | Same encoder pretrained on ImageNet (image) or AudioSet (audio) | Controls for "pretraining in general" |

All encoders are **fine-tuned** during policy training. Same architecture, same hyperparameters, same training steps. Only the initialization differs.

### Modality selection

An objective **suitability metric** picks which encoder (image or audio) to pretrain:
1. Split demonstrations 80/20 train/validation
2. Pretrain image encoder → record validation score (accuracy for classification, R² for regression)
3. Pretrain audio encoder → record validation score
4. Pick the higher-scoring modality
5. **Report both scores in the paper** — not just the winner. The suitability metric is part of the method: the claim is "when instrumentation is perceivable in a modality (validated by the suitability metric), pretraining on it improves data efficiency." Reporting both scores preempts cherry-picking critiques.

### Data collection

- **Scripted demonstrations** — chosen because exact force profiles are required (petri dish, wiping tasks) and because scripting ensures repeatability.
- **Diversity in scripted demos:** despite being scripted, diversity is maintained by varying conditions across episodes. For example, the non-dominant arm holds the bottle in different positions, the robot starts from different approach angles, or objects are placed at varied locations. The same underlying trajectory logic is parameterized over a distribution of conditions.
- **Instrumentation aids scripting:** an additional advantage of instrumentation is that it provides ground-truth feedback that can be used to *guide* the scripted trajectory. For example, the load cell reading can close a feedback loop during scripted force application, or the circuit closure signal confirms when a plug is fully seated and the episode can end.
- **Successful episodes only** — filtered by instrumentation signal at episode end.
- **Iterative data collection** to find N (total episodes):
  1. Script and collect a batch of 20 episodes
  2. Train from-scratch policy on all episodes collected so far
  3. Evaluate with 20 rollouts
  4. Repeat until: success ≥ 90%, OR plateau (≤ 5% improvement over 2 consecutive batches), OR cap of 100 episodes
  5. N is then fixed per task

### Data reduction levels

100%, 75%, 50%, 25% of N. Uniform prefix after shuffle with fixed seed. Both pretraining and fine-tuning use the same reduced subset (honest data efficiency test).

### Training

- **Diffusion Policy** via lerobot (image + audio encoders)
- **Fixed step budget** for all variants (equal compute budget). Starting point: lerobot default of 100K steps. Checkpoint selected by rollout success, not training loss.
- **Pretraining as weight initialization:** pretraining updates only the encoder (~5-20% of total policy parameters, e.g., ResNet18 ≈ 11M vs. full DP ≈ 50-200M). Pretraining converges quickly (typically < 5% of the encoder's total optimization steps across pretraining + fine-tuning). The claim is framed as: "instrumentation-supervised initialization produces better starting weights than random or generic initialization."
- Pretraining: LR 1e-4, early stopping (patience 500), BCE for binary, MSE for continuous.
- **Normalization:** continuous instrumentation signals normalized using the **sensor's hardware range** (e.g., load cell rated capacity, ADC max value, grid dimensions). Deterministic, zero data leakage, consistent across all reduction levels.

### Evaluation

- **20 real-world rollouts per configuration**, split:
  - **10 in-distribution (ID)** rollouts — conditions seen during training
  - **10 out-of-distribution (OOD)** rollouts — conditions NOT seen during training
- **Success rate (%)** = primary metric
- Rollout timeout: 2× median demonstration duration
- Statistics: 95% binomial CIs; Fisher's exact test for pairwise variant comparisons

### Paper reporting structure

Each task yields two complementary claims from the same rollouts:
- **Figure A (data efficiency):** Success rate vs. % training data (ID rollouts only)
- **Figure B (generalization):** Success rate ID vs. OOD at each data level

> ⚠️ **Concern:** Data efficiency and generalization are two different claims. They must be reported separately in the paper to avoid conflation. The 10+10 rollout split allows both without doubling total rollouts.

### Experimental scale

| Tasks | Configurations | Rollouts | Est. hands-on time |
|---|---|---|---|
| 5 tasks × 3 variants × 4 data levels = 60 configs | 60 | 60 × 20 = 1,200 | ~60 hours |
| + iterative N-finding (~5 iterations × 5 tasks × 20 rollouts) | — | 500 | ~25 hours |
| **Total** | **60** | **~1,700** | **~85 hours** |

> ⚠️ **Concern:** 1,200 rollouts + N-finding is a significant commitment (~3 weeks of full-time robot operation). A priority order beyond "plugs first" should be confirmed.

---

## Task 1: Rubber Plug Insertion (Pilot Experiment)

### Setup

Multiple rubber plugs of varying sizes placed in a **cluttered box** on a metal sheet with corresponding holes. The robot must pick a plug from the box and insert it into the correct hole in the metal sheet. This mirrors an **industry setting** where parts are stored in bins and must be placed into fixtures.

- **Robot:** UR5 + Robotiq/Schunk gripper (single-arm)
- **Objects:** 6 plug sizes (sizes 1 through 6), all **black** (no color cue — size discrimination is geometric/tactile only), placed in a single cluttered box
- **Target:** Metal sheet with holes corresponding to each plug size

### Instrumentation

**Button-like contacts embedded in the metal sheet** around each hole. When a plug is fully seated, the contact closes, producing a binary signal. The sheet itself is instrumented (not the plug).

- **Signal type:** Binary (0 = not seated, 1 = seated)
- **Pretraining objective:** Binary cross-entropy (classification)

### Modality hypothesis

**Audio** — the click/snap sound when a plug seats into the hole is distinctive in the spectrogram. Visual cues may be ambiguous (occlusion from gripper, similar appearance before/after seating).

### Success criterion

Circuit closed (instrumentation = success signal). Binary, automatic.

### Generalization axis

**Plug size.** Six sizes total.

| Split | Sizes | Role |
|---|---|---|
| **Train** | 1, 3, 5 | Seen during data collection |
| **Test ID** | 1, 3, 5 | In-distribution evaluation |
| **Test OOD** | 2, 4, 6 | Out-of-distribution evaluation |
| **Test (all)** | 1, 2, 3, 4, 5, 6 | Full evaluation |

### Open items

🔶 Confirm with promotor: mechanical construction of instrumented metal sheet (button contacts around each hole). Need the sheet and work out wiring.
🔶 **Plugs per trial:** does the box contain all 6 sizes cluttered together per trial (robot must pick the correct plug for a target hole), or is a single plug presented per trial? Industry setting implies cluttered bins — confirm.
✅ **Plug color:** all black (no visual color cue — size discrimination is geometric/tactile only).
🔶 Confirm: box dimensions, plug randomization per trial (reshuffled or fixed positions)?

---

## Task 2: Bottle Opening

### Setup

Bottle with a **flicking-switch turning mechanism** for the cap. The robot must grasp the bottle and unscrew/open the cap through rotation.

- **Robot:** UR5 + gripper (primary arm) + **second UR5 holding the bottle** (non-dominant arm)
- **Objects:** Multiple 3D-printed bottle variants
- **Bottle variants:** Same shape, but varying in:
  - **Cap stiffness** (tight vs. loose threading resistance)
  - **Visual appearance** (some with stickers, some without, different colors)
- **Bottle is held by the non-dominant arm** in different positions across episodes. The exact 3D orientation of the cap is known relative to the non-dominant arm's position, which aids scripting of the opening trajectory.

### Instrumentation

**Phototransistor(s)** inside the cap. Provides a continuous signal measuring how open the cap is. Multiple phototransistors may be used for a richer signal.

- **Signal type:** Continuous (normalized to [0, 1] via sensor hardware range)
- **Pretraining objective:** Mean squared error (regression)

### Modality hypothesis

**Image** — cap rotation is visually observable. The encoder learns to correlate visual features (cap position, thread visibility) with the phototransistor reading.

### Success criterion

Phototransistor reading above a predefined threshold = cap fully open. Threshold determined during calibration (e.g., 95th percentile of "fully open" phototransistor readings across all bottle variants).

### Generalization axis

**Bottle variant** (cap stiffness + visual appearance).

| Split | Variants | Role |
|---|---|---|
| **Train** | Subset of variants (e.g., 3 of 6) | Seen during training |
| **Test ID** | Same subset | In-distribution evaluation |
| **Test OOD** | Remaining variants (e.g., 3 of 6) | Out-of-distribution evaluation |
| **Test (all)** | All 6 variants | Full evaluation |

### Open items

🔶 Confirm: how many bottle variants total? Exact train/test split?
🔶 **3D-print design:** spring-based cap mechanism — who designs? (Operator expressed uncertainty about mechanical design confidence.)
🔶 Phototransistor placement details and calibration protocol
🔶 Success threshold value (to be set after calibration across all variants)
✅ **Bottle held by non-dominant arm** in different positions; exact 3D cap orientation known — aids scripting.

---

## Task 3: Petri Dish Rolling

### Setup

A petri dish containing agar is used to sample surfaces by pressing it in a **rolling motion** over target objects. Objects can be flat, round, or arbitrary shapes. The policy receives **target coordinates** (3D point + surface normal) as input so it knows where to position the petri dish.

**Full task sequence:**
1. Navigate to target object/location (3D point + surface normal provided as input)
2. Open petri dish lid
3. Apply rolling motion over target area with consistent force
4. Close petri dish lid

- **Robot:** UR5 + gripper (single-arm)
- **Objects:** Petri dish with agar, various target objects
- **Surfaces:** 6 different materials (see generalization axis)

### Instrumentation

**Load cell** under the sampling surface. Measures contact force during the rolling motion.

- **Signal type:** Continuous (normalized to [0, 1] via sensor hardware range)
- **Pretraining objective:** Mean squared error (regression)

### Modality hypothesis

**Audio** — friction and contact sounds during rolling vary with force and surface. The encoder learns to correlate acoustic features with the load cell reading.

> **Sidenote:** The robot's **internal force-torque (FT) sensor** could be passed through an encoder to **predict the external load cell reading from the internal FT signal**. This tests whether the internal sensor, when properly learned, can substitute for external instrumentation. Optional extension, not part of the core experiment.

### Success criterion

Contact force within an acceptable range [F_min, F_max] during the rolling motion for a sufficient duration. Range defined from demonstration statistics (e.g., mean ± 1 standard deviation of expert force).

### Generalization axis

**Surface material.** Six surfaces, train/test split mirrors the plug task (odd/even):

| # | Surface | Split | Notes |
|---|---|---|---|
| 1 | Tabletop (Thomas) | **Train / Test ID** | Standard lab surface |
| 2 | Wood board with holes | **Test OOD** | Existing lab equipment |
| 3 | Metal table | **Train / Test ID** | Rigid, reflective |
| 4 | Plastic plate | **Test OOD** | Smooth, low friction |
| 5 | White shelf wood | **Train / Test ID** | Painted wood |
| 6 | Glass | **Test OOD** | Smooth, transparent |

### Open items

✅ **Coordinate input format:** 3D point with surface normal (provided as policy input).
🔶 Force range [F_min, F_max] for success — needs empirical determination from demonstrations
🔶 Internal FT sensor as optional side experiment — confirm scope with promotor
🔶 Petri dish handling: how is the lid opened/closed with one gripper?

---

## Task 4: Wiping (S-motion)

### Setup

The robot takes a **tissue or cleaning cloth from a box** and wipes the surface in an **S-shaped motion**. This is a **standalone task**.

- **Robot:** UR5 + gripper (single-arm)
- **Objects:** Tissue/cloth in a dispenser box, surface to clean
- **Motion:** S-shaped wiping trajectory
- **Cloth extraction is learned** (not scripted) — the policy must figure out how to pull the cloth from the box

### Instrumentation

**Vision-based coverage tracking.** An overhead camera monitors a **grid of reference points** (spaced ~5 cm apart) on the wiping area. The camera analyzes the image feed and reports a **progress index** — which grid point the gripper is currently nearest to. The policy learns to visit the next unvisited point in sequence, similar to whack-a-mole with sequential targets.

- **Signal type:** Discrete/sequential (grid index, e.g., 1–25 for a 5×5 grid)
- **Pretraining objective:** Classification (which grid point) or regression (normalized progress 0–1)

### Modality hypothesis

**Image** — coverage is directly visible in the camera feed.

### Success criterion

All grid points covered within a tolerance (gripper passed within X cm of each point during the wiping trajectory).

### Generalization axis

**Surface material** — same 6 surfaces as the petri dish task:

| Split | Surfaces | Role |
|---|---|---|
| **Train** | Tabletop, Metal table, White shelf wood (1, 3, 5) | Seen during training |
| **Test ID** | Tabletop, Metal table, White shelf wood (1, 3, 5) | In-distribution evaluation |
| **Test OOD** | Wood board with holes, Plastic plate, Glass (2, 4, 6) | Out-of-distribution evaluation |

### Open items

🔶 Confirm: grid dimensions (number of points? 5×5 = 25?)
🔶 Confirm: how is "covered" determined? (gripper within what distance of each point?)
✅ **Generalization axis:** surface material — same 6 surfaces as petri dish task
🔶 Is force instrumentation also relevant here (load cell for pressure)? Or vision-only?
✅ **Cloth extraction:** learned by the policy (not scripted)
✅ **Standalone task** — not paired with petri dish

---

## Task 5: Casting Piece Placement (Company ABC)

### Setup

Casting pieces (industrial components) must be placed into a **mold or receptacle** where they need to fit perfectly. The robot must position and seat each piece precisely.

- **Robot:** UR5 + gripper (single-arm)
- **Objects:** Casting pieces, mold/receptacle
- **Constraint:** Pieces must seat flush — no visible gap

### Instrumentation

**Distance sensor** (ultrasonic, laser, or camera-based) measuring the gap between the casting piece and the receptacle surface. Provides a continuous signal indicating how well-seated the piece is.

- **Signal type:** Continuous (gap distance, normalized)
- **Pretraining objective:** Mean squared error (regression)

### Modality hypothesis

**Image** — the gap between piece and receptacle is visually observable (shadows, depth cues).

### Success criterion

Gap below a predefined threshold (piece fully seated). Threshold determined by the manufacturing tolerance of the casting pieces.

### Generalization axis

**TBD** — potential axes:
- Different piece geometries
- Different piece materials
- Different mold/receptacle configurations

### Open items

🔶 Confirm: what kind of casting pieces? (dimensions, material, weight)
🔶 Confirm: manufacturing tolerance for "perfect fit" (success threshold)
🔶 Confirm: generalization axis
🔶 Confirm: is this a priority task or long-term?
🔶 Sensor choice: ultrasonic vs. laser vs. camera-based — trade-offs in precision and cost
🔶 **Concern:** This task feels different from the others — it's about precision placement and force adjustment rather than learning a manipulation trajectory. Does it fit the same experimental framework?

---

## Cross-cutting Concerns

### C1: Data efficiency vs. generalization

The 10 ID + 10 OOD rollout split serves two claims from one experiment:
- **Data efficiency** is reported using ID rollouts only (success rate vs. % data)
- **Generalization** is reported as the ID/OOD gap at each data level
- These are distinct claims and must not be conflated in the narrative

### C2: Scripted demonstrations with diversity

All tasks use **scripted demonstrations**. Despite being scripted, diversity is preserved by parameterizing the trajectories over a distribution of conditions:

- **Bottle opening:** the non-dominant arm holds the bottle in different positions and orientations across episodes. The 3D cap orientation is known and used to adapt the scripted opening trajectory, but the primary arm still encounters varied visual and proprioceptive conditions.
- **Plug insertion:** plugs are randomized in the cluttered box; the robot must pick from different initial configurations each episode.
- **Petri dish / wiping:** target locations and surface materials vary across episodes.

An **additional advantage of instrumentation** is that it provides ground-truth feedback that can *guide* the scripting process itself. For example, the load cell reading can be used to close a feedback loop during scripted force application, the phototransistor confirms when the cap is fully open, and the circuit closure signal tells the script exactly when to end the episode. This makes demonstrations more reliable and reproducible compared to purely open-loop scripted motions.

> **Reviewer risk:** A reviewer might argue that scripted demos produce less diverse data than teleoperation. Mitigate by showing that: (1) diversity is preserved through parameterized conditions, (2) the policy must still handle significant perceptual variation, and (3) instrumentation-assisted scripting produces higher-quality demonstrations than noisy teleop.

### C3: Task scope

Five tasks is ambitious. Recommended priority:
1. **Plugs** (pilot, validates pipeline)
2. **Bottle** (confirms approach on a second task)
3. **Petri dish** (adds force-critical dimension)
4. **Wiping** (adds vision-based instrumentation)
5. **Casting pieces** (long-term, industry partnership)

Consider whether 3 tasks (plugs, bottle, petri dish) are sufficient for a strong paper.

### C4: Internal FT sensor as instrumentation proxy

For the petri dish task, the internal FT sensor could predict the external load cell reading. If successful, it raises the question of whether external instrumentation is always necessary. Mark as an extension, not a core result.

### C5: Reviewer defense — key framing decisions

| Potential reviewer concern | Defense |
|---|---|
| "Pretrained variant sees data twice (pretraining + fine-tuning)" | Pretraining updates only the encoder (~5-20% of total params) and represents < 5% of the encoder's total optimization. Framed as weight initialization, not extra training. |
| "Why IL if everything is scripted?" | Scripting works for individual conditions but doesn't generalize across variations (plug sizes, surfaces, bottle types). IL learns a single policy handling variation. |
| "Modality cherry-picking via suitability metric" | Suitability metric is part of the method. Both modalities' scores reported. Claim: when signal is perceivable in a modality, pretraining helps. |
| "Generic pretraining already shows pretraining helps" | Generic-pretrained variant IS the "any pretraining helps" control. If instrumentation beats generic, the gain comes specifically from the task-relevant signal. |
| "Small N at 25% — can encoder learn anything?" | Suitability metric measures validation performance (not train), so overfitting is already accounted for. If 25% is below learning threshold, report in supplementary. |
| "OOD rollouts underpowered (n=10)" | Acknowledged as limitation. Results interpreted as directional evidence. Real-robot rollout costs constrain sample size. |

---

## Reminders

🔶 Check: force tolerance for wiping/petri dish (if any force works → task too easy)

🔶 **Can audio disambiguate visually identical materials?** This question arises when two surfaces look the same but require different forces or produce different contact dynamics.
- **Concrete example:** A *white painted wood board* and a *white plastic plate* may appear nearly identical to the camera under the same lighting. However, when the petri dish rolls over each surface, the **acoustic signature differs**: wood produces a duller, lower-frequency thud with more energy in the 200–800 Hz range, while plastic produces a sharper, higher-frequency sound with more energy above 1 kHz. Similarly, during the wiping task, a cloth dragged over painted wood generates a rougher, lower-pitched friction sound compared to the smoother, higher-pitched sound of the same cloth on plastic. If the audio encoder is pretrained to predict the load cell force reading, it must learn to associate these distinct acoustic patterns with the correct force profile — even when the visual input is ambiguous. A suitability check (pretrain audio encoder on validation set: can it predict force from audio alone when visual input is held constant?) would confirm whether this disambiguation is possible.

🔶 Implement: instrumentation hardware for all tasks
🔶 Implement: suitability metric pipeline (pretrain image vs audio, compare)
🔶 Implement: scripted demonstration framework for all tasks
