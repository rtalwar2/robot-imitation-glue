# RA-L Journal Extension Plan (Instrumentation-Centered, Mechanism Diversity Scope)

## A. Executive Summary
This journal extension stays fully aligned with the workshop thesis: instrumentation improves robotic manipulation task learning by improving data collection, supervision quality, and interaction behavior. The extension does not broaden to unrelated tasks. Instead, it addresses the current scope limitation through mechanism diversity under one common task family (instrumented button/contact pressing).

Core strategy:
- Keep one task template: contact-rich press-and-retract behavior.
- Expand to three mechanism variants with distinct force-travel/acoustic properties.
- Keep modality analysis as transfer of privileged instrumentation supervision to deployable sensing (audio vs click/no-click vision), not as a standalone multimodal paper.
- Use a realistic confirmatory budget: 50 rollouts per key model.
- Use a seed protocol that avoids 5x evaluation explosion.

Submission target: a compact, evidence-dense RA-L paper where each claim has a direct experiment, a falsification criterion, and statistically credible reporting.

---

## B. Current Workshop Claim, Evidence, and Limitations

### Current claim (already established at workshop level)
- Instrumentation enables cleaner automated demonstration collection.
- Privileged instrumentation signals improve learning and interaction quality.
- Distilling privileged supervision into deployable sensing is feasible.
- Audio-informed policies can improve contact quality while success gains remain modest.

### Strong evidence already available from current analysis
- Success-rate comparisons across multiple policy variants.
- Force-quality analyses (Fz distributions, violin/box plots, median-based comparisons).
- Pairwise nonparametric tests and CLD grouping.
- Distributional similarity analyses vs expert (Wasserstein/JS/KL/Energy).
- Bayesian posterior intervals for success rates.

### Main limitations to address for RA-L
- Scope narrowness (single mechanism dominates narrative).
- Statistical power and seed stability for small success deltas.
- Incomplete causal chain proof from demonstration quality to downstream policy quality.
- Modality comparison not yet fully controlled/fair in journal-review terms.

---

## C. Research Questions
1. Does instrumentation improve downstream policy behavior primarily through demonstration quality, privileged supervision, or both?
2. Can privileged instrumentation supervision be transferred to deployable modalities while preserving contact-quality benefits?
3. Under identical supervision and training budgets, is audio a stronger deployable contact cue than click/no-click vision?
4. Do instrumentation-derived gains persist across mechanism diversity within the same task family?
5. Are gains concentrated in success, interaction gentleness, or both?
6. How robust are conclusions under realistic sample and seed constraints?

---

## D. Hypotheses
- H1: Instrumented collection reduces demonstration noise (timing jitter, force overshoot, retries) versus non-instrumented collection.
- H2: Policies trained on instrumented demonstrations show better force behavior and at least non-inferior success compared with matched non-instrumented datasets.
- H3: Distilled deployable signals improve over vision-only baseline for contact-centric behavior.
- H4: Audio distillation outperforms click/no-click vision distillation on contact-event sensitivity under occlusion/mechanism variability.
- H5: Instrumentation benefits persist across three mechanism variants.
- H6: If shuffled/delayed privileged labels do not degrade performance, the claimed supervision mechanism is not causal.

---

## E. Recommended Journal Narrative

### Narrative spine
Instrumentation is the main contribution. Modality distillation is the mechanism that transfers instrumentation benefit into deployable inference-time sensing.

### Recommended claim hierarchy
1. Instrumentation improves the learning pipeline quality.
2. This benefit can be transferred to deployable modalities.
3. Mechanism diversity demonstrates non-trivial scope beyond a single button implementation.
4. Improvements are strongest in contact quality and safety-relevant interaction behavior, with bounded success-rate gains.

### Explicit non-claims (important for reviewer trust)
- Do not claim universal modality superiority across all manipulation tasks.
- Do not claim broad generalization beyond the defined mechanism family.
- Do not claim large success-rate improvements unless statistically supported.

---

## F. Refined Paper Contributions
1. An instrumentation-centered manipulation learning pipeline integrating automated demonstration collection, privileged supervision, and deployable policy conditioning.
2. A controlled modality-distillation comparison (audio vs click/no-click vision) under identical privileged labels and training budgets.
3. A mechanism-diversity validation across three contact mechanisms under a common task template.
4. A causal demonstration-quality analysis linking dataset quality improvements to downstream interaction quality.
5. A statistics-forward evaluation protocol designed for realistic robot budgets (50 rollouts per key model with seed-aware reporting).

---

## G. Prioritized Experiment Plan

### G1. Essential Experiments (must have)

#### E1. Confirmatory core benchmark (mechanism-aggregated)
- Claim tested: instrumentation-guided variants outperform vision-only in contact quality and are closer to oracle behavior.
- Why it matters: anchors continuity with workshop and creates powered journal baseline.
- Setup: 3 mechanisms, 4 key models (Vision-only, Oracle, Audio-distilled, Vision-click-distilled).
- Controls: identical training budget, data splits, preprocessing, rollout protocol.
- Metrics: success rate, peak |Fz|, force impulse, false trigger rate.
- Supports claim if: consistent directional improvement and non-trivial effect sizes across mechanisms.
- Weakens claim if: no consistent gains over vision-only or unstable ranking.
- Priority: Essential.

#### E2. Demonstration quality characterization
- Claim tested: instrumentation produces objectively better demonstrations.
- Why it matters: validates pipeline contribution independent of policy architecture.
- Setup: per mechanism, collect matched demo pools: instrumented-auto vs weak/manual baseline.
- Controls: same number of demos, similar initial states.
- Metrics: contact-time jitter, retries, force overshoot, annotation uncertainty.
- Supports claim if: instrumented pool has lower variance/error on quality metrics.
- Weakens claim if: quality metrics are indistinguishable.
- Priority: Essential.

#### E3. Demonstration quality -> policy outcome causality
- Claim tested: better demos lead to better learned behavior.
- Why it matters: closes causal chain for reviewers.
- Setup: train same policy on demo pools from E2.
- Controls: same seeds, architecture, optimizer, epochs.
- Metrics: success, force metrics, failure mode composition.
- Supports claim if: instrumented-demo-trained policies show better interaction quality and/or success.
- Weakens claim if: no downstream difference.
- Priority: Essential.

#### E4. Fair modality distillation comparison
- Claim tested: audio is stronger than click/no-click vision as deployable distillation channel for contact cues.
- Why it matters: central mechanism-transfer question.
- Setup: same privileged labels, same context window, same compute, same policy backbone.
- Controls: synchronized splits, matched parameter budgets, matched latency constraints.
- Metrics: success, false triggers, contact timing error, force quality.
- Supports claim if: audio better on contact-centric metrics and robustness.
- Weakens claim if: click/no-click vision equals or exceeds audio across key metrics.
- Priority: Essential.

#### E5. Negative control: shuffled and delayed supervision
- Claim tested: privileged supervision quality is causally necessary.
- Why it matters: de-risks confound/leakage objections.
- Setup: shuffled labels and temporal shifts (+/- k frames).
- Controls: all else fixed.
- Metrics: same as E4.
- Supports claim if: clear degradation with corruption/delay.
- Weakens claim if: performance unchanged.
- Priority: Essential.

### G2. High-Value Ablations (should have)

#### E6. Data regime scaling
- Claim tested: instrumentation lowers sample complexity.
- Setup: 25%, 50%, 100% training demos.
- Metrics: success and force metrics vs data budget.
- Priority: High-value.

#### E7. Distillation target ablation
- Claim tested: event-level vs embedding-level supervision impacts transfer quality.
- Setup: binary event head vs representation distillation head.
- Metrics: false triggers, timing error, force outcomes.
- Priority: High-value.

### G3. Generalization within mechanism diversity (must include at least one)

#### E8. Leave-one-mechanism-out transfer
- Claim tested: learned supervision transfer is not mechanism-specific.
- Setup: train on 2 mechanisms, test on held-out mechanism.
- Metrics: zero-shot/few-shot success and force quality.
- Priority: High-value (close to essential).

### G4. Optional stretch (only if budget remains)
- E9: Mild perturbation robustness (ambient noise, mild visual occlusion) under fixed mechanisms.
- E10: Audio+vision fused distillation as complementary model.

---

## H. Modality Comparison Plan (Dedicated)

### Fairness protocol
- Same supervision source: identical instrumentation-derived labels.
- Same train/val/test split per mechanism.
- Same policy backbone and optimizer settings.
- Same temporal window and inference budget.
- Same rollout conditions and randomization.

### Models to compare
- Vision-only baseline.
- Oracle privileged signal baseline.
- Audio-distilled deployable model.
- Click/no-click vision-distilled deployable model.
- Shuffled-label negative control.

### Required baselines for interpretation
- If audio > vision-click: supports modality-specific contact sensitivity claim.
- If audio ~= vision-click: supports deployable distillation claim without modality superiority.
- If both < vision-only: indicates transfer path failure, not instrumentation invalidity.

### Likely failure modes and diagnostics
- Audio overfits to robot motor noise.
- Vision-click overfits to viewpoint/background texture.
- Temporal misalignment between inferred contact and control action.
- Mechanism-specific cue memorization.

Diagnostics:
- Error-conditioned confusion around contact transition frames.
- Per-mechanism calibration curves for event detection.
- False positive/false negative breakdown tied to force outcomes.

---

## I. Demonstration-Quality Evaluation Plan (Dedicated)

### What to measure in demonstration quality
- Contact transition timing consistency.
- Peak overshoot after first contact.
- Number of retries/corrections per episode.
- Label cleanliness (automatic instrumentation labels vs manual labels).

### Causal protocol
1. Build two matched datasets per mechanism:
   - Instrumented-auto demonstrations.
   - Non-instrumented baseline collection.
2. Train identical policy variants on each dataset.
3. Compare downstream behavior on same test episodes.

### Expected evidence pattern
- Better demo quality metrics in instrumented pool.
- Better force regularization and reduced false triggers downstream.
- Success-rate gains may be modest; force-quality gains should be clearer.

---

## J. Generalization / Scope-Expansion Plan (Mechanism Diversity Only)

### Scope definition
No additional task family. Generalization claim is restricted to mechanism diversity under one contact-rich pressing template.

### Mechanism set
- M1: Clicky tactile button (distinct click impulse).
- M2: Damped/quiet mechanism (reduced acoustic saliency, different force profile).
- M3: Latching/toggle mechanism (different travel and release behavior).

### Why this is sufficient
- Practical instrumentation remains easy.
- Preserves continuity with workshop setup.
- Directly addresses the "one button" criticism with bounded, realistic expansion.

### Impact-to-effort rationale
- High impact: demonstrates non-trivial breadth.
- Low-to-moderate effort: no full redesign of task stack.
- High comparability: shared pipeline and metrics across mechanisms.

---

## K. Statistical Analysis Plan

### Primary endpoints (pre-registered before confirmatory runs)
1. Success rate.
2. Peak |Fz| after contact (or force impulse).
3. False trigger rate.

### Secondary endpoints
- Contact timing error.
- Wasserstein and JS distance to expert force distribution.
- Failure mode composition.

### Inference methods
- Success: Bayesian Beta-Binomial intervals + frequentist proportion tests.
- Force/distribution metrics: bootstrap CIs, Mann-Whitney/permutation tests.
- Multiple comparisons: Holm-Bonferroni.
- Effect sizes: rank-biserial or Cliff delta for nonparametric contrasts.

### Seed and rollout protocol under your budget
- Key models: train 5 seeds.
- Evaluate with fixed total budget of 50 rollouts per model using split evaluation:
  - 10 rollouts per seed x 5 seeds = 50/model.
- Report seed-aware aggregates (mean and dispersion across seeds, and pooled uncertainty).

### Why this solves workload concerns
- Avoids 250 rollouts per model.
- Avoids cherry-picking single-seed results.
- Keeps evaluation realistic and statistically defensible.

---

## L. Reviewer-Risk Analysis

### Likely objections
- "This is incremental from the workshop."
- "Still narrow in scope."
- "Small success gains are underpowered."
- "Modality comparison is unfair."
- "No causal proof that instrumentation helps learning pipeline."

### De-risk mapping
- Incremental concern -> E2/E3 causal demo-quality chain + E8 transfer.
- Narrow scope concern -> 3-mechanism validation.
- Underpowered concern -> 50-rollout confirmatory with seed-aware protocol.
- Fairness concern -> E4 strict parity setup.
- Causality concern -> E5 corrupted-supervision controls.

### Overclaims to avoid
- Universal audio superiority.
- Broad manipulation generalization beyond mechanism family.
- Claims of large success gains without effect-size and uncertainty support.

---

## M. Section-by-Section Paper Outline
1. Introduction
- Instrumentation-centered manipulation learning problem and contributions.

2. Related Work
- Instrumented manipulation, privileged supervision, deployable distillation.

3. Method
- Instrumentation pipeline, supervision extraction, distillation variants.

4. Experimental Setup
- Robot, mechanisms M1-M3, datasets, model families, evaluation protocol.

5. Main Results
- Core benchmark and modality comparison across mechanisms.

6. Demonstration-Quality Causal Study
- Dataset quality metrics and downstream policy effects.

7. Generalization Across Mechanisms
- Leave-one-mechanism-out and pooled analyses.

8. Controls and Negative Results
- Shuffled and delayed supervision outcomes.

9. Discussion and Limitations
- Boundary conditions, practical deployment implications, non-claims.

10. Conclusion
- Instrumentation value proposition and next-step outlook.

11. Supplementary
- Full stats tables, per-seed plots, additional diagnostics.

---

## N. Priority-Ordered Execution Roadmap

### Phase 1 (must do first)
- Finalize mechanism hardware definitions (M1-M3) and shared logging schema.
- Generate click/no-click vision pretraining dataset from instrumentation labels.
- Lock pre-registered primary endpoints.

### Phase 2 (must do)
- Run E1, E2, E3, E4, E5 with 5-seed training and 50-rollout/model evaluation.
- Produce core tables and figures for success and force-quality claims.

### Phase 3 (strongly recommended)
- Run E6 and E8 to strengthen sample-efficiency and mechanism-transfer claims.
- Add one robustness perturbation if budget allows.

### Phase 4 (optional)
- Fusion model and extended diagnostics only if Phase 2+3 results are stable.

### Minimum credible submission package
- E1 + E2 + E3 + E4 + E5 + mechanism diversity (M1-M3) + seed-aware statistics.

---

## O. Final Clarifications
No additional questions are required to execute this plan. Scope, rollout budget, and modality-comparison feasibility are already sufficient to proceed.
