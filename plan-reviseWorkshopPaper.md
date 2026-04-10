# Plan: Revise Workshop Paper with Corrected Data & Narrative

## TL;DR
Rewrite conference.tex with (1) corrected experimental numbers from analyze_policy_clean.ipynb, (2) new narrative about instrumentation enabling automated data collection, (3) tool-frame action space, (4) additional failure categories, (5) tropes.md-compliant writing style.

## Key Findings from Research

### Corrected Results (from analyze_policy_clean.ipynb cell 2)

| # | Model | Hover | Too Hard | Success | Sideways | Fake | Rate |
|---|-------|-------|----------|---------|----------|------|------|
| 1 | Expert | 0 | 0 | 40 | 0 | 0 | 100% |
| 2 | Vision Only | 16 | 0 | 22 | 2 | 0 | 55.0% |
| 3 | Oracle (Vision + Button) | 15 | 0 | 25 | 0 | 0 | 62.5% |
| 4 | V + Button + Generic AST Frozen | 21 | 0 | 18 | 1 | 0 | 45.0% |
| 5 | V + Button + Generic AST Unfrozen | 15 | 0 | 25 | 0 | 0 | 62.5% |
| 6 | Generic AST Frozen Embed | 21 | 0 | 19 | 0 | 0 | 47.5% |
| 7 | Generic AST Unfrozen Embed | 24 | 0 | 16 | 0 | 0 | 40.0% |
| 8 | Pseudo-Instr (Signal Swap) | 17 | 0 | 22 | 0 | 1 | 55.0% |
| 9 | Button AST Frozen Classif | 17 | 0 | 23 | 0 | 0 | 57.5% |
| 10 | Button AST Unfrozen Classif | 23 | 0 | 17 | 0 | 0 | 42.5% |
| 11 | Button AST Frozen Embed | 14 | 3 | 18 | 1 | 4 | 45.0% |
| 12 | Button AST Unfrozen Embed | 17 | 0 | 23 | 0 | 0 | 57.5% |

### Intermediate Layer Ablation (cell 8)
| Model | Hover | Success | Sideways | Rate |
|-------|-------|---------|----------|------|
| Vision Only | 16 | 22 | 2 | 55.0% |
| Generic AST Frozen | 21 | 19 | 0 | 47.5% |
| Generic AST Frozen + layer | 22 | 17 | 1 | 42.5% |
| Generic AST Unfrozen | 24 | 16 | 0 | 40.0% |
| Generic AST Unfrozen + layer | 24 | 16 | 0 | 40.0% |

### Action Space (from collect_data_delta.py)
- Actions are 3D deltas in TOOL frame (not base frame as paper currently says)
- `create_action()`: computes vector from TCP to button in tool space via R_tool^{-1} * (target - tcp_pos)
- `action_to_tcp_pose()`: converts tool-frame action back to base frame via R_tool @ action

### Automated Data Collection (3-phase system in collect_data_xyz)
- Phase 1: XY alignment with proportional control (terminates at lateral_error < 2mm)
- Phase 2: Z descent monitoring button (terminates when btn_state == 0)
- Phase 3: Fast retraction 2cm/step (terminates when z_dist >= 10cm above button)
- Random style sampling: approach_style = exp(uniform(ln(0.1), ln(10.0)))
  - low style = aggressive Z descent; high style = careful XY first
- Hyperparameters (MIN_STEP=2mm, MAX_STEP=3cm, BASE_SPEED_GAIN=0.1, BASE_LATERAL_GAIN=0.1) tuned in simulation.py
- Randomized initial poses: ±5cm XY, 0-5cm Z, ±90° rotation around Z

### Three Advantages of Instrumentation
1. Enables automated/scripted expert demonstrations (no human teleop needed)
2. Cleaner demonstrations with minimal force (better than human teleop)
3. Privileged knowledge can be distilled into non-privileged modalities (audio)

### Additional Failure Categories
- **Sideways**: Gripper approached from wrong angle
- **Fake**: Audio predicted button press but it wasn't actually pressed (false positive from AST)

### Force Analysis
- Bayesian credible intervals computed (Beta(k+1, n-k+1) posterior with uniform prior)
- Mann-Whitney U pairwise tests with Bonferroni correction
- Compact Letter Display (CLD) for statistical grouping
- Force data calibrated at t=-10 before button press, filtered to exclude "Too Hard" and "Sideways" episodes

## Steps

### Phase 1: Fix Critical Data Errors
1. Update abstract with correct numbers: vision-only=55%, best audio=57.5%, oracle=62.5%
2. Rewrite Table I with all correct counts including Sideways and Fake columns
3. Update Table II (intermediate layer) with correct numbers
4. Fix action space description: tool frame, not base frame

### Phase 2: Add Automated Data Collection Section
5. Replace the "demonstrations via GELLO kinesthetic teaching" with automated scripted collection description
6. Add subsection describing the 3-phase automated collection system
7. Describe how button instrumentation enables this automation
8. Mention simulation-based hyperparameter tuning

### Phase 3: Restructure Narrative (Three Advantages of Instrumentation)
9. Restructure Introduction to frame three advantages of instrumentation:
   - (a) enables automated expert data collection
   - (b) produces cleaner demonstrations than human teleop
   - (c) privileged knowledge can be distilled to cheaper modalities (audio)
10. The paper's contribution is showing advantage (c): distilling instrumentation knowledge into audio

### Phase 4: Fix Results Discussion
11. Rewrite results analysis around the correct numbers
12. Note: NO "too hard" failures for vision-only baseline (key change from draft)
13. Add discussion of "Fake" and "Sideways" failure modes
14. Update discussion of force analysis (Bayesian posteriors, CLD grouping)
15. Best audio-only model is Button AST Frozen/Unfrozen Classif/Embed at 57.5% (not 60%)

### Phase 5: Style Pass (tropes.md)
16. Remove "It's not X -- it's Y" patterns
17. Avoid "delve", "leverage", "robust", "notably", "interestingly"
18. Avoid patronizing analogies and false profundity
19. Keep it direct and specific, no grandiose stakes inflation
20. No signposted conclusions ("In conclusion...")
21. No bold-first bullets in text

## Relevant Files
- `conference.tex` — Full rewrite needed with all changes above
- `analyze_policy_clean.ipynb` — Source of truth for results data (cells 2, 4, 8)
- `robot_imitation_glue/collect_data_delta.py` — `collect_data_xyz()`, `create_action()`, `action_to_tcp_pose()`
- `simulation.py` — Hyperparameter tuning reference for data collection
- `tropes.md` — AI writing anti-patterns to avoid

## Verification
1. Cross-check every number in the tables against the experiments dict in analyze_policy_clean.ipynb
2. Verify action space description matches create_action / action_to_tcp_pose code
3. Verify data collection description matches collect_data_xyz implementation
4. Run tropes.md checklist against final text
5. Ensure paper compiles (pdflatex conference.tex)

## Decisions
- Action space: deltas in tool frame (not base frame) — corrected from first draft
- Vision-only baseline: 55% not 42.5% — major correction
- No "Too Hard" failures for vision-only — changes the narrative (can't claim audio eliminates "too hard" as the main benefit)
- New narrative: instrumentation enables automation + distillation to audio
- Include Sideways and Fake columns in results table
- Keep Bayesian credible intervals from notebook in discussion
