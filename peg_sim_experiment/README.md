# Plug-in-hole sim: does a goal-conditioned policy insert into a hole it never trained on?

A go/no-go check before the real plug task. If the bottle pipeline's Diffusion Policy cannot use a target-hole
input to reach and insert into an unseen hole in simulation, it will not do so on the robot either. It also gives
a first look at the instrumentation question (the sheet's "seated" contact as an auxiliary target) on this task.

## Setup

- **Scene** (`peg_env.py`): a faceted car-body sheet with five flat panels at +15/0/−10/−25/−40°, 40 × 70 cm, and
  10 identical holes at least 12 cm apart. 6 are **train** holes. 3 are **test_a** holes (unseen location on a
  panel angle that has train holes). 1 is the **test_b** hole (on the −25° panel, which has no train hole). The
  sheet moves ±30 cm and yaws ±45° per episode; only reachable holes are targets.
- **Plug**: a rubber plug with a lip and a flange. The snap-through is an explicit force law: the lip resists up
  to 15 N, and if the plug is aligned (< 4° tilt) it pops and the resistance drops to 2.5 N, then the flange
  seats. Misaligned, the force climbs (a jam).
- **Arm**: the plug is held by a mocap target through a soft weld, with gravity compensation. The wrist camera
  is placed like the bottle rig's: fingers at the bottom of a 160×120 frame, 58° vertical field of view.
- **Policy inputs**: wrist image plus `observation.state`, 18 dims:
  - tool pose (6);
  - wrist force/torque (6);
  - **goal** (6): the target hole's centre and normal in the current tool frame.
- **Actions**: relative pose chunks, the same as the bottle (`use_relative_actions`).
- **Expert** (`expert.py`): a closed-loop funnel onto the hole axis (12 cm/s far, 1 cm/s near), a slow push
  through the lip, and a retry on a jam. Smooth, correlated noise is added during the approach; the clean target
  is what gets recorded.

## Arms (`make_configs.py`, derived from `bottle_generic_100.json`; parity asserted)

| Run | Action | Instrumentation |
|---|---|---|
| `peg_200` | 9 | none (control) |
| `peg_c_200` | 9 + `seated` | the sheet contact as an auxiliary target, sliced off at inference |

Both use 200 demos on train holes, 150K steps, batch 32, AdamW, ImageNet ResNet-18, no audio.

## Pipeline

MuJoCo is installed beside the conda env, not into it:
`pip install --target <dir> mujoco==3.3.7 --no-deps` plus glfw, absl-py, etils and pyopengl; put `<dir>` on
`PYTHONPATH`. Rendering must use EGL (the scripts set `MUJOCO_GL=egl`); GLFW over the X display renders on the
CPU at about 20× the cost.

```bash
python collect.py --root datasets/peg_sim/peg_train_200 --episodes 200 --max-steps 300
python make_instrumented.py --src datasets/peg_sim/peg_train_200 --dst datasets/peg_sim/peg_train_200_c
python make_configs.py --dataset peg_train_200
eval "$(bash ~/.claude/skills/gpulab/bottle/gateway.sh)"; bash submit_gpulab.sh peg_200; bash submit_gpulab.sh peg_c_200
python eval.py --checkpoint <pretrained_model> --rollouts 50     # -> outputs/peg_sim/eval_results.jsonl
python make_video.py --out outputs/peg_sim/videos/x.mp4 --holes test_a --checkpoint <pretrained_model>
```

## Evaluation rules (fixed from the demos before any policy was run)

- Timeout: 25 s (2 × the 11.15 s median demo, rounded up to 5 s).
- Force abort: 48 N (2 × the demos' p95 per-episode peak, 24.1 N).
- Success: seated in the **target** hole for 5 steps. Seated in another hole counts as `wrong_hole`.
- Conditions:
  - `train`, `test_a`, `test_b`;
  - `goal_noise`: train holes with N(0, 5 mm) on the goal position, mimicking a fiducial localization error.
  - The scenes are seeded identically for every checkpoint, so the arms are paired rollout by rollout.
