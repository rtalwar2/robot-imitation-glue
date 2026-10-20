"""Evaluate checkpoints in the plug sim: does the policy insert into the hole the goal input names, including holes
it never trained on?

    python eval.py --checkpoint <pretrained_model> --rollouts 50 [--conditions train test_a test_b goal_noise]

Conditions (the same seeded scenes for every checkpoint, so arms are paired rollout by rollout):
  train       a train hole as target, new sheet poses / starts         (in distribution)
  test_a      a held-out hole on a panel angle that has train holes     (new location)
  test_b      the held-out hole on the -25 deg panel (no train hole)    (new approach angle)
  goal_noise  as `train`, but the goal input's position carries N(0, 5 mm) per axis, fixed per episode
              (a fiducial-localisation error; the sim's true hole is unchanged)
Rules fixed from the 200 demos before any policy was evaluated (as the bottle protocol does):
  timeout 25 s = 2x the median demo (11.15 s) rounded up to 5 s;
  force abort 48 N = 2x the p95 per-episode peak |F| of the demos (24.1 N).
Outcomes: success (seated in the TARGET hole for 5 steps), wrong_hole (seated in another hole for 5 steps),
force_abort, timeout. One JSON row per rollout is appended to --results.
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"  # before mujoco is imported anywhere: GLFW would render on the CPU via the X display

import argparse
import json
import time
from pathlib import Path

import numpy as np

from peg_env import CONTROL_HZ, SPLITS, PlugEnv, sample_episode
from policy_runner import PolicyRunner

TIMEOUT_S = 25
FORCE_ABORT_N = 48.0
SUSTAIN_STEPS = 5
GOAL_NOISE_M = 0.005
SEED_BASE = {"train": 100_000, "test_a": 200_000, "test_b": 300_000, "goal_noise": 100_000}


def rollout(env: PlugEnv, policy: PolicyRunner, spec, goal_noise: np.ndarray | None) -> dict:
    policy.reset()
    obs = env.reset(spec)
    k = spec.target_hole
    seated_run, seated_hole, aux, peak_f, min_dist = 0, None, [], 0.0, np.inf
    outcome, steps = "timeout", 0
    for steps in range(1, TIMEOUT_S * CONTROL_HZ + 1):
        if goal_noise is not None:  # perturb only what the policy sees: goal = noisy hole position in the tool frame
            p, R = env.tcp()
            obs["state"][12:15] = (R.T @ (env.hole_pw[k] + goal_noise - p)).astype(np.float32)
        obs = env.step(policy(obs))
        if policy.last_aux is not None:
            aux.append(float(policy.last_aux[0]))
        f = env.force_magnitude()
        peak_f = max(peak_f, f)
        min_dist = min(min_dist, float(np.linalg.norm(env.tcp()[0] - env.hole_pw[k])))
        if f > FORCE_ABORT_N:
            outcome = "force_abort"
            break
        h = env.inserted_hole()
        seated_run = seated_run + 1 if (h is not None and h == seated_hole) else (1 if h is not None else 0)
        seated_hole = h
        if seated_run >= SUSTAIN_STEPS:
            outcome = "success" if h == k else "wrong_hole"
            break
    row = {"outcome": outcome, "success": outcome == "success", "steps": steps, "target_hole": k,
           "seated_hole": seated_hole, "peak_force": round(peak_f, 2), "min_dist_to_target": round(min_dist, 4),
           "popped_target": bool(env.popped[k])}
    if aux:
        row["aux_seated_mean_last5"] = round(float(np.mean(aux[-5:])), 3)
        row["aux_seated_max"] = round(float(np.max(aux)), 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--name", default=None, help="model name in the results rows (default: the run dir name)")
    ap.add_argument("--rollouts", type=int, default=50)
    ap.add_argument("--conditions", nargs="+", default=["train", "test_a", "test_b", "goal_noise"])
    ap.add_argument("--results", type=Path, default=Path(__file__).parents[1] / "outputs/peg_sim/eval_results.jsonl")
    args = ap.parse_args()
    name = args.name or next((p.name for p in args.checkpoint.resolve().parents if p.parent.name == "peg_sim"),
                             args.checkpoint.resolve().parent.name)

    env = PlugEnv()
    policy = PolicyRunner(args.checkpoint)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    for cond in args.conditions:
        holes = SPLITS["train" if cond == "goal_noise" else cond]
        counts: dict[str, int] = {}
        t0 = time.time()
        for i in range(args.rollouts):
            seed = SEED_BASE[cond] + i
            spec = sample_episode(seed, holes)
            noise = np.random.default_rng(seed + 7).normal(0, GOAL_NOISE_M, 3) if cond == "goal_noise" else None
            row = {"model": name, "condition": cond, "seed": seed, "rollout": i, **rollout(env, policy, spec, noise)}
            with args.results.open("a") as f:
                f.write(json.dumps(row) + "\n")
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        ok = counts.get("success", 0)
        print(f"{name:14s} {cond:10s} {ok}/{args.rollouts} = {100 * ok / args.rollouts:.0f}%  {counts}  "
              f"({(time.time() - t0) / args.rollouts:.1f} s/rollout)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
