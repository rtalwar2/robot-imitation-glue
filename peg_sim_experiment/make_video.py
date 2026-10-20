"""Render episodes side by side: third-person view (target hole marked green) | wrist camera (what the policy sees).

    python make_video.py --out outputs/peg_sim/videos/expert.mp4 --seeds 0 1 2 3 --holes train
    python make_video.py --out ... --checkpoint <pretrained_model>   # a policy instead of the expert

The green marker is in geom group 3, which only this script's scene camera renders.
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"  # before mujoco is imported anywhere: GLFW would render on the CPU via the X display

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from expert import Expert
from peg_env import SPLITS, PlugEnv, sample_episode

H = 360


def _upscale(img: np.ndarray, h: int) -> np.ndarray:
    w = h * img.shape[1] // img.shape[0]  # nearest neighbour, keeps the policy's pixels visible
    return img[np.arange(h) * img.shape[0] // h][:, np.arange(w) * img.shape[1] // w]


FORCE_H, FORCE_MAX = 90, 40.0  # strip height (px) and full-scale axial force (N)


def _force_strip(forces: list[float], width: int, max_steps: int) -> np.ndarray:
    """Axial force on the plug (N, positive = pushing it back out) so far, as a white trace; grey line = 15 N peak."""
    strip = np.full((FORCE_H, width, 3), 30, np.uint8)
    y15 = FORCE_H - 1 - int(15.0 / FORCE_MAX * (FORCE_H - 1))
    strip[y15, :] = 90
    xs = (np.arange(len(forces)) * (width - 1) / max(max_steps - 1, 1)).astype(int)
    ys = FORCE_H - 1 - (np.clip(forces, 0, FORCE_MAX) / FORCE_MAX * (FORCE_H - 1)).astype(int)
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        for t in np.linspace(0, 1, max(abs(y1 - y0), abs(x1 - x0), 1) + 1):
            strip[int(round(y0 + t * (y1 - y0))), int(round(x0 + t * (x1 - x0)))] = 255
    return strip


def _label(frame: np.ndarray, color) -> np.ndarray:
    frame[:6] = color
    return frame


def _aim_scene_camera(env: PlugEnv, look_at: np.ndarray) -> None:
    """Point the overview camera at this episode's sheet from a fixed offset (video only; the policy never sees it)."""
    pos = look_at + np.array([0.40, -0.45, 0.42])
    f = (look_at + np.array([0, 0, 0.08]) - pos) / np.linalg.norm(look_at + np.array([0, 0, 0.08]) - pos)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    cam = env.model.camera("scene").id
    env.model.cam_pos[cam] = pos
    env.model.cam_quat[cam] = np.roll(Rotation.from_matrix(np.column_stack([x, y, -f])).as_quat(), 1)


def render_episode(env: PlugEnv, scene: mujoco.Renderer, opt: mujoco.MjvOption, spec, policy=None, max_steps=250):
    obs = env.reset(spec)
    hp, hn = env.hole_pw, env.hole_nw
    mk = env.model.body("marker").mocapid[0]
    env.data.mocap_pos[mk] = hp[spec.target_hole] + 0.0005 * hn[spec.target_hole]
    z = np.array([0, 0, 1.0])
    env.data.mocap_quat[mk] = np.roll(Rotation.align_vectors([hn[spec.target_hole]], [z])[0].as_quat(), 1)
    _aim_scene_camera(env, spec.sheet_pos)
    mujoco.mj_forward(env.model, env.data)
    expert = Expert(env, np.random.default_rng(spec.seed)) if policy is None else None
    frames, outcome, forces = [], "timeout", []
    for _ in range(max_steps):
        scene.update_scene(env.data, "scene", scene_option=opt)
        top = np.concatenate([scene.render(), _upscale(obs["image"], H)], 1)
        forces.append(float(env.wrench[:3] @ env.hole_nw[spec.target_hole]))
        frames.append(np.concatenate([top, _force_strip(forces, top.shape[1], max_steps)], 0))
        if expert is not None:
            _, executed, done = expert.act()
        else:
            executed, done = policy(obs), False
        obs = env.step(executed)
        if done or (policy is not None and env.inserted_hole() is not None):
            break
    got = env.inserted_hole()
    outcome = "success" if got == spec.target_hole else ("wrong hole" if got is not None else outcome)
    color = {"success": (40, 200, 60), "wrong hole": (230, 60, 40), "timeout": (230, 160, 30)}[outcome]
    frames = [_label(f, color) for f in frames] + [_label(frames[-1].copy(), color)] * 10
    return frames, outcome, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--holes", choices=list(SPLITS), default="train")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()

    env = PlugEnv()
    scene = mujoco.Renderer(env.model, H, 480)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    policy = None
    if args.checkpoint is not None:
        from policy_runner import PolicyRunner
        policy = PolicyRunner(args.checkpoint)
    holes = SPLITS[args.holes]
    all_frames = []
    for seed in args.seeds:
        if policy is not None:
            policy.reset()
        spec = sample_episode(seed, holes)
        frames, outcome, got = render_episode(env, scene, opt, spec, policy)
        print(f"seed {seed}: target {spec.target_hole}, inserted {got} -> {outcome}, {len(frames) - 10} steps")
        all_frames += frames
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(all_frames), plugin="pyav", codec="libx264", fps=10)
    print(f"wrote {args.out} ({len(all_frames)} frames)")
    scene.close()
    env.close()


if __name__ == "__main__":
    main()
