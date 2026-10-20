"""Closed-loop scripted expert for the plug sim: a smooth funnel onto the hole axis, then a slow push through the lip.

Every control step it re-plans from the measured tool pose:
- lateral: converge onto the hole axis with time constant LATERAL_TAU (speed-capped);
- axial: stay above a cone of slope CONE_SLOPE around the axis (descend only as fast as the lateral error
  closes, rise if below the cone), so the motion is one continuous curve with no stop at a pre-insertion point;
- speed: MAX_SPEED far away, ramping down to PUSH_SPEED within SLOW_ZONE of the surface;
- orientation: rotate towards the hole axis (yaw kept, the plug is round) at up to ROT_STEP per step;
- push: once centred and aligned near the surface, push along the axis at PUSH_SPEED until seated. If the force
  climbs past JAM_FORCE without the lip popping, back off JAM_LIFT and re-centre (a recorded retry).
Robustness noise (DART, Laskey et al. 2017) is an Ornstein-Uhlenbeck offset on the EXECUTED target -- smooth,
active only far from the hole -- while the CLEAN expert target is what gets recorded.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from peg_env import CONTROL_HZ, PLUG_LEN, PlugEnv, pose_to_action

DT = 1.0 / CONTROL_HZ
MAX_SPEED = 0.12
PUSH_SPEED = 0.01
SLOW_ZONE = 0.06
LATERAL_TAU = 0.6
CONE_SLOPE = 1.5
CENTRED = 0.0004
ALIGNED = np.deg2rad(1.5)
ROT_STEP = np.deg2rad(4.0)
SEAT_OVERSHOOT = 0.0005  # command this far past seated: ~10 N flange preload at the weld stiffness
JAM_FORCE = 22.0
JAM_LIFT = 0.006
HOLD_STEPS = 5
NOISE_POS, NOISE_ROT, NOISE_THETA = 0.002, np.deg2rad(1.5), 0.85
NOISE_FADE = (0.05, 0.10)  # full noise above 10 cm from the surface, none below 5 cm


def aligned_rotation(R: np.ndarray, n: np.ndarray) -> np.ndarray:
    x = R[:, 0] - (R[:, 0] @ n) * n
    if np.linalg.norm(x) < 1e-6:
        x = R[:, 1] - (R[:, 1] @ n) * n
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(n, x), n])


def rotate_toward(R: np.ndarray, R_goal: np.ndarray, max_angle: float) -> np.ndarray:
    rel = Rotation.from_matrix(R.T @ R_goal).as_rotvec()
    ang = np.linalg.norm(rel)
    if ang > max_angle:
        rel *= max_angle / ang
    return R @ Rotation.from_rotvec(rel).as_matrix()


class Expert:
    def __init__(self, env: PlugEnv, rng: np.random.Generator, noise: bool = True):
        self.env, self.rng, self.noise = env, rng, noise
        self.phase, self.hold, self.retries, self.backoff = "funnel", 0, 0, 0.0
        self.ou_p, self.ou_r = np.zeros(3), np.zeros(3)

    def act(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """(clean action to record, noisy action to execute, done)."""
        env = self.env
        k = env.spec.target_hole
        h, n = env.hole_pw[k], env.hole_nw[k]
        p, R = env.tcp()
        R_goal = aligned_rotation(R, n)
        s = (p - h) @ n  # height of the tip above the surface along the hole axis
        e = (p - h) - s * n  # lateral offset from the axis
        e_norm = np.linalg.norm(e)
        tilt = np.linalg.norm(Rotation.from_matrix(R.T @ R_goal).as_rotvec())
        speed = PUSH_SPEED + (MAX_SPEED - PUSH_SPEED) * np.clip(s / SLOW_ZONE, 0, 1)
        R_next = rotate_toward(R, R_goal, ROT_STEP)

        if self.phase == "funnel":
            lat = -e * min(1.0, DT / LATERAL_TAU)
            if np.linalg.norm(lat) > speed * DT:
                lat *= speed * DT / np.linalg.norm(lat)
            s_floor = max(CONE_SLOPE * max(e_norm - CENTRED, 0.0) + max(tilt - ALIGNED, 0.0) * 0.05 + self.backoff, 0.0)
            s_next = max(s - speed * DT, s_floor) if s > s_floor else min(s + speed * DT, s_floor)
            p_next = h + (e + lat) + s_next * n
            if s < 0.003 and e_norm < CENTRED and tilt < ALIGNED:
                self.phase, self.backoff = "push", 0.0
        if self.phase == "push":
            s_next = max(s - PUSH_SPEED * DT, -PLUG_LEN - SEAT_OVERSHOOT)
            p_next, R_next = h + s_next * n, R_goal
            axial = env.wrench[:3] @ n  # external force pushing the plug back out of the hole
            if not env.popped[k] and axial > JAM_FORCE:
                self.phase, self.backoff = "funnel", JAM_LIFT
                self.retries += 1
            elif env.inserted_hole() == k:
                self.phase = "hold"
        if self.phase == "hold":
            p_next, R_next = h + (-PLUG_LEN - SEAT_OVERSHOOT) * n, R_goal
            self.hold += 1

        clean = pose_to_action(p_next, R_next)
        executed = clean
        if self.noise:
            self.ou_p = NOISE_THETA * self.ou_p + self.rng.normal(0, NOISE_POS, 3)
            self.ou_r = NOISE_THETA * self.ou_r + self.rng.normal(0, NOISE_ROT, 3)
            g = np.clip((s - NOISE_FADE[0]) / (NOISE_FADE[1] - NOISE_FADE[0]), 0, 1)
            if self.phase == "funnel" and g > 0:
                executed = pose_to_action(p_next + g * self.ou_p, R_next @ Rotation.from_rotvec(g * self.ou_r).as_matrix())
        return clean, executed, self.hold >= HOLD_STEPS
