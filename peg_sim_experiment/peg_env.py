"""MuJoCo plug-in-hole on a faceted car-body sheet: does a goal-conditioned policy insert into a hole it never trained on?

Sheet. Five flat panels bent along creases parallel to y, like a hood running into a fender: profile angles
+15 / 0 / -10 / -25 / -40 deg, 40 cm wide, ~70 cm of profile. Ten identical round holes, >= 12 cm apart:
  train   6 holes on the +15 / 0 / -10 / -40 deg panels
  test_a  3 holes on panels that ALSO carry train holes  -> unseen location, seen approach angle
  test_b  1 hole on the -25 deg panel, which has no train hole -> unseen approach angle (between -10 and -40)
All holes are always present, so the image alone cannot say which one to take: the goal input must.

Plug. A rubber plug (body r = 8 mm, 14 mm long, flange r = 13 mm) in a 8.6 mm hole (0.6 mm clearance) in 3 mm
sheet. MuJoCo friction has no static/dynamic distinction, so the snap-through is an explicit force law on the
plug's lip (`_lip_force`): past LIP_START of insertion the lip resists like a spring up to LIP_PEAK; if the plug is
aligned at that moment (tilt < POP_TILT) it pops and the resistance drops to LIP_DYNAMIC until the flange seats.
Misaligned, it never pops and the force keeps climbing -- a jam. So the force trace shows "peak, then drop" only
for a good insertion, as on the real plugs.

Arm. The plug hangs from a mocap target through a soft weld (a compliant controller), with gravity compensation
on the plug body (as an arm controller holds its payload). Tool frame: origin at the plug TIP, +z up the plug, so
insertion is along -z and the nominal pose is close to identity (no Euler wrap in observation.state). The wrist
camera is placed like the bottle rig's (see build_xml).

observation.state (18) = [tcp xyz, tcp extrinsic-xyz Euler, F_tool(3), T_tool(3), goal(6)]: the first 6 are what
the lerobot fork's SE3RelativeActionsProcessorStep reads as the reference pose; F/T is the external wrench on the
plug in the tool frame (contacts + lip), averaged over the control period like a filtered wrist sensor; goal =
target-hole centre and outward normal in the CURRENT tool frame. Action = absolute tool target [xyz, rot6d].
"""

from __future__ import annotations

import dataclasses
import os

os.environ["MUJOCO_GL"] = "egl"  # entry scripts must also set this before importing mujoco (GLFW = CPU rendering)

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# --- sheet ------------------------------------------------------------------------------------------------
PANEL_ANGLES_DEG = [15.0, 0.0, -10.0, -25.0, -40.0]
PANEL_LENGTHS = [0.14, 0.18, 0.14, 0.12, 0.12]
SHEET_WIDTH = 0.40
SHEET_TH = 0.0015  # half thickness (3 mm sheet)
HOLE_SQUARE = 0.02  # half side of the square cut-out around each hole (frame + rim fill it)
# (panel, u along the profile, v across, split)
HOLES = [
    (0, 0.07, -0.10, "train"), (0, 0.07, 0.10, "test_a"),
    (1, 0.05, 0.00, "train"), (1, 0.13, -0.12, "train"), (1, 0.13, 0.12, "test_a"),
    (2, 0.07, -0.08, "test_a"), (2, 0.07, 0.10, "train"),
    (3, 0.06, 0.00, "test_b"),
    (4, 0.06, -0.10, "train"), (4, 0.06, 0.10, "train"),
]
SPLITS = {s: [k for k, h in enumerate(HOLES) if h[3] == s] for s in ("train", "test_a", "test_b")}
TRAIN_HOLES = SPLITS["train"]
SHEET_BASE = np.array([0.0, 0.0, 0.10])

# --- plug and hole ----------------------------------------------------------------------------------------
PLUG_R = 0.008
PLUG_LEN = 0.014  # tip to flange
FLANGE_R = 0.013
HOLE_R = 0.0086
WALL = 0.003
LIP_START = 0.004  # insertion depth where the lip meets the sheet
LIP_STIFFNESS = 4000.0  # N/m
LIP_PEAK = 15.0  # N, break-away force
LIP_DYNAMIC = 2.5  # N, sliding resistance after the pop
POP_TILT = np.deg2rad(4.0)
SEATED_DEPTH = 0.013  # flange (at 14 mm) within 1 mm of the surface

# --- workspace --------------------------------------------------------------------------------------------
SHEET_TRAVEL = 0.30  # +- in x and y
SHEET_YAW_DEG = 45.0
REACH_XY = 0.42
MAX_NORMAL_TILT = np.deg2rad(55.0)
HOME = np.array([0.0, 0.0, 0.42])

CONTROL_HZ = 10
SIM_DT = 0.002
SUBSTEPS = int(round(1 / CONTROL_HZ / SIM_DT))
IMAGE_H, IMAGE_W = 120, 160  # 4:3, half the bottle rig's 240x320 wrist stream
SHEET_RGBA = "0.62 0.64 0.68 1"
RIM_RGBA = "0.12 0.12 0.14 1"


def panel_frames() -> list[tuple[np.ndarray, np.ndarray]]:
    """(origin, R = [u, v, n]) of each panel in the sheet frame; the profile is centred in x with its top at z=0."""
    q, starts = np.zeros(3), []
    for ang, length in zip(PANEL_ANGLES_DEG, PANEL_LENGTHS):
        t = np.deg2rad(ang)
        u = np.array([np.cos(t), 0.0, np.sin(t)])
        starts.append((q.copy(), np.column_stack([u, [0.0, 1.0, 0.0], np.cross(u, [0.0, 1.0, 0.0])])))
        q = q + length * u
    pts = np.array([s[0] for s in starts] + [q])
    shift = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, 0.0, pts[:, 2].max()])
    return [(o - shift, R) for o, R in starts]


PANELS = panel_frames()


def hole_local(k: int) -> tuple[np.ndarray, np.ndarray]:
    panel, u, v, _ = HOLES[k]
    o, R = PANELS[panel]
    return o + R @ np.array([u, v, 0.0]), R


def _quat(R: np.ndarray) -> str:
    q = Rotation.from_matrix(R).as_quat()
    return f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}"


def _box(name, pos, R, size, rgba, collide=True) -> str:
    c = "" if collide else ' contype="0" conaffinity="0"'
    return (f'<geom name="{name}" type="box" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{_quat(R)}" '
            f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" rgba="{rgba}"{c}/>')


def _rect_minus_squares(u0, u1, v0, v1, squares) -> list[tuple[float, float, float, float]]:
    """Axis-aligned rectangles covering [u0,u1]x[v0,v1] minus the given squares (u_c, v_c, half)."""
    us = sorted({u0, u1, *[c - h for c, _, h in squares], *[c + h for c, _, h in squares]})
    us = [u for u in us if u0 <= u <= u1]
    out = []
    for a, b in zip(us[:-1], us[1:]):
        mid = (a + b) / 2
        cuts = sorted((vc - h, vc + h) for uc, vc, h in squares if uc - h <= mid <= uc + h)
        v = v0
        for lo, hi in cuts:
            if lo > v:
                out.append((a, b, v, lo))
            v = max(v, hi)
        if v < v1:
            out.append((a, b, v, v1))
    return out


def _sheet_geoms() -> str:
    geoms = []
    for p, ((o, R), length) in enumerate(zip(PANELS, PANEL_LENGTHS)):
        local = lambda uvw: o + R @ np.asarray(uvw)  # noqa: E731
        squares = [(u, v, HOLE_SQUARE) for (panel, u, v, _) in HOLES if panel == p]
        ext = 0.002  # overlap across the creases
        for r, (a, b, c, d) in enumerate(_rect_minus_squares(-ext, length + ext, -SHEET_WIDTH / 2, SHEET_WIDTH / 2, squares)):
            geoms.append(_box(f"p{p}_r{r}", local([(a + b) / 2, (c + d) / 2, -SHEET_TH]), R,
                              [(b - a) / 2, (d - c) / 2, SHEET_TH], SHEET_RGBA))
    for k in range(len(HOLES)):
        h, R = hole_local(k)
        local = lambda uvw: h + R @ np.asarray(uvw)  # noqa: E731
        ap = HOLE_R + WALL
        w = (HOLE_SQUARE - ap) / 2
        for s, (cx, cy, hx, hy) in enumerate([(-(ap + w), 0, w, HOLE_SQUARE), (ap + w, 0, w, HOLE_SQUARE),
                                              (0, ap + w, ap, w), (0, -(ap + w), ap, w)]):
            geoms.append(_box(f"h{k}_f{s}", local([cx, cy, -SHEET_TH]), R, [hx, hy, SHEET_TH], SHEET_RGBA))
        half_side = ap * np.tan(np.pi / 12) + 0.0005
        for j in range(12):
            th = 2 * np.pi * j / 12
            Rj = R @ Rotation.from_euler("z", th).as_matrix()
            rr = HOLE_R + WALL / 2
            geoms.append(_box(f"h{k}_rim{j}", local([rr * np.cos(th), rr * np.sin(th), -SHEET_TH]), Rj,
                              [WALL / 2, half_side, SHEET_TH], RIM_RGBA))
    return "\n      ".join(geoms)


def _camera_xyaxes(pos, target, up) -> str:
    f = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return " ".join(f"{v:.6f}" for v in (*x, *y))


def build_xml() -> str:
    # Like the bottle rig's wrist camera: behind the palm, offset 6 cm, looking forward along the tool axis
    # (tilted ~15 deg towards it) with the fingers and the gripped plug at the bottom of the frame. 58 deg vertical
    # FOV ~ the D405. Image "down" points from the camera towards the axis (+y_tool).
    cam_pos = np.array([0.0, -0.06, 0.075])
    wrist_xy = _camera_xyaxes(cam_pos, np.array([0.0, 0.0, -0.12]), np.array([0.0, -1.0, 0.0]))
    scene_pos = np.array([0.75, -0.85, 0.95])
    scene_xy = _camera_xyaxes(scene_pos, np.array([0.0, 0.0, 0.15]), np.array([0.0, 0.0, 1.0]))
    return f"""
<mujoco model="plug_sheet">
  <option timestep="{SIM_DT}" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag multiccd="enable"/>
  </option>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .82" rgb2=".72 .72 .70" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="16 16"/>
  </asset>
  <default><geom friction="0.6 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <light name="key" pos="0.4 -0.5 1.4" dir="-0.3 0.4 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light name="fill" pos="-0.5 0.4 1.2" dir="0.4 -0.3 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.01" material="grid" contype="0" conaffinity="0"/>
    <camera name="scene" pos="{scene_pos[0]} {scene_pos[1]} {scene_pos[2]}" xyaxes="{scene_xy}" fovy="45"/>
    <body name="sheet" mocap="true" pos="{SHEET_BASE[0]} {SHEET_BASE[1]} {SHEET_BASE[2]}">
      {_sheet_geoms()}
      <geom name="stand" type="box" pos="0 0 -0.08" size="0.06 0.06 0.05" rgba="0.3 0.3 0.32 1" contype="0" conaffinity="0"/>
    </body>
    <body name="target" mocap="true" pos="{HOME[0]} {HOME[1]} {HOME[2]}"/>
    <body name="marker" mocap="true" pos="0 0 -1">
      <geom name="marker" type="cylinder" size="0.018 0.0004" rgba="0.1 0.8 0.2 0.9" contype="0" conaffinity="0" group="3"/>
    </body>
    <body name="plug" pos="{HOME[0]} {HOME[1]} {HOME[2]}" gravcomp="1">
      <freejoint/>
      <geom name="plug_body" type="cylinder" pos="0 0 {PLUG_LEN / 2}" size="{PLUG_R} {PLUG_LEN / 2}" rgba="0.08 0.08 0.08 1" mass="0.02"/>
      <geom name="plug_flange" type="cylinder" pos="0 0 {PLUG_LEN + 0.0015}" size="{FLANGE_R} 0.0015" rgba="0.1 0.1 0.1 1" mass="0.01"/>
      <geom name="finger_l" type="box" pos="-0.0105 0 {PLUG_LEN + 0.017}" size="0.0025 0.006 0.014" rgba="0.75 0.75 0.78 1" mass="0.02" contype="0" conaffinity="0"/>
      <geom name="finger_r" type="box" pos="0.0105 0 {PLUG_LEN + 0.017}" size="0.0025 0.006 0.014" rgba="0.75 0.75 0.78 1" mass="0.02" contype="0" conaffinity="0"/>
      <geom name="palm" type="box" pos="0 0 {PLUG_LEN + 0.039}" size="0.025 0.015 0.008" rgba="0.2 0.2 0.25 1" mass="0.3" contype="0" conaffinity="0"/>
      <camera name="wrist" pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}" xyaxes="{wrist_xy}" fovy="58"/>
    </body>
  </worldbody>
  <equality><weld body1="plug" body2="target" relpose="0 0 0 1 0 0 0" solref="0.02 1"/></equality>
</mujoco>"""


def pose_to_action(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.concatenate([p, R[:, 0], R[:, 1]]).astype(np.float32)


def action_to_pose(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c1 = a[3:6] / np.linalg.norm(a[3:6])
    c2 = a[6:9] - (c1 @ a[6:9]) * c1
    c2 /= np.linalg.norm(c2)
    return np.asarray(a[:3], dtype=float), np.column_stack([c1, c2, np.cross(c1, c2)])


@dataclasses.dataclass
class EpisodeSpec:
    seed: int
    target_hole: int
    sheet_pos: np.ndarray
    sheet_R: np.ndarray
    start_pos: np.ndarray
    start_R: np.ndarray


def reachable(sheet_pos: np.ndarray, sheet_R: np.ndarray, k: int) -> bool:
    p, R = hole_local(k)
    pw, nw = sheet_pos + sheet_R @ p, sheet_R @ R[:, 2]
    return bool(np.all(np.abs(pw[:2]) <= REACH_XY) and np.arccos(np.clip(nw[2], -1, 1)) <= MAX_NORMAL_TILT)


def sample_episode(seed: int, holes: list[int]) -> EpisodeSpec:
    """Sheet: +-30 cm in x/y, +-2 cm in z, yaw +-45 deg, tilt +-5 deg; target = a reachable hole of `holes`.
    Start: near the home pose 42 cm up (+-3 cm, yaw +-30 deg, tilt +-8 deg)."""
    rng = np.random.default_rng(seed)
    while True:
        sheet_pos = SHEET_BASE + np.array([*rng.uniform(-SHEET_TRAVEL, SHEET_TRAVEL, 2), rng.uniform(-0.02, 0.02)])
        sheet_R = Rotation.from_euler("zxy", [rng.uniform(-SHEET_YAW_DEG, SHEET_YAW_DEG), *rng.uniform(-5, 5, 2)],
                                      degrees=True).as_matrix()
        ok = [k for k in holes if reachable(sheet_pos, sheet_R, k)]
        if ok:
            break
    start_pos = HOME + np.array([*rng.uniform(-0.03, 0.03, 2), rng.uniform(-0.02, 0.02)])
    start_R = Rotation.from_euler("zxy", [rng.uniform(-30, 30), *rng.uniform(-8, 8, 2)], degrees=True).as_matrix()
    return EpisodeSpec(seed, int(rng.choice(ok)), sheet_pos, sheet_R, start_pos, start_R)


class PlugEnv:
    def __init__(self, render: bool = True):
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMAGE_H, IMAGE_W) if render else None
        self.plug_body = self.model.body("plug").id
        self.target_mocap = self.model.body("target").mocapid[0]
        self.sheet_mocap = self.model.body("sheet").mocapid[0]
        self.plug_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.plug_body]]
        local = [hole_local(k) for k in range(len(HOLES))]
        self._hole_p = np.array([p for p, _ in local])
        self._hole_n = np.array([R[:, 2] for _, R in local])
        self.spec: EpisodeSpec | None = None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def holes_world(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.spec
        return s.sheet_pos + self._hole_p @ s.sheet_R.T, self._hole_n @ s.sheet_R.T

    def reset(self, spec: EpisodeSpec, goal_hole: int | None = None) -> dict:
        """goal_hole overrides the hole fed to the policy (success is always judged against spec.target_hole)."""
        self.spec = spec
        self.goal_hole = spec.target_hole if goal_hole is None else goal_hole
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.sheet_mocap] = spec.sheet_pos
        self.data.mocap_quat[self.sheet_mocap] = np.roll(Rotation.from_matrix(spec.sheet_R).as_quat(), 1)
        q = np.roll(Rotation.from_matrix(spec.start_R).as_quat(), 1)
        self.data.qpos[self.plug_qadr:self.plug_qadr + 3] = spec.start_pos
        self.data.qpos[self.plug_qadr + 3:self.plug_qadr + 7] = q
        self.data.mocap_pos[self.target_mocap] = spec.start_pos
        self.data.mocap_quat[self.target_mocap] = q
        self.hole_pw, self.hole_nw = self.holes_world()
        self.popped = np.zeros(len(HOLES), dtype=bool)
        self.lip_axial = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SUBSTEPS):
            self._substep()
        self.wrench = np.zeros(6)
        return self.observe()

    def tcp(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.plug_body].copy(), self.data.xmat[self.plug_body].reshape(3, 3).copy()

    def hole_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per hole: insertion depth of the tip (along -normal), radial offset of the tip, plug tilt to the hole axis."""
        p, R = self.tcp()
        d = p - self.hole_pw
        depth = -(d * self.hole_nw).sum(1)
        radial = np.linalg.norm(d + depth[:, None] * self.hole_nw, axis=1)
        tilt = np.arccos(np.clip(self.hole_nw @ R[:, 2], -1, 1))
        return depth, radial, tilt

    def _lip_force(self) -> tuple[np.ndarray, float]:
        """World force on the plug from the rubber lip (see the module docstring), and its axial magnitude."""
        depth, radial, tilt = self.hole_coords()
        k = int(np.argmin(radial))
        if radial[k] > HOLE_R or depth[k] <= LIP_START:
            return np.zeros(3), 0.0
        if self.popped[k]:
            f = LIP_DYNAMIC
        else:
            f = LIP_STIFFNESS * (depth[k] - LIP_START)
            if f >= LIP_PEAK and tilt[k] < POP_TILT:
                self.popped[k] = True
                f = LIP_DYNAMIC
        return f * self.hole_nw[k], f

    def _substep(self) -> np.ndarray:
        self.data.qfrc_applied[:] = 0
        f_lip, self.lip_axial = self._lip_force()
        p, _ = self.tcp()
        if self.lip_axial > 0:
            mujoco.mj_applyFT(self.model, self.data, f_lip, np.zeros(3), p, self.plug_body, self.data.qfrc_applied)
        mujoco.mj_step(self.model, self.data)
        return self._external_wrench(f_lip)

    def _external_wrench(self, f_lip: np.ndarray) -> np.ndarray:
        """External force and torque (about the tip) on the plug, world frame: contacts + lip."""
        p, _ = self.tcp()
        force, torque, buf = f_lip.copy(), np.zeros(3), np.zeros(6)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1, b2 = self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]
            if self.plug_body not in (b1, b2):
                continue
            mujoco.mj_contactForce(self.model, self.data, c, buf)
            f = con.frame.reshape(3, 3).T @ buf[:3]  # world force of geom1 on geom2
            if b1 == self.plug_body:
                f = -f
            force += f
            torque += np.cross(con.pos - p, f)
        return np.concatenate([force, torque])

    def goal_in_tool(self, hole: int | None = None) -> np.ndarray:
        k = self.goal_hole if hole is None else hole
        p, R = self.tcp()
        return np.concatenate([R.T @ (self.hole_pw[k] - p), R.T @ self.hole_nw[k]]).astype(np.float32)

    def observe(self) -> dict:
        p, R = self.tcp()
        ft = np.concatenate([R.T @ self.wrench[:3], R.T @ self.wrench[3:]])
        state = np.concatenate([p, Rotation.from_matrix(R).as_euler("xyz"), ft, self.goal_in_tool()]).astype(np.float32)
        obs = {"state": state}
        if self.renderer is not None:
            self.renderer.update_scene(self.data, "wrist")
            obs["image"] = self.renderer.render().copy()
        return obs

    def step(self, action: np.ndarray) -> dict:
        """Move the mocap target to the absolute tool pose `action` [xyz, rot6d], interpolated over one control period."""
        p1, R1 = action_to_pose(action)
        p0 = self.data.mocap_pos[self.target_mocap].copy()
        r0 = Rotation.from_quat(np.roll(self.data.mocap_quat[self.target_mocap], -1))
        rel = (r0.inv() * Rotation.from_matrix(R1)).as_rotvec()
        wrench = np.zeros(6)
        for s in range(1, SUBSTEPS + 1):
            u = s / SUBSTEPS
            self.data.mocap_pos[self.target_mocap] = p0 + u * (p1 - p0)
            self.data.mocap_quat[self.target_mocap] = np.roll((r0 * Rotation.from_rotvec(u * rel)).as_quat(), 1)
            wrench += self._substep()
        self.wrench = wrench / SUBSTEPS
        return self.observe()

    def inserted_hole(self) -> int | None:
        """The hole the plug is seated in: popped, flange within 1 mm of the surface, tip inside the hole."""
        depth, radial, _ = self.hole_coords()
        ok = np.nonzero(self.popped & (depth >= SEATED_DEPTH) & (radial <= HOLE_R))[0]
        return int(ok[0]) if len(ok) else None

    def force_magnitude(self) -> float:
        return float(np.linalg.norm(self.wrench[:3]))
