"""The scripted opening motion's geometry, in one place: touch point + cap pose -> waypoints.

Every script that plans or runs the opening (open_bottle_demo, open_bottle_demo_analyze_sensors,
open_bottle_agent, robot_imitation_glue/collect_data_bottle) calls plan_opening_motion; the shape itself
is defined by the constants in motion_constants.py. Pure geometry: no robot, no camera, so a plan can be
checked offline with `plan_opening_motion(np.eye(4), np.array([0.04, 0, 0]))`.

Cap-plane frame used below, at the touch point: `outward` points from the cap centre through the touch
point, `tangent` = cap_normal x outward is "left" (counter-clockwise about the cap's outward normal).
"""

import numpy as np
from calibrate_and_hover_bottle import orthonormalize_rotation
from motion_constants import (
    APPROACH_OFFSET,
    GRIPPER_YAW_DEG,
    LEG_DEPTH_STEP_M,
    LEG_YAW_MODE,
    LEGS,
    PUSH_ANGLE_DEG,
    PUSH_LENGTH_M,
    RADIAL_OUTWARD_OFFSET,
    TANGENTIAL_OFFSET,
)


def rotation_about_axis(axis, angle_rad):
    """Rodrigues rotation matrix about a unit axis."""
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def compute_yawed_gripper_orientation(cap_normal, reference_direction, yaw_deg):
    """Gripper orientation facing the cap (TCP z-axis = -cap normal) with a prescribed yaw
    about that z-axis: at yaw 0 the gripper's x-axis points along `reference_direction`
    (touch point -> bottle center); positive yaw rotates it counter-clockwise about the
    cap's outward normal."""
    z_axis = -cap_normal
    x_axis = rotation_about_axis(cap_normal, np.radians(yaw_deg)) @ reference_direction
    x_axis = x_axis - z_axis * z_axis.dot(x_axis)  # keep exactly perpendicular to z
    x_axis *= -1
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return orthonormalize_rotation(np.column_stack([x_axis, y_axis, z_axis]))


def make_pose(rotation, position):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


def plan_opening_motion(cap_pose, touch_point):
    """Grip point, push and legs for one cap pose and one (real or hypothetical) touch point.

    1. grip point: the touch point moved RADIAL_OUTWARD_OFFSET outward and TANGENTIAL_OFFSET along the
       tangent; the gripper descends there to APPROACH_OFFSET along the cap normal;
    2. push: a straight line of PUSH_LENGTH_M from the grip point, in the tangent direction turned
       PUSH_ANGLE_DEG counter-clockwise about the cap normal (i.e. toward the cap centre);
    3. legs: each LEGS entry (angle_deg, length_m) turns the previous direction by angle_deg
       counter-clockwise and travels length_m, ending LEG_DEPTH_STEP_M deeper than the previous leg.
       The gripper turns WITH the path: before each leg it rotates in place about the cap normal --
       by that leg's angle (LEG_YAW_MODE "direction"), keeping its yaw to the direction of travel, or
       by the angle the path has swept around the cap centre since the push end ("radius"), keeping
       its yaw to the local radius.

    Returns a dict. The *_end / grip_point entries lie in the cap plane (for drawing); approach_position,
    push_target and leg_targets are where the TCP actually goes. 'waypoint_poses' holds those as 4x4
    poses in visiting order (approach, push target, leg targets), each leg pose already carrying its
    leg's rotation; 'leg_turn_poses'[i] is the in-place turn to do BEFORE leg i -- same position as the
    previous target, leg i's rotation -- meant to run at YAW_SPEED so the legs keep MOVE_SPEED.
    """
    cap_center = cap_pose[:3, 3]
    cap_normal = cap_pose[:3, 2]

    outward = touch_point - cap_center
    outward = outward - cap_normal * cap_normal.dot(outward)
    outward /= np.linalg.norm(outward)
    tangent = np.cross(cap_normal, outward)
    tangent /= np.linalg.norm(tangent)

    grip_point = touch_point + RADIAL_OUTWARD_OFFSET * outward + TANGENTIAL_OFFSET * tangent
    push_direction = rotation_about_axis(cap_normal, np.radians(PUSH_ANGLE_DEG)) @ tangent
    push_end = grip_point + PUSH_LENGTH_M * push_direction
    approach_position = grip_point + APPROACH_OFFSET * cap_normal
    push_target = push_end + APPROACH_OFFSET * cap_normal

    leg_ends, leg_targets = [], []
    leg_direction, leg_end, leg_target = push_direction, push_end, push_target
    for angle_deg, length in LEGS:
        leg_direction = rotation_about_axis(cap_normal, np.radians(angle_deg)) @ leg_direction
        leg_direction -= cap_normal * cap_normal.dot(leg_direction)  # keep exactly in-plane
        leg_direction /= np.linalg.norm(leg_direction)
        leg_end = leg_end + length * leg_direction
        leg_target = leg_target + length * leg_direction - LEG_DEPTH_STEP_M * cap_normal
        leg_ends.append(leg_end)
        leg_targets.append(leg_target)

    yawed_rotation = compute_yawed_gripper_orientation(cap_normal, -outward, GRIPPER_YAW_DEG)
    def around_center(point):  # angle of a point about the cap centre, counter-clockwise about the normal
        v = point - cap_center
        return np.arctan2(np.cross(outward, v) @ cap_normal, outward @ v)

    if LEG_YAW_MODE not in ("direction", "radius"):
        raise ValueError(f"LEG_YAW_MODE must be 'direction' or 'radius', not {LEG_YAW_MODE!r}")
    leg_rotations, leg_turn_angles, turned = [], [], 0.0
    for (angle_deg, _), end in zip(LEGS, leg_ends):
        if LEG_YAW_MODE == "direction":
            turned += np.radians(angle_deg)
        else:  # swept angle, accumulated leg by leg so a path going round more than 180 deg stays unwrapped
            previous = leg_ends[len(leg_rotations) - 1] if leg_rotations else push_end
            turned += np.angle(np.exp(1j * (around_center(end) - around_center(previous))))
        leg_rotations.append(rotation_about_axis(cap_normal, turned) @ yawed_rotation)
        leg_turn_angles.append(turned)
    starts = [push_target, *leg_targets[:-1]]
    return {
        "outward": outward,
        "tangent": tangent,
        "grip_point": grip_point,
        "push_direction": push_direction,
        "push_end": push_end,
        "push_target": push_target,
        "leg_ends": leg_ends,
        "leg_targets": leg_targets,
        "approach_position": approach_position,
        "yawed_rotation": yawed_rotation,
        "leg_rotations": leg_rotations,
        "leg_turn_angles": leg_turn_angles,
        "leg_turn_poses": [make_pose(r, p) for r, p in zip(leg_rotations, starts)],
        "waypoint_poses": [make_pose(yawed_rotation, approach_position), make_pose(yawed_rotation, push_target),
                           *[make_pose(r, p) for r, p in zip(leg_rotations, leg_targets)]],
    }


def turn_angle_of(plan, cap_normal, rotation):
    """How far `rotation` is turned about the cap normal from the push's yawed rotation, in radians, UNWRAPPED
    onto the plan's own turns: a gripper 240 deg into its legs reads 240, not -120. Rotations are only
    known modulo a full turn, so of the candidates 2*pi apart the one nearest any planned turn is taken."""
    base = plan["yawed_rotation"]
    relative = rotation[:3, :3] @ base.T
    raw = np.arctan2(np.cross(base[:, 0], relative @ base[:, 0]) @ cap_normal, base[:, 0] @ (relative @ base[:, 0]))
    planned = np.array([0.0, *plan["leg_turn_angles"]])
    candidates = raw + 2 * np.pi * np.arange(-2, 3)
    return float(min(candidates, key=lambda c: np.abs(planned - c).min()))


def in_place_turns(plan, cap_normal, from_angle, to_angle, max_step_deg=120.0):
    """Rotations for turning in place from one turn angle to another (as returned by turn_angle_of),
    in steps small enough that each Cartesian or servo move takes the intended way round.

    A pose-to-pose move always takes the SHORTEST rotation, so a single move from 240 deg into the legs
    back to the push's yaw would carry on turning instead of unwinding -- the wrist-winding seen on the rig
    on 2026-10-09. Each returned rotation is at most max_step_deg from the previous one.
    """
    steps = int(np.ceil(abs(to_angle - from_angle) / np.radians(max_step_deg)))
    return [rotation_about_axis(cap_normal, a) @ plan["yawed_rotation"]
            for a in np.linspace(from_angle, to_angle, steps + 1)[1:]]


def retreat_rotations(plan, cap_normal, current_rotation, hover_rotation, max_step_deg=120.0):
    """Rotations that take the gripper from where the motion left it back to the hover orientation by
    undoing its NET turn since hover, in one direction, in steps of at most max_step_deg.

    The gripper turned twice on the way in: hover -> push yaw (the shortest way, so anywhere in +-180 deg,
    depending on where the touch point sits on the cap) and then the leg turns (plan's turn angles). Undoing
    those one after the other reverses direction whenever the two had opposite signs -- seen on the rig
    2026-10-09 as "turns the wrong way, then back". Their signed sum is the wrist's net rotation since hover;
    stepping that back to zero retraces it monotonically, so it neither winds nor reverses.
    """
    yawed = plan["yawed_rotation"]
    relative = yawed @ hover_rotation.T  # a rotation about the cap normal (both face the cap)
    x = hover_rotation[:, 0]
    initial_yaw = np.arctan2(np.cross(x, relative @ x) @ cap_normal, x @ (relative @ x))
    net = initial_yaw + turn_angle_of(plan, cap_normal, current_rotation)
    steps = max(1, int(np.ceil(abs(net) / np.radians(max_step_deg))))
    return [rotation_about_axis(cap_normal, net * (1 - k / steps)) @ hover_rotation for k in range(1, steps + 1)]

