"""One source of truth for the scripted bottle-opening motion.

Four modules run or plan this motion and must never disagree about its shape:

    open_bottle_demo.py                       -- tunes it interactively (the bench knob panel)
    open_bottle_demo_analyze_sensors.py       -- records sensor traces of it (threshold derivation)
    open_bottle_agent.py                      -- plans it (waypoints for the collector)
    robot_imitation_glue/collect_data_bottle.py -- collects the demonstrations with it

The numbers lived in two files once and it cost the calibration campaign: runs 0000-0011 were
recorded at MOVE_SPEED = 0.05 while the demo dragged at 0.01, so every threshold derived from
them described a five-times-faster motion than the one actually collected -- enough on its own
to turn a gradual yield into a snap. Anything two of these modules share is defined HERE, once;
consumers import it and the "keep equal to X" comments retire.

Deliberately NOT here: the sensor thresholds and checkpoint map (canonical home:
robot_imitation_glue/hardware/bottle_sensor.py -- import it, never copy it) and rig state --
robot IPs, home/neutral joint configurations, transit speeds (calibrate_and_hover_bottle.py /
open_bottle_agent.py). This module stays import-free so every consumer can afford it.
"""

# --- waypoints in the cap's frame ----------------------------------------------------------
APPROACH_OFFSET = -0.022  # metres above the grip point where the gripper stops (negative = press below the cap plane)
RADIAL_OUTWARD_OFFSET = 0.01  # metres from the touch point radially outward (away from the cap center)
# metres to the "left" of the touch point: along the cap-plane tangent at the touch point, in the
# counter-clockwise direction about the cap's outward normal (the direction the old circular
# motion went). Negate if "left" turns out to be the other way on the real bottle.
TANGENTIAL_OFFSET = -0.04
# metres to the "left" of the cap center (same tangential axis as TANGENTIAL_OFFSET):
# the push line ends here instead of at the center itself. Together with PUSH_OVERSHOOT the two
# 30cm-scale constants only DEFINE A DIRECTION -- the line actually walked is short.
PUSH_TARGET_TANGENTIAL_OFFSET = 0.30
# metres to keep pushing past the push end point along the same line (0.0 = stop exactly there;
# negative = stop short of it). Length matters more than it looks: at -0.27 the push was a 7.2cm
# line, a ~102 deg sweep about the cap centre -- over a quarter turn of the unscrewing before leg
# 1 even ran (run_0011: S0 at 100% of its excursion 0.7s before push_end). -0.29 slides the push
# back along its OWN line, so the push direction -- and therefore every leg angle -- keeps its
# meaning.
PUSH_OVERSHOOT = -0.29
# legs of the opening motion, executed in order after the push, each as (angle_deg, offset_m).
# The angle is measured counter-clockwise about the cap's outward normal RELATIVE TO THE PREVIOUS
# leg's direction (the first entry is relative to the push direction; 90 = exactly
# perpendicular/"to the left"); the offset is the distance travelled. Each leg additionally
# descends LEG_DEPTH_STEP_M further below the cap plane (cumulative), so the path is a shallow
# helix rather than a flat polyline. Add/remove/tune entries freely -- entry i is the leg gated as
# leg_{i + 2}_end; the planned path preview, prints, and execution all follow this list.
LEGS = [
    (45.0, 0.03),  # leg 2
    (20.0, 0.06),  # leg 3
    (30.0, 0.04),  # leg 4
    (30.0, 0.03),  # leg 5
    (30.0, 0.01),  # leg 6
]
# metres to press deeper along -cap_normal with each successive leg (cumulative over LEGS), so
# the fingers keep biting into the rim as the cap turns instead of riding up out of the
# serrations. 0.0 = every leg at the push height, i.e. the old flat cap-plane polyline. The TCP
# ends the sequence APPROACH_OFFSET + LEG_DEPTH_STEP_M * len(LEGS) below the cap plane -- with
# the current values -2.2cm + 1.5cm = -3.7cm, which is below the cap's own height, so watch the
# gripper against the bottle shoulder on the later legs.
LEG_DEPTH_STEP_M = 0.003
# yaw of the gripper about its own TCP z-axis (which faces the cap), applied by rotating in place
# BEFORE descending to the grip point. Defined relative to the touch-point -> bottle-center axis:
# at 0 the gripper's x-axis points from the touch point toward the center; positive rotates
# counter-clockwise about the cap's outward normal. 90 puts the finger-pair line tangential, so
# the leading finger tip rakes the serrated rim and levers it (yaw 0 would pinch the rim and drag
# by friction).
GRIPPER_YAW_DEG = 90

# --- speeds --------------------------------------------------------------------------------
MOVE_SPEED = 0.05  # m/s for the straight moves (approach, push, legs)
# Speed of the in-place yaw ONLY, which is a PURE rotation: moveL charges rotation at one
# "metre" per radian of the rotation vector, so a 90 deg yaw is a 1.57m path -- 31.4s even at
# today's MOVE_SPEED of 0.05, and 157s at 0.01. Both exceed airo's 30s AwaitableAction timeout,
# and that timeout only WARNS: the async moveL keeps running, so the very next move (the descend)
# calls servoStop() + isPoseWithinSafetyLimits() mid-trajectory, is answered with "RTDE control
# script is not running!", and aborts with a bogus "pose is not reachable". Yawing at 0.2 takes
# ~8s and stays inside 30s up to a ~340 deg yaw -- so the yaw needs its own speed no matter what
# MOVE_SPEED is.
YAW_SPEED = 0.2

# --- recovery and stop-on-open (the collector and the calibration recorder) -----------------
DEPTH_NUDGE_M = 0.003  # metres to press deeper per retry attempt, along -cap_normal
MAX_MOTION_RETRIES = 3  # how many times to retract and redo the whole push+legs motion
RETRACT_LIFT_METERS = 0.05  # metres to lift off the cap along its normal before retrying
# Skip the remaining legs once EVERY channel has read at or above its own threshold continuously
# for OPEN_DWELL_S. Measured on the calibration batch: all three channels reach their plateau
# during leg 3, so the later legs were always post-open travel (~2 s over ~8 cm, getting deeper
# each leg). The dwell is what separates a real pop from run_0017's lift-and-recover (S2 rose for
# ~1.2 s there and then fell back as the later legs pressed the loose cap down again); a single
# reading would call that success. The episode still ends with the RETRACT_LIFT_METERS disengage
# lift the retry path uses, so the gripper never drags across an open cap and the terminal state
# stays uniform.
STOP_LEGS_WHEN_OPEN = True
OPEN_DWELL_S = 0.5
OPEN_POLL_S = 0.05
