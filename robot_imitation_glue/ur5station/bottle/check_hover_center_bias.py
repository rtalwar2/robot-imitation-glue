"""Measure whether the collector's hover pose actually centres the cap on the gripper axis.

The scripted opening is built on a *predicted* cap pose: `DEFAULT_TCP_RIGHT_TO_BOTTLE` turns
ur_right's TCP into a cap frame, and `compute_hover_pose_above_bottle` parks the gripper axis on that
prediction 10 cm above it. The symptom that started this is that in Rerun the logged cap frame sits
off the middle of the cap in the wrist image while its plane (height and tilt) looks right -- which
points at that transform's TRANSLATION, not at the hover geometry, and `sometimes` in the
description points at the bottle settling differently in the gripper rather than at a constant
error. This script separates those two, from a sticker on the cap plus the cap's own rim, at the
same seeded poses the collector draws (imported from splits.py, never copied) and with the same
release-transit-regrasp sequence.

Off-centre in the image is not the bias
---------------------------------------
The wrist RealSense is not on ur_left's tool axis: `tcp_left_to_camera` puts it 69 mm off the axis,
139 mm back along it, tilted 12 deg from the approach direction. At hover the cap centre is therefore
expected ~140 px from the principal point (the script prints the prediction every pose, and the
prediction is the thing to compare against), so a frame with the lid to one side proves nothing. What
is worth measuring is a disagreement, and rolling the wrist is what separates the two candidates:
  * the projection chain is wrong -- a bad lateral term or principal point in the camera calibration:
    an offset fixed in the IMAGE frame;
  * the cap really is not where the model puts it -- an offset fixed in the WORLD.
Rolling the wrist about the tool z axis at a fixed TCP leaves the gripper axis exactly where it was
(it IS the rotation axis) and rotates the world around it, so a landmark's normalised image offset is

    n(theta) = b + Rz(-theta) q                      b, q constant

which is exact rather than merely first order -- a rotation about z leaves the landmark's view-axis
depth invariant, so n = v_perp / v_z rotates rigidly and the locus is a true circle. Checked against
the real mount: on noise-free data through the actual 12 deg tilt and fx=931 this model leaves
0.35 px of residual at every mount tilt from 0 to 18 deg, so a residual of a few px means the
LANDMARK is sliding, not that the maths is wrong, and the fit may be trusted to that scale. A
least-squares fit of b and q over the sweep splits one measurement into two numbers by the encoders
alone:

  b - b_pred        where the gripper axis crosses the cap plane appears in the image (b) against
                    where the calibration says it must (b_pred). The projection chain's lateral
                    error. Not attributable further: a wrong lateral `tcp_left_to_camera`, a wrong
                    principal point and a wrong camera-to-tool tilt all look identical under a roll,
                    so fix whichever you trust least.
  |q| * view_depth  how far the gripper axis misses the landmark, in mm -- the centring error the
                    motion inherits. Reported in the tool, base and cap frames; the cap-frame vector
                    is the one that must repeat across poses if the grasp transform's translation is
                    what is biased, and must stay put across REPEATS of one grasp if the error is
                    constant rather than the bottle shifting in the fingers.

Only its rotation is borrowed from the calibration to express directions (a checkerboard pins an
angle long before it pins a 2 mm lateral offset); its translation is never used, which is what makes
the miss an independent read on the world.

Two landmarks, because one is not enough
----------------------------------------
  * the STICKER: whatever the operator stuck on the cap. Good, but its colour must be separable from
    the cap's red rim and the bench, and a mis-stuck sticker is indistinguishable from a model bias.
  * the cap's own red RIM ARC, circle-fitted with the repo's `_fit_rim_circle` -- the geometric cap
    centre, needing no sticker at all, so it answers the Rerun complaint directly.

Their separation is a third number: if the sticker sits 5 mm from the rim's fitted centre, the
sticker is not on the cap centre and every sticker-derived number carries that. The rim arc is
measured against the *model's* centre, so the model-vs-rim miss is the one to quote for
`DEFAULT_TCP_RIGHT_TO_BOTTLE`.

A cv2.HoughCircles pass over the greyscale disc edge was tried and is NOT used: it is stable in
radius (171 +- 4 px) but its centre sits a persistent ~27 px from the cap's features, because the
visible silhouette is the lid plus its skirt and is partly occluded by the finger. A circle fit to
the rim arc, which is a true circle in the cap plane, does not have that problem.

Why every landmark is searched in an anchored window
---------------------------------------------------
`touch_point_detector._red_components` crops a central ROI on the assumption that "the hover routine
centers the lid". On these frames the lid is 140 px off-centre by mount geometry, so that crop
threw away 40% of the rim pixels and let the wooden bench in -- the un-anchored fit put its circle on
the background, centre (303, 405), residual 27 px. Every search here is instead confined to a window
around where the MODEL says the cap is, which is a prediction of the thing being measured and gets
the geometry right without assuming the answer.

Why the lock is rebuilt per pose, on illumination-invariant evidence
--------------------------------------------------------------------
Absolute colour does not survive a new right-arm pose: re-standing the bottle changes the light on the
sticker, and on the first run the window accepted at pose 1 (h 8, s 208, v 166) still matched the
sticker there but found only 76-291 px fragments scattered over the frame at pose 2 -- which is how a
41 mm "miss" and a 17 mm "bias" got reported for a bottle that was simply re-lighted. So the lock is
rebuilt at every pose from chromaticity (a ratio, so brightness barely enters) thresholded relative to
the search disc's own peak, plus the disc-shape gates: together they recovered exactly one candidate at
10 of the 10 frames of those two poses, including the pose-2 ones the colour window lost entirely.
Where two or more survive the script asks instead of guessing. A click is needed at most once per run
because the search disc is a fixed pixel (see `predicted_cap_pixel`).

What the fit residual means, measured
-------------------------------------
Noise-free data through this mount leaves 0.35 px, so the estimator is not the limit; on the real pose 1
sweep both detectors leave ~5 px and agree with each other to 0.4-3.9 px per angle, 0.2 mm on the bias
and 0.2 mm on the miss. The residual is radial and a function of the distance from the principal point
(-8 px at r=136, +7 px at r=150, same sign at both ends of the sweep) -- the sticker's own appearance
changing as the camera orbits it by +-16 deg, plus uncorrected radial distortion at 140 px off-axis.
It is therefore a floor on this method, worth roughly +-1 mm on the mm numbers, and MAX_FIT_RESIDUAL_PX
is set to catch a landmark that JUMPS rather than one that glints.

Limits worth reading before quoting a number
--------------------------------------------
  * Nothing along the view axis is observable: the 3D cross-checks intersect each landmark's ray with
    the *model* cap plane, so that plane's position is assumed, not measured. The mm scale of |q|
    borrows the same depth (~3% scale error, verified against noise-free data).
  * The rim arc's fitted centre is the projection of the rim circle's centre; at the cap's 12-30 deg
    view angle the rim is a slight ellipse, worth ~1 px here, and a short arc is unstable -- the script
    prints the arc's angular span and refuses below `MIN_RIM_ARC_DEG`. On the real frames the arc's
    fitted radius collapsed to 134 px against the 164 px prior at +90 deg (the gripper finger covering
    the near rim), which is why radius agreement is also a gate.
  * A mis-stuck sticker is indistinguishable from a model bias, both being a constant vector in the cap
    frame. That is what the sticker-vs-rim separation is for: the rim centre is the cap's own geometry,
    so a sticker 5 mm from it is a 5 mm error in every sticker-derived number.
  * Per-pose SCATTER mixes two things this script cannot separate in one pass: the systematic
    transform error and the hand re-grasp. Use `--grasps 3` on ONE pose to split them -- repeats of
    the same pose differ only by how the bottle settled, so the within-pose spread IS the re-grasp
    noise and the mean across repeats IS the bias.
  * Measurement only: ur_left never leaves the computed hover pose apart from the wrist roll, so what
    this records is what the collector would have recorded. Corrections are printed, never applied.

Usage (imitation_button_py312, from the repo root; rig up -- both UR5s, the right arm's gripper, the
wrist camera):

    python -m robot_imitation_glue.ur5station.bottle.check_hover_center_bias
    python -m robot_imitation_glue.ur5station.bottle.check_hover_center_bias --pose-index 0 --grasps 3

Keys: Enter advances every gate (Ctrl+C aborts, and the JSON still gets written). A live window stays up
for the whole run, drawing the search disc, the model's prediction, the gripper axis's predicted pixel
and each landmark, with the numbers that produced them -- `r` re-locks the sticker, `q` aborts. The
automatic lock asks for a click only when it cannot decide; then click the sticker, tune with [ ] until
one blob survives the gates, Enter to accept.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from airo_camera_toolkit.utils.image_converter import ImageConverter

from robot_imitation_glue.collect_data_bottle import (
    GRIPPER_GRASP_WIDTH_M,
    GRIPPER_OPEN_WIDTH_M,
    _move_gripper_to,
)
from robot_imitation_glue.hardware.gripper_startup import GRIPPER_PORT_RIGHT, make_gripper
from robot_imitation_glue.ur5station.bottle.splits import (
    POSE_BLACKLIST,
    SPLIT_N_POSES,
    SPLIT_SEED_OFFSETS,
)
from robot_imitation_glue.ur5station.bottle_station_env import BottleStation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_PATHS = [
    _REPO_ROOT / "airo-mono" / "airo-typing",
    _REPO_ROOT / "airo-mono" / "airo-spatial-algebra",
    _REPO_ROOT / "airo-mono" / "airo-robots",
    _REPO_ROOT / "airo-mono" / "airo-camera-toolkit",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-agents",
    _REPO_ROOT / "airo-teleop-agents" / "airo-teleop-devices",
]
for _path in _LOCAL_PATHS:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

# bottle_station_env already put bottle_experiment/ on sys.path.
import touch_point_detector as rim_detector  # noqa: E402
from calibrate_and_hover_bottle import (  # noqa: E402
    DEFAULT_TCP_RIGHT_TO_BOTTLE,
    HOVER_HEIGHT_METERS,
    compute_hover_pose_above_bottle,
    orthonormalize_rotation,
    project_point_to_pixel,
    tcp_left_to_camera,
)
from lid_touch_point_annotator import ClickState  # noqa: E402
from motion_constants import MOVE_SPEED  # noqa: E402
from open_bottle_agent import (  # noqa: E402
    LEFT_HOME_JOINTS,
    LEFT_TRANSIT_JOINT_SPEED,
    RANDOM_SEED,
    RIGHT_JOINT_SPEED,
    RIGHT_NEUTRAL_JOINTS,
    generate_reachable_bottle_poses,
    make_pose,
)
from opening_motion import rotation_about_axis  # noqa: E402

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# The arm is stationary while measuring, so the grab is the only noise source: averaging several and
# requiring they agree is cheaper than trusting one, and the spread is the error bar on every number.
N_GRABS = 4
GRAB_SETTLE_S = 0.4
MAX_GRAB_SPREAD_PX = 3.0

# Measured floor of the fit residual on real data is ~5 px, and it is NOT the estimator: noise-free
# data through the same 12 deg mount tilt leaves 0.35 px. Two independent detectors (the absolute colour
# window and the colour-free chromaticity lock) agree to 0.2 px on the fitted bias and leave the same
# ~5 px radial residual, which is the sticker's own appearance changing as the camera orbits and the
# view direction swings +-16 deg. So the gate is set to catch a lock that MOVES, not a sticker that
# glints: residuals past this mean the landmark jumped, and the bias/miss split stops being trustworthy.
MAX_FIT_RESIDUAL_PX = 6.0

# A circle fit on a short arc is unstable: the rim must be seen over at least this much of its
# circumference before its centre is believed.
MIN_RIM_ARC_DEG = 100.0

# The anchored search window, as a multiple of the rim's predicted pixel radius. Below ~1.2 the rim
# itself gets clipped; above ~1.5 the bench starts coming in at steep cap tilts.
SEARCH_WINDOW_FACTOR = 1.35

PLAUSIBLE_AXIS_ERROR_MM = 8.0
PLAUSIBLE_MISS_MM = 25.0

# Shape gates, not colour gates: on the rig frames the accepted colour window matched 19 blobs, so
# colour alone cannot say which is the sticker. A filled disc covers pi/4 = 0.785 of its bounding box
# and is convex; the cap's red rim is a thin arc (extent ~0.2-0.6, low solidity) and the wooden bench
# is amorphous. These cut the 19 down to the one real sticker on every frame tested.
MIN_BLOB_AREA_PX = 150
MAX_BLOB_AREA_PX = 8000
MIN_BLOB_EXTENT = 0.62
MAX_BLOB_EXTENT = 0.95
MIN_BLOB_SOLIDITY = 0.80

# A gross area collapse across a sweep does mean the blob is being clipped -- but only grossly: on the
# first run's pose 1 the colour window lost 46% of its area at -90 deg (1353 px -> 616 px) while its
# centroid moved 0.9 px against the colour-free detector. So this reports rather than discards until the
# deviation is large enough to have actually moved the landmark.
MAX_AREA_DEVIATION = 0.5
MAX_RADIUS_DEVIATION = 0.12

# Chromaticity is a ratio, so it survives the directional lighting that moves the sticker's ABSOLUTE
# hsv between right-arm poses (s 208 -> 168 with v 166 -> 205 across the two poses of the first run),
# which is what makes a per-pose automatic re-lock possible instead of a click every time.
CHROMA_MIN = 0.06
# Keep everything within this fraction of the search disc's peak chromaticity. Tuned on the rig's own
# frames, where it recovers exactly one candidate at 10/10 angles: the sticker measured 0.538 against a
# disc peak of 0.670 (the peak is some other colourful pixel, not the target), so requiring 0.6 or more
# of the peak clips the sticker's own body and it fragments -- 0.6 found 5/10 and 0.9 found none. Low
# enough to keep the sticker whole, and the shape gates are what then reject the rim arc.
CHROMA_PEAK_RATIO = 0.4

# The sticker search disc, about the predicted cap pixel. At ~3.8 px/mm this allows ~24 mm of miss,
# which is far more than any believable centring error and far less than the distance to the bench.
STICKER_SEARCH_RADIUS_PX = 90.0

# A fit needs enough circumference to separate the constant term from the rotating one; a set
# symmetric about 0 keeps the fit's columns orthogonal, which is why +-90 deg already does it.
MIN_ROLL_SPAN_DEG = 60.0

# +-90 deg of wrist roll is already a lot for the camera cable and for joint 6 from the hover
# configuration, and the sweep alternates sides (0, -45, +45, -90, +90) so a slow drift lands on both
# halves of the circle instead of pushing the whole fit one way.
DEFAULT_ROLL_DEG = [-90, -45, 0, 45, 90]

LANDMARKS = ("sticker", "rim")


def to_bgr(image_rgb_int: np.ndarray) -> np.ndarray:
    """env observation image -> BGR. Same conversion the collector applies before
    detect_touch_point, so the pixels measured here are the pixels the dataset records."""
    return ImageConverter.from_numpy_int_format(image_rgb_int).image_in_opencv_format


def chromaticity(image_bgr: np.ndarray) -> np.ndarray:
    """(max - min) / (sum + 1) per pixel: how colourful, independent of how bright.

    The first run died because it trusted absolute HSV: re-standing the bottle at a new right-arm pose
    changed the sticker's own pixel values (s 208 -> 168, v 166 -> 205) and the window that was correct
    at pose 1 found nothing but background at pose 2. Directional light scales intensity, which moves S
    and V together, while the ratio of channel spread to channel sum barely notices -- so the lock that
    has to survive a relighting is built on this, and the absolute window is only ever used within a
    single pose, where it is good enough to track the same sticker across a +-90 deg wrist roll.
    """
    f = image_bgr.astype(np.float32)
    return (f.max(axis=2) - f.min(axis=2)) / (f.sum(axis=2) + 1.0)


def gated_blobs(mask: np.ndarray, anchor: tuple[float, float] | None = None,
                anchor_radius_px: float = 90.0) -> tuple[list[dict], int]:
    """Blobs in a binary mask that pass the shape gates -- discs in, arcs and amorphous regions out.

    Returns (survivors sorted by distance to the anchor, n_before_gates). Each survivor carries its
    centroid, area, extent and solidity; extent/solidity of a chosen component are computed from its
    contour because a thin rim arc and a filled sticker disc are indistinguishable by area alone.
    """
    if anchor is not None:
        window = np.zeros_like(mask)
        cv2.circle(window, _pt(anchor), int(anchor_radius_px), 255, -1)
        mask = cv2.bitwise_and(mask, window)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not MIN_BLOB_AREA_PX <= area <= MAX_BLOB_AREA_PX:
            continue
        box_area = float(stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT])
        extent = area / box_area if box_area > 0 else 1.0
        if not MIN_BLOB_EXTENT <= extent <= MAX_BLOB_EXTENT:
            continue
        component = np.zeros(mask.shape, np.uint8)
        component[labels == i] = 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 1.0
        if solidity < MIN_BLOB_SOLIDITY:
            continue
        distance = 0.0 if anchor is None else float(np.hypot(*(centroids[i] - np.asarray(anchor))))
        blobs.append({"px": (float(centroids[i][0]), float(centroids[i][1])), "area_px": area,
                      "extent": round(float(extent), 3), "solidity": round(float(solidity), 3),
                      "distance_to_anchor": distance})
    blobs.sort(key=lambda b: b["distance_to_anchor"])
    return blobs, max(0, n_labels - 1)


@dataclass
class StickerModel:
    """The sticker's appearance, learned per pose: a chromaticity lock, then a tight colour window.

    Two stages because neither alone works. Colour alone cannot pick the sticker out of this scene (on
    the rig frames one accepted HSV window matched 19 blobs, the largest a 2192 px patch of wooden
    bench), and shape alone is jittery frame to frame. So `lock` finds it by illumination-invariant
    chromaticity plus the disc gates and no prior knowledge at all, and only then is the absolute HSV
    window -- valid for the rest of THIS pose -- sampled from where that lock says the sticker is.

    Centroids come from connectedComponentsWithStats, which is invariant to in-plane rotation: the one
    property a roll sweep demands, since a template match would drift through the sweep and be mistaken
    for bias. Measured on saved frames a filled disc holds its centroid to 0.02 px across a 315 deg
    rotation, while an off-centre-mass shape slides 3.4 px -- 1.4 mm of invented bias. The shape gates
    make that a refusal rather than a risk: a triangle or arrow is never selected at any angle, so a
    non-round sticker stops the run instead of silently corrupting it. A round sticker is safe at any
    view angle, since an ellipse always fills pi/4 of its bounding box.
    """

    bands: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    sat_band: tuple[int, int] = (0, 255)
    val_band: tuple[int, int] = (0, 255)
    tol: int = 20
    seed_xy: tuple[int, int] | None = None
    last_center: tuple[float, float] | None = None
    hsv_here: tuple[int, int, int] | None = None

    @property
    def seeded(self) -> bool:
        return bool(self.bands)

    def seed_at(self, image_bgr: np.ndarray, xy: tuple[int, int]) -> None:
        """Rebuild the absolute colour window from a patch of THIS frame -- the per-pose re-lock."""
        if xy is None:
            return
        self.seed_xy = xy
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        r = 4
        x, y = int(round(xy[0])), int(round(xy[1]))
        ys, xs = np.mgrid[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)]
        h0, s0, v0 = (int(round(float(np.median(hsv[ys, xs, c])))) for c in range(3))
        self.hsv_here = (h0, s0, v0)

        def band(center, limit):
            return (max(0, center - self.tol), min(limit, center + self.tol))

        self.sat_band, self.val_band = band(s0, 255), band(v0, 255)
        hue_lo, hue_hi = h0 - self.tol, h0 + self.tol
        if hue_lo < 0:
            self.bands = [((hue_lo % 180, 0), (179, 255)), ((0, 0), (hue_hi, 255))]
        elif hue_hi > 179:
            self.bands = [((hue_lo, 0), (179, 255)), ((0, 0), (hue_hi % 180, 255))]
        else:
            self.bands = [((hue_lo, 0), (hue_hi, 255))]
        self.last_center = None

    def reseed(self, image_bgr: np.ndarray) -> None:
        """Re-apply the window around the remembered point, so [ and ] take effect immediately."""
        if self.seed_xy is None:
            return
        self.seed_at(image_bgr, self.seed_xy)
        print(f"[sticker] h={self.hsv_here[0]} s={self.hsv_here[1]} v={self.hsv_here[2]} "
              f"tol={self.tol} -> hue bands {self.bands}, sat {self.sat_band}, val {self.val_band}")

    def lock(self, image_bgr: np.ndarray, anchor: tuple[float, float],
             anchor_radius_px: float) -> dict | None:
        """Find the sticker with no colour prior at all: the most chromatic gated blob near the anchor.

        This is what makes a new right-arm pose survivable -- it re-finds the sticker after the
        relighting that defeated the absolute window, so the operator clicks once rather than per pose.

        The threshold is relative to the peak chromaticity in the search disc rather than absolute or
        quantile-based: an absolute floor dies with the lighting (the whole point of this lock), and a
        top-k%-quantile floor mis-scales badly because the sticker is only a few percent of the disc's
        area, so any colourful clutter in the disc pushes it above the sticker. Peaking on the sticker
        itself and keeping everything within `CHROMA_PEAK_RATIO` of that is both illumination-free and
        size-free; the shape gates then drop the rim arc even when the arc is the more chromatic thing
        in view, because a thin arc cannot survive them.
        """
        chroma = chromaticity(image_bgr)
        reach = int(anchor_radius_px)
        x0, y0 = max(0, int(anchor[0] - reach)), max(0, int(anchor[1] - reach))
        x1, y1 = min(chroma.shape[1], int(anchor[0] + reach)), min(chroma.shape[0], int(anchor[1] + reach))
        patch = chroma[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        peak = float(patch.max())
        floor = max(CHROMA_MIN, CHROMA_PEAK_RATIO * peak)
        mask = np.zeros(chroma.shape, np.uint8)
        mask[y0:y1, x0:x1] = np.where(patch >= floor, 255, 0).astype(np.uint8)
        blobs, n_raw = gated_blobs(mask, anchor=anchor, anchor_radius_px=anchor_radius_px)
        if not blobs:
            return {"blobs_seen": n_raw, "survivors": 0, "chroma_floor": round(float(floor), 3),
                    "chroma_peak": round(peak, 3)}
        return {"blobs_seen": n_raw, "survivors": len(blobs), "chroma_floor": round(float(floor), 3),
                "chroma_peak": round(peak, 3), **blobs[0]}

    def mask(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (hue_lo, _), (hue_hi, _) in self.bands:
            lower = np.array([hue_lo, self.sat_band[0], self.val_band[0]])
            upper = np.array([hue_hi, self.sat_band[1], self.val_band[1]])
            mask |= cv2.inRange(hsv, lower, upper)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def find(self, image_bgr: np.ndarray, anchor: tuple[float, float],
             anchor_radius_px: float) -> dict | None:
        """The sticker at this angle: colour-free lock first, the pose's colour window as backup.

        The window cannot be trusted WITHIN a sweep either, not merely across poses: orbiting the camera
        around the gripper axis changes the light falling on the sticker, and a tol=20 window learned at
        roll 0 lost the sticker at every other angle of the dry run while the lock held all five. The
        lock is therefore the primary cue per frame and the window a fallback, which also buys a free
        consistency check -- when both fire, `cue_gap_px` is how much the two disagree (0.4-3.9 px on
        the rig's own frames).
        """
        locked = self.lock(image_bgr, anchor, anchor_radius_px)
        windowed = self.detect(image_bgr, anchor=anchor, anchor_radius_px=anchor_radius_px)
        from_lock = bool(locked and locked.get("px") and locked.get("survivors") == 1)
        from_window = bool(windowed and windowed.get("px"))
        if from_lock:
            chosen = {**locked, "source": "chroma"}
        elif from_window:
            chosen = {**windowed, "source": "colour"}
        else:
            return {"source": None, "survivors": (locked or {}).get("survivors", 0),
                    "chroma_peak": (locked or {}).get("chroma_peak"),
                    "blobs_matched": (windowed or {}).get("blobs_matched", 0)}
        if from_lock and from_window:
            chosen["cue_gap_px"] = round(float(np.linalg.norm(
                np.asarray(locked["px"]) - np.asarray(windowed["px"]))), 2)
        self.last_center = chosen["px"]
        return chosen

    def detect(self, image_bgr: np.ndarray, anchor: tuple[float, float] | None = None,
               anchor_radius_px: float = 90.0) -> dict | None:
        """The colour window's blob, gated by shape and confined to the anchor disc.

        Anchoring is not cosmetic: without it this window matched 19 blobs, the largest a 2192 px patch
        of bench at the frame edge, and the nearest-to-previous-centroid rule then happily chained the
        lock across the scene. The shape gates do the rest -- the cap's red rim matches the colour and
        is only excluded by being an arc.
        """
        if not self.seeded:
            return None
        blobs, n_raw = gated_blobs(self.mask(image_bgr), anchor=anchor,
                                   anchor_radius_px=anchor_radius_px)
        if not blobs:
            return {"blobs_matched": n_raw, "survivors": 0}
        target = self.last_center if self.last_center is not None else anchor
        pick = min(blobs, key=lambda b: float(np.hypot(*(np.asarray(b["px"]) - np.asarray(target))))) \
            if target is not None else max(blobs, key=lambda b: b["area_px"])
        self.last_center = pick["px"]
        return {"blobs_matched": n_raw, "survivors": len(blobs), **pick}


def detect_rim_centre(image_bgr: np.ndarray, anchor_xy: tuple[float, float], expected_radius_px: float,
                      hover_height_m: float) -> dict | None:
    """The cap's own rim, circle-fitted: the geometric cap centre, with no sticker involved.

    Reuses `touch_point_detector`'s red segmentation and algebraic circle fit -- the machinery the
    motion itself is built on -- but hands it a window anchored on the model's predicted cap centre
    instead of accepting the module's central-ROI crop, which assumes a centred lid and on these
    frames cut 40% of the rim pixels while keeping the bench.

    Returns the fitted centre (full-frame px), its radius, the fit's own residual and the arc's
    angular span, or None when there is no trustworthy circle.
    """
    half = int(expected_radius_px * SEARCH_WINDOW_FACTOR)
    x0 = int(np.clip(anchor_xy[0] - half, 0, image_bgr.shape[1]))
    y0 = int(np.clip(anchor_xy[1] - half, 0, image_bgr.shape[0]))
    x1 = int(np.clip(anchor_xy[0] + half, x0 + 1, image_bgr.shape[1]))
    y1 = int(np.clip(anchor_xy[1] + half, y0 + 1, image_bgr.shape[0]))
    crop = image_bgr[y0:y1, x0:x1]
    if crop.shape[0] < 40 or crop.shape[1] < 40:
        return None
    components = rim_detector._red_components(crop, roi_margin_frac=0.0)
    if not components:
        return None
    points = np.concatenate(components, axis=0).astype(float)
    if len(points) < 30:
        return None
    center, radius, residual = rim_detector._fit_rim_circle(points, expected_radius_px)
    span = rim_detector._angular_span_deg(points, center)
    out = {"px": (float(center[0] + x0), float(center[1] + y0)), "radius_px": round(float(radius), 2),
           "fit_residual_px": round(float(residual), 2), "arc_deg": round(float(span), 1),
           "rim_pixels": int(len(points)), "components": len(components)}
    # A clipped arc fits a smaller, wrong circle: on the real sweep the radius fell to 133.9 px against
    # the 163.9 px prior at +90 deg, exactly where the gripper finger covers the near rim. Radius
    # agreement is what says whether this centre is the cap's or an artifact of what was visible.
    radius_error = abs(float(radius) - expected_radius_px) / expected_radius_px
    if not np.isfinite(residual) or residual > 8.0:
        out["rejected"] = f"circle fit residual {residual:.1f}px"
    elif span < MIN_RIM_ARC_DEG:
        out["rejected"] = f"arc spans {span:.0f} deg (needs {MIN_RIM_ARC_DEG:.0f})"
    elif radius_error > MAX_RADIUS_DEVIATION:
        out["rejected"] = (f"fitted radius {radius:.0f}px is {100 * radius_error:.0f}% off the "
                           f"{expected_radius_px:.0f}px prior -- arc clipped, probably by the finger")
    return out


def _xy(value) -> tuple[float, float] | None:
    """A usable (x, y) pair or None -- and never `if value`, which dies on a 2-element array.

    The marks handed to the live view come from different sources, some tuples and some the raw
    `project_point_to_pixel` ndarray, and `if marks.get("predict")` on that array raises "truth value
    of an array with more than one element is ambiguous". That killed the first rig run of this script
    at its very first angle, after the arm was already in position, so every mark goes through here.
    """
    if value is None:
        return None
    flat = np.asarray(value, dtype=float).reshape(-1)
    return None if flat.size < 2 else (float(flat[0]), float(flat[1]))


def _draw_marks(overlay: np.ndarray, marks: dict, lines: list[str]) -> None:
    """Annotate the live frame in place: prediction first, so the measured locks draw on top of it."""
    cv2.drawMarker(overlay, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    principal = _xy(marks.get("principal"))
    if principal is not None:
        cv2.drawMarker(overlay, _pt(principal), (200, 200, 200), cv2.MARKER_TILTED_CROSS, 16, 1)
    predict = _xy(marks.get("predict"))
    if predict is not None:
        cv2.drawMarker(overlay, _pt(predict), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
    window = marks.get("window")
    if window is not None and _xy(window[0]) is not None:
        cv2.circle(overlay, _pt(window[0]), int(window[1]), (90, 90, 90), 1)
    sticker = _xy(marks.get("sticker"))
    if sticker is not None:
        cv2.drawMarker(overlay, _pt(sticker), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
    rim = marks.get("rim")
    if rim is not None and _xy(rim[0]) is not None:
        cv2.circle(overlay, _pt(rim[0]), int(rim[1]), (255, 0, 0), 2)
    for i, text in enumerate(lines):
        cv2.putText(overlay, text, (10, 22 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class LiveView:
    """One window for the whole run, so the lock can be watched rather than trusted.

    Keys only register while the window has focus, so it also watches WND_PROP_VISIBLE and the run is
    aborted if the window is closed -- an unnoticed closed window would otherwise mean flying blind,
    which is the failure mode this exists to remove.
    """

    def __init__(self, name: str = "hover centre bias (live)"):
        self.name = name
        self.click = ClickState()
        self.window = None
        self.pending_click = None

    def _ensure(self):
        if self.window is not None and cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
            raise SystemExit("Live window was closed -- aborting rather than measuring unseen.")
        if self.window is None:
            cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.name, self.click.on_mouse_event)
            self.window = self.name

    def show(self, image_bgr: np.ndarray, lines: list[str], marks: dict) -> str | None:
        """Draw, pump the event loop, and return 'r' (re-seed) or 'q' (abort) if pressed."""
        self._ensure()
        overlay = image_bgr.copy()
        _draw_marks(overlay, marks, lines)
        cv2.imshow(self.window, overlay)
        key = cv2.waitKey(1) & 0xFF
        if self.click.point is not None:
            self.pending_click = self.click.point
            self.click.point = None
        if key in (27, ord("q")):
            return "q"
        return "r" if key == ord("r") else None

    def close(self) -> None:
        if self.window is not None:
            cv2.destroyWindow(self.window)
            self.window = None


def _pt(xy) -> tuple[int, int]:
    return (int(round(float(xy[0]))), int(round(float(xy[1]))))


def _grab_frame(env: BottleStation) -> np.ndarray:
    return to_bgr(env.get_observations()["wrist_image_original"])


def lock_sticker_for_pose(env: BottleStation, live: LiveView, model: StickerModel | None,
                          anchor: tuple[float, float]) -> StickerModel:
    """Get the sticker's colour window for THIS pose, asking the operator at most once for the run.

    The reason one click can be enough is the hover construction itself: the gripper axis crosses the
    cap plane exactly `hover_height` from the TCP and the camera is rigid to the TCP, so the model's cap
    centre projects to the same pixel at roll 0 for every pose -- which is why the search disc is a
    fixed place on screen and the previous pose's window and position are both still valid predictions.
    So try, in order: carry the previous window into the same disc, then the colour-free chromaticity
    lock, and only then ask.

    Carrying is not blind trust -- the carried window has to produce exactly one gated blob near the
    predicted pixel, which is what catches the relighting that defeated the first run's pose 2.
    """
    model = model if model is not None and model.seeded else StickerModel()
    image_bgr = _grab_frame(env)

    if model.seeded:
        probe = StickerModel(tol=model.tol, bands=model.bands, sat_band=model.sat_band,
                             val_band=model.val_band)
        found = probe.detect(image_bgr, anchor=anchor, anchor_radius_px=STICKER_SEARCH_RADIUS_PX)
        if found and found.get("px") and found["survivors"] == 1:
            model.last_center = found["px"]
            print(f"[sticker] carried the previous window: locked at ({found['px'][0]:.1f},"
                  f"{found['px'][1]:.1f}) {found['distance_to_anchor']:.1f}px from the predicted cap "
                  f"pixel, area={found['area_px']}px extent={found['extent']}")
            return model
        why = "carried window found nothing that passes the shape gates" if not found or not found.get("px") \
            else f"carried window found {found['survivors']} disc-shaped candidates"
    else:
        why = "no colour window yet"

    locked = model.lock(image_bgr, anchor, STICKER_SEARCH_RADIUS_PX)
    survivors = (locked or {}).get("survivors", 0)
    if survivors == 1:
        model.seed_at(image_bgr, _pt(locked["px"]))
        model.last_center = locked["px"]
        print(f"[sticker] auto-locked by chromaticity at ({locked['px'][0]:.1f},{locked['px'][1]:.1f}) "
              f"area={locked['area_px']}px extent={locked['extent']} hsv={model.hsv_here} "
              f"(floor {locked['chroma_floor']}, {locked['blobs_seen']} raw blobs, "
              f"{why})")
        return model
    if survivors == 0:
        reason = f"{why}, and the colour-free lock found nothing disc-shaped " \
                 f"({(locked or {}).get('blobs_seen', 0)} raw blobs, floor {(locked or {}).get('chroma_floor')})"
    else:
        reason = f"{why}, and the colour-free lock found {survivors} disc-shaped candidates"
    return _seed_by_click(live, model, anchor, env, reason)


def _seed_message(model: StickerModel, found: dict | None) -> str | None:
    """None when the current selection is worth accepting, otherwise why it is not."""
    if not model.seeded:
        return "click the sticker first"
    if found is None:
        return "this window selects nothing that passes the shape gates -- widen with ] or click again"
    if found["survivors"] > 1:
        return (f"{found['survivors']} blobs survive the gates -- narrow with [ so only the sticker "
                "does, unless the extra one really is a second sticker")
    return None


def _seed_lines(model: StickerModel, found: dict | None) -> list[str]:
    if found is None:
        return [f"SEED  click the sticker, [ ] tol={model.tol}, Enter accept, q abort",
                "NOTHING survives the gates -- widen with ] or click the sticker"]
    return [f"SEED  click the sticker, [ ] tol={model.tol}, Enter accept, q abort",
            f"area={found['area_px']}px extent={found['extent']} solidity={found['solidity']} at "
            f"{_pt(found['px'])} {found['distance_to_anchor']:.1f}px from the model's cap pixel",
            f"colour matches {found['blobs_matched']} blobs, {found['survivors']} survive the gates"]


def _apply_tolerance_key(model: StickerModel, pressed: int, image_bgr: np.ndarray) -> None:
    if pressed == ord("["):
        model.tol = max(4, model.tol - 4)
        model.reseed(image_bgr)
    elif pressed == ord("]"):
        model.tol = min(60, model.tol + 4)
        model.reseed(image_bgr)


def _seed_by_click(live: LiveView, model: StickerModel, anchor: tuple[float, float],
                   env: BottleStation, reason: str) -> StickerModel:
    """Point-and-tune until the window selects the sticker and nothing else."""
    print(f"[sticker] {reason} -- clicking needed. Click the sticker in the live window, [ ] to tune "
          "the tolerance until only one blob survives, Enter to accept, q to abort.")
    while True:
        image_bgr = _grab_frame(env)
        if live.pending_click is not None:
            model.seed_at(image_bgr, live.pending_click)
            live.pending_click = None
            print(f"[sticker] h={model.hsv_here[0]} s={model.hsv_here[1]} v={model.hsv_here[2]} "
                  f"tol={model.tol}")
        model.last_center = None
        hit = model.detect(image_bgr, anchor=anchor, anchor_radius_px=STICKER_SEARCH_RADIUS_PX)
        found = hit if hit and hit.get("px") else None
        key = live.show(image_bgr, _seed_lines(model, found),
                        {"predict": anchor, "sticker": None if found is None else found["px"],
                         "window": (anchor, STICKER_SEARCH_RADIUS_PX)})
        if key == "q":
            raise SystemExit("Aborted at sticker seeding -- nothing measured.")
        for _ in range(3):
            pressed = cv2.waitKey(30) & 0xFF
            _apply_tolerance_key(model, pressed, image_bgr)
            if pressed in (13, 10):
                problem = _seed_message(model, found)
                if problem is None:
                    return model
                print(f"[sticker] {problem}")


def rolled_pose(base_pose: np.ndarray, theta_rad: float) -> np.ndarray:
    """base_pose rotated about its own z axis (the gripper approach axis), position untouched.

    Rotating about the TOOL z through the TCP is the whole trick: that line is the rotation axis, so
    it does not move, which is what makes the roll-invariant term of the fit a property of the camera
    alone.
    """
    rotation = base_pose[:3, :3] @ rotation_about_axis(np.array([0.0, 0.0, 1.0]), theta_rad)
    return make_pose(orthonormalize_rotation(rotation), base_pose[:3, 3].copy())


def measured_roll_rad(hover_rotation: np.ndarray, actual_rotation: np.ndarray) -> float:
    """The roll angle ACTUALLY achieved, from the encoders, not the one commanded.

    airo-mono's moveL awaitable only warns when its 30 s timeout expires, so a commanded 90 deg can
    stop short of it. Feeding measured angles to the fit turns an incomplete roll into a weaker sample
    instead of a poisoned one.
    """
    relative = hover_rotation.T @ actual_rotation
    return float(np.arctan2(relative[1, 0], relative[0, 0]))


def tool_axis_cap_crossing(tcp: np.ndarray, cap_pose: np.ndarray) -> np.ndarray | None:
    """Where the achieved gripper axis crosses the model cap plane: the point whose image position is
    the roll-invariant term of the fit, and the point the hover pose exists to aim at."""
    cap_normal, cap_center = cap_pose[:3, 2], cap_pose[:3, 3]
    axis = tcp[:3, 2]
    denom = float(np.dot(cap_normal, axis))
    if abs(denom) < 1e-6:
        return None
    t = float(np.dot(cap_normal, cap_center - tcp[:3, 3]) / denom)
    return tcp[:3, 3] + t * axis


def predicted_cap_pixel(intrinsics: np.ndarray, hover_height_m: float) -> tuple[float, float]:
    """Where the model's cap centre must appear at roll 0 -- for ANY bottle pose.

    This is the reason one click covers a whole run: the gripper axis crosses the cap plane exactly
    `hover_height` in front of the TCP by construction, and the camera is rigid to the TCP, so the
    projected pixel depends on the mount and the standoff and on nothing about the bottle. Verified
    against the first run's own log: the model's cap centre came out at the same pixel in all five
    frames of a sweep, as it must.
    """
    v = np.linalg.inv(tcp_left_to_camera) @ np.array([0.0, 0.0, float(hover_height_m), 1.0])
    px = intrinsics @ (v[:3] / v[2])
    return (float(px[0]), float(px[1]))


def normalized_offset(pixel_xy, intrinsics: np.ndarray) -> np.ndarray:
    """Pixel -> normalised camera coordinates about the PRINCIPAL POINT.

    The fit must be referenced to the principal point, not the frame centre: the roll-invariant term
    is defined relative to the pinhole model, and only that difference is a calibration error.
    """
    cx, cy, fx, fy = intrinsics[0, 2], intrinsics[1, 2], intrinsics[0, 0], intrinsics[1, 1]
    xy = np.asarray(pixel_xy, dtype=float)
    return np.array([(xy[0] - cx) / fx, (xy[1] - cy) / fy])


def normalized_point_of(point_base: np.ndarray, tcp: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    """A base-frame point's normalised image position at this TCP -- run through the same two
    functions the measured offsets go through, so prediction and measurement are comparable."""
    pixel = project_point_to_pixel(point_base, intrinsics, tcp @ tcp_left_to_camera)
    return None if pixel is None else normalized_offset(pixel, intrinsics)


def fit_roll_circle(samples: list[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares for n(theta) = b + Rz(-theta) q; returns (b, q, rms residual in those units).

    Linear in the four unknowns (b_x, b_y, q_x, q_y) because Rz(-theta) = [[cos, sin], [-sin, cos]].
    """
    rows, obs = [], []
    for theta, n in samples:
        c, s = np.cos(theta), np.sin(theta)
        rows.extend(([1.0, 0.0, c, s], [0.0, 1.0, -s, c]))
        obs.extend((n[0], n[1]))
    design = np.asarray(rows)
    sol, *_ = np.linalg.lstsq(design, np.asarray(obs), rcond=None)
    rms = float(np.sqrt(np.mean((design @ sol - np.asarray(obs)) ** 2)))
    return sol[:2], sol[2:], rms


def analyse_landmark(samples: list[dict], landmark: str, intrinsics: np.ndarray, cap_pose: np.ndarray,
                     depth_mm: float) -> dict:
    """Split one landmark's sweep into the projection bias (roll-invariant) and the axis miss."""
    # Callers hand this either an ndarray (live) or the JSON-roundtripped list (reporting), and the
    # difference is a crash on the rig's first report if it is not normalised here.
    intrinsics = np.asarray(intrinsics, dtype=float)
    cap_pose = np.asarray(cap_pose, dtype=float)
    seen = [s for s in samples if s["landmarks"].get(landmark, {}).get("offset") is not None]
    dropped = []
    if landmark == "sticker" and seen:
        # Occlusion is invisible to a colour lock -- a partly hidden sticker is the same colour, just
        # smaller -- so the areas across a sweep must agree before the fit is allowed to treat them as
        # one rigid point. Pose 1's -90 deg sample (616 px against a 1200 px median) is what this is for.
        areas = np.array([s["landmarks"][landmark]["area_px"] for s in seen], dtype=float)
        median = float(np.median(areas))
        keep = []
        for s, area in zip(seen, areas):
            if abs(area - median) > MAX_AREA_DEVIATION * median:
                dropped.append({"roll_deg": s["roll_deg"], "area_px": int(area),
                                "reason": f"area {int(area)}px is {100 * abs(area - median) / median:.0f}% "
                                          f"off the sweep median {int(median)}px -- clipped"})
            else:
                keep.append(s)
        seen = keep
    usable = seen
    if len(usable) < 3:
        return {"error": f"only {len(usable)}/{len(samples)} angles yielded the {landmark}"
                         + (f" ({len(dropped)} dropped as clipped)" if dropped else ""),
                "dropped": dropped}
    rolls = [s["roll_rad"] for s in usable]
    span = max(rolls) - min(rolls)
    if np.degrees(span) < MIN_ROLL_SPAN_DEG:
        return {"error": f"the {landmark} angles cover only {np.degrees(span):.0f} deg -- under "
                         f"{MIN_ROLL_SPAN_DEG:.0f} deg bias and miss are not separable"}

    bias_norm, q_norm, rms = fit_roll_circle([(s["roll_rad"], np.asarray(s["landmarks"][landmark]["offset"]))
                                              for s in usable])
    focal = float(np.mean(intrinsics[[0, 1], [0, 1]]))
    reference = min(usable, key=lambda s: abs(s["roll_rad"]))
    tcp_ref = np.asarray(reference["tcp"])
    axis_point = tool_axis_cap_crossing(tcp_ref, cap_pose)
    predicted = None if axis_point is None else normalized_point_of(axis_point, tcp_ref, intrinsics)
    if predicted is None:
        return {"error": f"{landmark}: the gripper axis does not project -- cannot predict it"}

    error_norm = bias_norm - predicted
    error_px = np.array([error_norm[0] * intrinsics[0, 0], error_norm[1] * intrinsics[1, 1]])
    m_rot = tcp_left_to_camera[:3, :3]
    miss_tool = depth_mm * (m_rot[:2, :2] @ q_norm)
    miss_base = tcp_ref[:3, :2] @ miss_tool
    miss_cap = cap_pose[:3, :2].T @ miss_base
    return {
        "n_angles": len(usable),
        "roll_span_deg": round(float(np.degrees(span)), 1),
        "axis_px_measured": [round(float(v), 2) for v in np.asarray(bias_norm) * intrinsics[[0, 1], [0, 1]]],
        "axis_px_predicted": [round(float(v), 2) for v in np.asarray(predicted) * intrinsics[[0, 1], [0, 1]]],
        "axis_error_px": [round(float(v), 2) for v in error_px],
        "axis_error_mag_px": round(float(np.hypot(*error_px)), 2),
        "axis_error_mm": round(float(np.linalg.norm(depth_mm * error_norm)), 2),
        "miss_mm": round(float(np.linalg.norm(miss_tool)), 2),
        "miss_tool_mm": [round(float(v), 2) for v in miss_tool],
        "miss_base_mm": [round(float(v), 2) for v in miss_base],
        "miss_cap_mm": [round(float(v), 2) for v in miss_cap],
        "miss_cap_dir_deg": round(float(np.degrees(np.arctan2(miss_cap[1], miss_cap[0]))), 1),
        "fit_residual_px": round(rms * focal, 2),
        "dropped": dropped,
        "move_tcp_by_base_mm": [round(float(v), 2) for v in miss_base],
    }


def grab_landmarks(env: BottleStation, model: StickerModel, intrinsics: np.ndarray, cap_pose: np.ndarray,
                   tcp: np.ndarray, anchor_px: np.ndarray, hover_height_m: float) -> tuple[dict, np.ndarray]:
    """Both landmarks, averaged over N_GRABS stationary frames, with their spread as the error bar."""
    sticker_pts, sticker_areas, sticker_extents, sticker_sources = [], [], [], []
    rim_pts, rim_infos, frames = [], [], []
    for _ in range(N_GRABS):
        image_bgr = to_bgr(env.get_observations()["wrist_image_original"])
        frames.append(image_bgr)
        radius = rim_detector._predict_rim_radius(hover_height_m)
        hit = model.find(image_bgr, tuple(anchor_px), STICKER_SEARCH_RADIUS_PX)
        if hit and hit.get("px"):
            sticker_pts.append(hit["px"])
            sticker_areas.append(hit["area_px"])
            sticker_extents.append(hit["extent"])
            sticker_sources.append((hit.get("source"), hit.get("cue_gap_px")))
        rim = detect_rim_centre(image_bgr, tuple(anchor_px), radius, hover_height_m)
        if rim is not None and "rejected" not in rim:
            rim_pts.append(rim["px"])
            rim_infos.append(rim)
        time.sleep(1 / 15)

    landmarks = {}
    if sticker_pts:
        pts = np.asarray(sticker_pts)
        landmarks["sticker"] = {"px": pts.mean(axis=0).tolist(), "area_px": int(np.median(sticker_areas)),
                                "extent": float(np.median(sticker_extents)), "n_grabs": len(pts),
                                "source": sorted({s for s, _ in sticker_sources if s}),
                                "cue_gap_px": max((g for _, g in sticker_sources if g is not None), default=None),
                                "spread_px": round(float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))), 3)}
    if rim_pts:
        pts = np.asarray(rim_pts)
        landmarks["rim"] = {"px": pts.mean(axis=0).tolist(), "n_grabs": len(pts),
                            "spread_px": round(float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))), 3),
                            **{k: float(np.median([ri[k] for ri in rim_infos]))
                               for k in ("radius_px", "fit_residual_px", "arc_deg")}}
    return landmarks, frames[-1]


def _attach_offsets(landmarks: dict, intrinsics: np.ndarray) -> dict:
    """Record each found landmark's normalised offset, the quantity the roll fit consumes."""
    for name in LANDMARKS:
        if name in landmarks:
            landmarks[name]["offset"] = [float(v) for v in
                                         normalized_offset(landmarks[name]["px"], intrinsics)]
    return landmarks


def _landmark_status(landmarks: dict, anchor) -> list[str]:
    """One status line per landmark, for both the console and the live window, so what he sees on
    screen is exactly what got recorded."""
    lines = []
    for name in LANDMARKS:
        got = landmarks.get(name)
        if got is None:
            lines.append(f"{name:7s} NOT FOUND")
            continue
        off = np.asarray(got["px"], float) - np.asarray(anchor, float)
        detail = (f"r={got['radius_px']:.0f}px arc={got['arc_deg']:.0f}deg" if name == "rim"
                  else f"area={got['area_px']}px extent={got['extent']} via {'+'.join(got.get('source') or ['?'])}")
        lines.append(f"{name:7s} {_pt(got['px'])} {float(np.linalg.norm(off)):5.1f}px off model  "
                     f"spread {got['spread_px']}px  {detail}")
        if name == "sticker" and got.get("cue_gap_px") and got["cue_gap_px"] > 5.0:
            lines.append(f"        note: the chromaticity lock and the colour window disagree by "
                         f"{got['cue_gap_px']}px -- one of the two is not the sticker")
        if got.get("spread_px", 0) > MAX_GRAB_SPREAD_PX:
            lines.append(f"        WARNING grab spread {got['spread_px']}px > {MAX_GRAB_SPREAD_PX} "
                         "-- not settled, or the blob is unstable")
    return lines


def measure_angle(env: BottleStation, intrinsics: np.ndarray, cap_pose: np.ndarray,
                  hover_pose: np.ndarray, model: StickerModel, roll_deg: float, axis_px: tuple,
                  hover_height_m: float, live: LiveView, frame_dir: Path, tag: str) -> tuple[dict, StickerModel]:
    """One roll angle: move, settle, measure both landmarks, show the operator what was found."""
    target = rolled_pose(hover_pose, np.radians(roll_deg))
    if not env.robot.is_tcp_pose_reachable(target):
        # Still attempted: reachability claims lie both ways, and because the fit consumes the achieved
        # angle, wherever the move lands is still a valid sample rather than a poisoned one.
        print(f"[measure]  roll {roll_deg:+6.1f}: reported unreachable -- attempting anyway")
    env.robot.move_linear_to_tcp_pose(target, linear_speed=MOVE_SPEED).wait()
    time.sleep(GRAB_SETTLE_S)

    tcp = env.get_robot_pose_se3()
    roll_rad = measured_roll_rad(hover_pose[:3, :3], tcp[:3, :3])
    if abs(np.degrees(roll_rad) - roll_deg) > 5.0:
        print(f"[measure]  roll {roll_deg:+6.1f} commanded but {np.degrees(roll_rad):+.1f} achieved "
              "-- fitting the achieved angle")
    anchor = project_point_to_pixel(cap_pose[:3, 3], intrinsics, tcp @ tcp_left_to_camera)
    sample = {"roll_cmd_deg": roll_deg, "roll_deg": round(float(np.degrees(roll_rad)), 2),
              "roll_rad": float(roll_rad), "tcp": tcp.tolist(), "model_px": None if anchor is None
              else _pt(anchor), "landmarks": {}}
    if anchor is None:
        print(f"[measure]  roll {np.degrees(roll_rad):+6.1f}: the model's cap centre is behind the "
              "camera -- nothing to search around, sample dropped")
        return sample, model

    landmarks, image_bgr = grab_landmarks(env, model, intrinsics, cap_pose, tcp,
                                          np.asarray(anchor, float), hover_height_m)
    _attach_offsets(landmarks, intrinsics)
    if not landmarks:
        print(f"[measure]  roll {np.degrees(roll_rad):+6.1f}: NEITHER landmark found -- dropped")
    sample["landmarks"] = landmarks
    _save_pair(image_bgr, sample, intrinsics, axis_px, frame_dir, tag)

    marks = {"principal": (intrinsics[0, 2], intrinsics[1, 2]), "predict": anchor,
             "window": (tuple(anchor), rim_detector._predict_rim_radius(hover_height_m) * SEARCH_WINDOW_FACTOR),
             "sticker": landmarks.get("sticker", {}).get("px"),
             "rim": (landmarks.get("rim", {}).get("px"), landmarks.get("rim", {}).get("radius_px", 0))}
    lines = [f"{tag}  roll {sample['roll_deg']:+.1f}deg (cmd {roll_deg:+.0f})",
             f"model cap centre {_pt(anchor)}   gripper axis predicts "
             f"{_pt(axis_px) if axis_px is not None else 'off-frame'}"] + _landmark_status(landmarks, anchor)
    key = live.show(image_bgr, lines, marks)
    if key == "q":
        raise SystemExit("Aborted from the live window.")
    if key == "r":
        model = _seed_by_click(live, model, tuple(anchor), env, "re-seed requested")
        landmarks, _ = grab_landmarks(env, model, intrinsics, cap_pose, tcp,
                                      np.asarray(anchor, float), hover_height_m)
        _attach_offsets(landmarks, intrinsics)
        sample["landmarks"] = landmarks

    print(f"[measure]  roll {sample['roll_deg']:+6.1f}: " + "  ".join(
        f"{name}={_pt(sample['landmarks'][name]['px']) if name in sample['landmarks'] else 'missing'}"
        for name in LANDMARKS))
    return sample, model


def measure_pose(env: BottleStation, intrinsics: np.ndarray, cap_pose: np.ndarray, hover_pose: np.ndarray,
                 model: StickerModel, roll_deg_list: list[float], hover_height_m: float,
                 live: LiveView, frame_dir: Path, tag: str) -> tuple[dict, StickerModel]:
    """Roll the wrist through the sweep at the fixed hover pose, recording both landmarks per angle."""
    cap_center, cap_normal = cap_pose[:3, 3], cap_pose[:3, 2]
    tcp0 = env.get_robot_pose_se3()
    camera_pose0 = tcp0 @ tcp_left_to_camera
    height_mm = float(np.dot(cap_normal, tcp0[:3, 3] - cap_center)) * 1000.0
    # The px<->mm scale needs the depth along the VIEW axis; the hover height is along the cap normal,
    # and the two agree only if the gripper looks straight down that normal -- this mount is 12 deg off
    # it, so they differ by a lot.
    depth_mm = float(np.dot(camera_pose0[:3, 2], cap_center - camera_pose0[:3, 3])) * 1000.0
    predict0 = project_point_to_pixel(cap_center, intrinsics, camera_pose0)
    axis0 = tool_axis_cap_crossing(tcp0, cap_pose)
    axis_px = None if axis0 is None else project_point_to_pixel(axis0, intrinsics, camera_pose0)
    print(f"[measure] TCP {height_mm:.1f}mm above the cap plane; view-axis depth {depth_mm:.1f}mm "
          f"({intrinsics[0, 0] / depth_mm:.2f}px per mm); model puts the cap centre at "
          f"{None if predict0 is None else _pt(predict0)} and the gripper axis at "
          f"{None if axis_px is None else _pt(axis_px)}")

    angles = sorted(roll_deg_list, key=lambda a: (abs(a), a))
    samples = []
    for roll_deg in angles:
        sample, model = measure_angle(env, intrinsics, cap_pose, hover_pose, model, roll_deg, axis_px,
                                      hover_height_m, live, frame_dir, tag)
        samples.append(sample)

    return {"samples": samples, "hover_height_mm": round(height_mm, 1),
            "view_depth_mm": round(depth_mm, 1), "axis_px_predicted_first": None if axis_px is None
            else [round(float(v), 1) for v in axis_px]}, model


def _save_pair(image_bgr, sample: dict, intrinsics: np.ndarray, axis_px, frame_dir: Path, tag: str) -> None:
    """Raw + overlay per angle, so every number in the report traces back to its pixels."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    name = f"{tag}_roll{sample['roll_cmd_deg']:+04.0f}"
    cv2.imwrite(str(frame_dir / f"{name}_raw.jpg"), image_bgr)
    overlay = image_bgr.copy()
    cv2.drawMarker(overlay, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.drawMarker(overlay, _pt((intrinsics[0, 2], intrinsics[1, 2])), (200, 200, 200),
                   cv2.MARKER_TILTED_CROSS, 16, 1)
    if sample.get("model_px") is not None:
        cv2.drawMarker(overlay, _pt(sample["model_px"]), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
    if axis_px is not None:
        cv2.drawMarker(overlay, _pt(axis_px), (0, 165, 255), cv2.MARKER_CROSS, 26, 2)
    if "sticker" in sample["landmarks"]:
        cv2.drawMarker(overlay, _pt(sample["landmarks"]["sticker"]["px"]), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
    if "rim" in sample["landmarks"]:
        rim = sample["landmarks"]["rim"]
        cv2.circle(overlay, _pt(rim["px"]), int(rim["radius_px"]), (255, 0, 0), 2)
    cv2.imwrite(str(frame_dir / f"{name}_overlay.jpg"), overlay)
    sample["image"] = f"frames/{name}_overlay.jpg"


def report_measurement(measurements: list[dict]) -> None:
    """Per landmark, per grasp: the bias, the miss, and the agreement between them."""
    for measured in measurements:
        # The caller passes one measurement at a time, so an enumerate() here would label every grasp
        # "grasp 1" -- which it did, on the rig, for three grasps running.
        grasp_index = measured.get("grasp_index", 0)
        print(f"[result]   grasp {grasp_index + 1}: hover {measured['hover_height_mm']}mm, view depth "
              f"{measured['view_depth_mm']}mm, axis predicted at {measured['axis_px_predicted_first']}px")
        fits = {}
        for landmark in LANDMARKS:
            analysis = analyse_landmark(measured["samples"], landmark, measured["intrinsics"],
                                        measured["cap_pose"], measured["view_depth_mm"])
            fits[landmark] = analysis
            if "error" in analysis:
                print(f"[result]   {landmark:7s}: FIT SKIPPED -- {analysis['error']}")
                continue
            print(f"[result]   {landmark:7s}: projection bias {analysis['axis_error_mag_px']:5.1f}px = "
                  f"{analysis['axis_error_mm']:5.2f}mm | axis miss {analysis['miss_mm']:5.2f}mm at "
                  f"{analysis['miss_cap_dir_deg']:+6.1f} deg cap frame | residual "
                  f"{analysis['fit_residual_px']:.2f}px over {analysis['n_angles']} angles")
            for dropped in analysis.get("dropped", []):
                print(f"[result]            dropped roll {dropped['roll_deg']:+6.1f}: {dropped['reason']}")
            print(f"[result]            miss mm tool {analysis['miss_tool_mm']} base "
                  f"{analysis['miss_base_mm']} cap {analysis['miss_cap_mm']} | to centre it, move TCP "
                  f"by base mm {analysis['move_tcp_by_base_mm']} (NOT applied)")
            if analysis["fit_residual_px"] > MAX_FIT_RESIDUAL_PX:
                print(f"[result]            WARNING residual {analysis['fit_residual_px']}px > "
                      f"{MAX_FIT_RESIDUAL_PX}px. Noise-free data leaves 0.35px and a real sweep about "
                      "5px (the sticker's own appearance, see the header), so this much means the "
                      "landmark JUMPED between angles: check the overlays in the output directory. "
                      "The bias/miss split is unreliable until it is under the gate.")
            for name, value, limit in (("projection bias", analysis["axis_error_mm"], PLAUSIBLE_AXIS_ERROR_MM),
                                       ("miss", analysis["miss_mm"], PLAUSIBLE_MISS_MM)):
                if value > limit:
                    print(f"[result]            WARNING {name}={value}mm > {limit}mm -- this is a "
                          "placement/setup fault rather than a bias worth reporting as one")
        sticker = fits.get("sticker", {})
        rim = fits.get("rim", {})
        if "miss_cap_mm" in sticker and "miss_cap_mm" in rim:
            gap = float(np.linalg.norm(np.asarray(sticker["miss_cap_mm"]) - np.asarray(rim["miss_cap_mm"])))
            print(f"[result]   sticker vs rim-centre discrepancy {gap:.2f}mm -- the sticker is that far "
                  "from the geometric cap centre; quote the RIM numbers for DEFAULT_TCP_RIGHT_TO_BOTTLE")
        measured["fits"] = fits


def _flat(results: list[dict], landmark: str, key: str) -> np.ndarray | None:
    vals = []
    for r in results:
        fit = r.get("fits", {}).get(landmark, {})
        if "error" in fit or key not in fit:
            continue
        vals.append(fit[key])
    return None if not vals else np.asarray(vals, dtype=float)


def aggregate(results: list[dict]) -> dict:
    """Mean and spread across poses AND across repeats of the same pose, which is the whole point.

    One number is not a bias; a repeating number is. But a number that repeats across poses while
    scattering across repeats of one pose is a CONSTANT grasp-transform error riding on a bottle that
    settles differently each time -- which is precisely the "sometimes it is off-centre" symptom. The
    within-pose spread over repeats is the re-grasp noise; the mean across everything is the bias.
    """
    out: dict = {}
    print("\n===== aggregate =====")
    pose_ids = [r["pose_index"] for r in results]
    for landmark in LANDMARKS:
        bias = _flat(results, landmark, "axis_error_px")
        miss = _flat(results, landmark, "miss_cap_mm")
        mag = _flat(results, landmark, "miss_mm")
        if miss is None:
            print(f"[bias] {landmark}: no fits")
            continue
        n = len(miss)
        mean, std = miss.mean(axis=0), (miss.std(axis=0, ddof=1) if n > 1 else np.zeros(2))
        print(f"[bias] {landmark:7s} over {n} measurements: miss mean {np.round(mean, 2)}mm |mean| "
              f"{np.linalg.norm(mean):.2f}mm  spread {np.round(std, 2)}  s.e. "
              f"{np.round(std / np.sqrt(n), 2)}")
        bias_spread = bias.std(axis=0, ddof=1) if bias is not None and len(bias) > 1 else np.zeros(2)
        if bias is not None:
            print(f"[bias] {landmark:7s} projection bias mean {np.round(bias.mean(axis=0), 2)}px  "
                  f"spread {np.round(bias_spread, 2)}")
        if mag is not None:
            sem = float(mag.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            print(f"[bias] {landmark:7s} miss magnitude mean {mag.mean():.2f}mm +/- {sem:.2f} s.e.")
        out[landmark] = {
            "miss_mean": [round(float(v), 3) for v in mean],
            "miss_std": [round(float(v), 3) for v in std],
            "bias_mean_px": [round(float(v), 3) for v in bias.mean(axis=0)] if bias is not None else None,
            "n": n,
        }

    counts = {p: pose_ids.count(p) for p in sorted(set(pose_ids))}
    repeated = [p for p, c in counts.items() if c > 1]
    if repeated:
        print(f"[bias] poses measured more than once: {repeated} -- the spread WITHIN a pose is the "
              "hand re-grasp, the mean ACROSS poses is the bias")
        for landmark in LANDMARKS:
            per_pose = []
            for p in repeated:
                vals = [r for r in results if r["pose_index"] == p]
                m = _flat(vals, landmark, "miss_cap_mm")
                if m is not None and len(m) > 1:
                    per_pose.append((p, float(np.linalg.norm(m.std(axis=0, ddof=1))),
                                     float(np.linalg.norm(m.mean(axis=0)))))
            for p, jitter, centre in per_pose:
                verdict = ("grasp REPEATABILITY dominates: the same pose moves this much between "
                           "re-grasps, so the marker cannot be fixed by editing the transform alone"
                           if jitter > centre else
                           "the offset is stable across re-grasps: it is a constant error, and this "
                           "transform translation is the thing to correct")
                print(f"[bias]   pose {p + 1} ({landmark}): re-grasp jitter {jitter:.2f}mm vs mean "
                      f"offset {centre:.2f}mm -> {verdict}")
                out.setdefault("repeats", {})[f"{landmark}_pose{p + 1}"] = {"jitter_mm": round(jitter, 2),
                                                                            "mean_offset_mm": round(centre, 2)}
    else:
        print("[bias] every pose measured once, so pose-to-pose scatter mixes the transform error "
              "with the hand re-grasp. Run --grasps 3 --pose-index 0 to separate them.")
    return out


def describe_mount() -> None:
    """What the calibration itself claims, because it sets the expectation the question turns on."""
    rotation = tcp_left_to_camera[:3, :3]
    lateral = float(np.linalg.norm(tcp_left_to_camera[:2, 3])) * 1000.0
    optical_axis = rotation @ np.array([0.0, 0.0, 1.0])
    tilt_deg = float(np.degrees(np.arccos(np.clip(optical_axis[2], -1.0, 1.0))))
    print(f"[mount] per tcp_left_to_camera the camera sits {lateral:.1f}mm off the gripper axis and "
          f"tilts {tilt_deg:.1f} deg from it, so the hover pose should put the cap centre ~140px off "
          "the principal point -- only a DISAGREEMENT with that prediction is a bias")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=list(SPLIT_SEED_OFFSETS),
                        help="which seeded pose stream to walk (test = the poses eval_bottle draws)")
    parser.add_argument("--n-poses", type=int, default=None,
                        help="how many poses; default is the whole selected split's SPLIT_N_POSES")
    parser.add_argument("--pose-index", type=int, action="append", default=None,
                        help="measure only this pose slot (repeatable). Use with --grasps to test "
                             "re-grasp repeatability on one pose")
    parser.add_argument("--grasps", type=int, default=1,
                        help="times to release, re-stand and re-grasp EACH pose and re-measure -- the "
                             "within-pose spread of those repeats is the hand, the mean is the bias")
    parser.add_argument("--roll-deg", default=",".join(str(a) for a in DEFAULT_ROLL_DEG),
                        help="comma-separated wrist roll angles; a set symmetric about 0 keeps the "
                             "fit's columns orthogonal, which is what separates bias from miss")
    parser.add_argument("--hover-height", type=float, default=HOVER_HEIGHT_METERS,
                        help="standoff above the cap; default is the collector's HOVER_HEIGHT_METERS")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/hover_center_bias"),
                        help="JSON report + per-angle overlays (outputs/ is gitignored)")
    args = parser.parse_args()

    roll_deg_list = [float(a) for a in args.roll_deg.split(",") if a.strip()]
    n_poses = args.n_poses or SPLIT_N_POSES[args.split]

    describe_mount()
    schunk = make_gripper(GRIPPER_PORT_RIGHT)
    env = BottleStation(schunk, with_instrumentation=False, with_spectogram_model=False,
                        with_spectogram=False, use_internal_ft=True)
    intrinsics = env.get_camera_intrinsics()
    shape = to_bgr(env.get_observations()["wrist_image_original"]).shape[:2]
    if shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise RuntimeError(
            f"wrist stream is {shape[1]}x{shape[0]}, expected {FRAME_WIDTH}x{FRAME_HEIGHT} -- the "
            "px<->mm scale and the search windows are all 720p assumptions"
        )
    print(f"[camera] principal point ({intrinsics[0, 2]:.1f}, {intrinsics[1, 2]:.1f}) vs the frame "
          f"centre ({FRAME_WIDTH / 2:.1f}, {FRAME_HEIGHT / 2:.1f}) -- the fit is referenced to the "
          "principal point; their gap is itself a centre bias for anything that assumed they coincide")
    print(f"[anchor] at {args.hover_height:.3f}m hover the model's cap centre must land at "
          f"{_pt(predicted_cap_pixel(intrinsics, args.hover_height))} in the wrist image for EVERY "
          "bottle pose, so that is the search disc all run and one click covers the run")

    rng = np.random.default_rng(RANDOM_SEED + SPLIT_SEED_OFFSETS[args.split])
    print(f"[poses] generating {n_poses} reachable poses from the '{args.split}' stream "
          f"(seed {RANDOM_SEED + SPLIT_SEED_OFFSETS[args.split]}, "
          f"blacklist {POSE_BLACKLIST[args.split]})...")
    bottle_poses = generate_reachable_bottle_poses(
        n_poses, env.robot, env.robot_right, rng, blacklist=POSE_BLACKLIST[args.split]
    )
    if not bottle_poses:
        raise RuntimeError("no reachable poses generated -- check both arms are powered and reachable")
    if len(bottle_poses) < n_poses:
        print(f"[poses] WARNING only {len(bottle_poses)}/{n_poses} generated -- using what passed")
    slots = args.pose_index if args.pose_index else list(range(len(bottle_poses)))

    frame_dir = args.out_dir / "frames"
    live = LiveView()
    run = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "split": args.split, "seed": int(RANDOM_SEED + SPLIT_SEED_OFFSETS[args.split]),
        "blacklist": POSE_BLACKLIST[args.split], "roll_deg": roll_deg_list,
        "hover_height_m": args.hover_height, "grasps": args.grasps,
        "intrinsics": intrinsics.tolist(),
        "frame_center": [FRAME_WIDTH / 2, FRAME_HEIGHT / 2],
        "principal_point": [float(intrinsics[0, 2]), float(intrinsics[1, 2])],
        "grasp_transform": DEFAULT_TCP_RIGHT_TO_BOTTLE.tolist(),
        "tcp_left_to_camera": tcp_left_to_camera.tolist(),
        "rim_radius_prior": [rim_detector.RADIUS_K, rim_detector.RADIUS_C],
        "measurements": [],
    }
    model = StickerModel()

    try:
        for slot in slots:
            tcp_right_pose, planned_cap_pose = bottle_poses[slot]
            print(f"\n===== pose {slot + 1}/{len(bottle_poses)} "
                  f"({args.split} stream slot {slot}) =====")
            for grasp_index in range(args.grasps):
                print(f"--- grasp {grasp_index + 1}/{args.grasps} ---")
                input("Press Enter to send ur_left home (Ctrl+C to abort)...")
                env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
                input("Press Enter to open ur_right's gripper and bring it to the neutral pose...")
                _move_gripper_to(env, GRIPPER_OPEN_WIDTH_M)
                env.move_right_to_joint_configuration(RIGHT_NEUTRAL_JOINTS, joint_speed=RIGHT_JOINT_SPEED)
                input("Stand the bottle between the open fingers, then press Enter -- ur_right moves "
                      "onto it and closes (Ctrl+C to abort)...")
                env.move_right_to_tcp_pose(tcp_right_pose, joint_speed=RIGHT_JOINT_SPEED)
                _move_gripper_to(env, GRIPPER_GRASP_WIDTH_M)

                # Read the cap pose back the way the collector does -- from ur_right's live TCP rather
                # than the planned pose: what matters is where the gripper actually holds the bottle.
                cap_pose = env.get_bottle_cap_pose()
                hover_pose = compute_hover_pose_above_bottle(cap_pose, args.hover_height)
                print(f"[move] ur_left to the hover pose at {np.round(hover_pose[:3, 3], 4)}")
                env.move_robot_to_tcp_pose(hover_pose, joint_speed=LEFT_TRANSIT_JOINT_SPEED)

                fixed = predicted_cap_pixel(intrinsics, args.hover_height)
                projected = project_point_to_pixel(cap_pose[:3, 3], intrinsics,
                                                   env.get_robot_pose_se3() @ tcp_left_to_camera)
                if projected is not None and float(np.hypot(*(np.asarray(projected) - np.asarray(fixed)))) > 2.0:
                    print(f"[anchor] this pose projects the cap centre to {_pt(projected)} but the mount "
                          f"predicts {_pt(fixed)} -- the arm is not where it was told, so the search "
                          "window follows the projection")
                anchor = tuple(projected) if projected is not None else fixed
                # Once per pose, once per run at most: the sticker's absolute colour moves with the
                # lighting that each new bottle orientation brings, and the pose-independent anchor is
                # what lets the previous window be re-tried in place of another click.
                model = lock_sticker_for_pose(env, live, model, anchor)

                tag = f"pose{slot + 1:02d}g{grasp_index + 1}"
                measured, model = measure_pose(
                    env, intrinsics, cap_pose, hover_pose, model, roll_deg_list, args.hover_height,
                    live, frame_dir, tag)
                measured.update({
                    "pose_index": slot,
                    "grasp_index": grasp_index,
                    "cap_center_base": [round(float(v), 5) for v in cap_pose[:3, 3]],
                    "planned_cap_center_base": [round(float(v), 5) for v in planned_cap_pose[:3, 3]],
                    "hover_pose": hover_pose.tolist(),
                    "cap_pose": cap_pose.tolist(),
                    "intrinsics": intrinsics.tolist(),
                })
                report_measurement([measured])
                run["measurements"].append(measured)

                print("[move] ur_left home")
                env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
                if grasp_index + 1 < args.grasps:
                    input("Leave the bottle where it stands -- press Enter to release and re-grasp the "
                          "SAME pose (Ctrl+C to stop)...")
            if slot != slots[-1]:
                input("Re-stand the bottle for the next pose, then press Enter...")
    finally:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run["aggregate"] = aggregate(run["measurements"])
        out_file = args.out_dir / f"hover_center_bias_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_file.write_text(json.dumps(run, indent=2))
        print(f"[out] wrote {out_file} ({len(run['measurements'])} measurements, overlays in {frame_dir})")
        try:
            env.robot.move_to_joint_configuration(LEFT_HOME_JOINTS, joint_speed=LEFT_TRANSIT_JOINT_SPEED).wait()
        except Exception as exc:  # cleanup must never hide the failure that triggered it
            print(f"[cleanup] could not retreat ur_left home: {exc}")
        live.close()
        env.close()


if __name__ == "__main__":
    main()
