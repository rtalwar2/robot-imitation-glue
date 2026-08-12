"""
Detection algorithm for the bottle-cap lid touch point (where the gripper should press to
start the circular opening motion), factored out of lid_touch_point_annotator.py so it can
be imported/benchmarked/reused independently of the interactive annotation tool.
"""

from __future__ import annotations

import cv2
import numpy as np

# HSV thresholds for the lid's red rim -- tune against captured frames if lighting changes.
RED_HSV_RANGES = [((0, 100, 50), (10, 255, 255)), ((170, 100, 50), (180, 255, 255))]

# apparent_rim_radius_px = RADIUS_K / (hover_height_m + RADIUS_C), fit from labeled samples
# at heights 0.05/0.1/0.15m -- used as a prior so the rim circle fit isn't thrown off by
# background clutter that happens to also be red. Re-fit if the camera/lens/rim size changes.
RADIUS_K, RADIUS_C = 38.2653, 0.1335


def _predict_rim_radius(hover_height: float) -> float:
    return RADIUS_K / (hover_height + RADIUS_C)


def _red_components(
    bgr_image: np.ndarray,
    min_component_area: int = 150,
    roi_margin_frac: float = 0.12,
    max_solidity: float = 0.75,
) -> list[np.ndarray]:
    """Pixel coords of the rim's red pixels, as a list of per-component arrays, restricted
    to a central ROI and filtered to sizeable, ring-shaped components -- the hover routine
    centers the lid, so the ROI excludes most background clutter that would otherwise
    corrupt the circle fit below.

    (An earlier version dropped this crop in favor of seeding the fit at the image center,
    reasoning that a frame-relative crop risks clipping real rim pixels near the edge. That
    was wrong in practice -- tested against the 11 labeled samples, it broke 5 of them
    outright and badly mispredicted 2 more, because without the crop, clutter dominates the
    point cloud badly enough that seeding near it (rather than amid already-clutter-free
    points) derails the fit before the radius filter gets a chance to help. Restored the
    crop; the edge-clipping concern is real in theory but unobserved in practice so far.)

    `max_solidity` (area / convex-hull-area) rejects compact/filled red blobs -- e.g. a red
    logo on another object visible in frame -- that land inside the ROI and are big enough
    to pass the area filter. Genuine rim arcs are thin curves: even a short, fragmented arc
    piece measured across 25 labeled samples topped out at 0.61 solidity, while stray solid
    blobs measured 0.89-0.94 -- a clean gap, so 0.75 is a safe cutoff between them.
    """
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv_image, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    h, w = mask.shape
    mx, my = int(w * roi_margin_frac), int(h * roi_margin_frac)
    roi = np.zeros_like(mask)
    roi[my:h - my, mx:w - mx] = 255
    mask = cv2.bitwise_and(mask, roi)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_component_area:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 1.0
        if solidity > max_solidity:
            continue
        component_mask = np.zeros_like(mask)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        ys, xs = np.where(component_mask > 0)
        components.append(np.column_stack([xs, ys]).astype(np.float64))
    return components


def _fit_circle_lsq(points: np.ndarray) -> tuple[np.ndarray, float]:
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    return np.array([cx, cy]), float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))


def _fit_rim_circle(points: np.ndarray, expected_radius: float, tol: float = 0.35):
    """Robust circle fit: repeatedly keep only points within `tol` of the expected radius
    (from `_predict_rim_radius`) and refit, so a few points don't derail the whole fit.
    Seeded at the (already ROI-restricted, so already mostly clutter-free) data mean.

    Returns (center, radius, residual_px) -- residual is the mean absolute distance of the
    final inlier set from the fitted circle, i.e. how well the fit actually explains the
    data. Returns residual=inf if it never found enough inliers to trust.
    """
    center, r = points.mean(axis=0), expected_radius
    inliers = None
    for _ in range(5):
        dists = np.linalg.norm(points - center, axis=1)
        candidate_inliers = points[np.abs(dists - expected_radius) < tol * expected_radius]
        if len(candidate_inliers) < 20:
            break
        new_center, new_r = _fit_circle_lsq(candidate_inliers)
        if abs(new_r - expected_radius) >= tol * expected_radius:
            break
        center, r, inliers = new_center, new_r, candidate_inliers
    if inliers is None:
        return center, r, float("inf")
    residual = float(np.mean(np.abs(np.linalg.norm(inliers - center, axis=1) - r)))
    return center, r, residual


def _angular_span_deg(points: np.ndarray, center: np.ndarray) -> float:
    """How much of the circle (in degrees) this component's points span around `center` --
    360 minus the single largest internal gap, so a component broken across the 0/360
    wraparound still reports its true span."""
    angles = np.degrees(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])) % 360
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 360]]))
    return 360 - gaps.max()


def _touch_point_from_gap(points: np.ndarray, center: np.ndarray, r: float, min_gap_deg: float):
    """The point where `points`' angular coverage "resumes" (clockwise) after its largest
    gap, or None if there's no gap clearly bigger than ordinary point-to-point spacing."""
    angles = np.degrees(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])) % 360
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 360]]))
    gap_idx = np.argmax(gaps)
    if gaps[gap_idx] < min_gap_deg:
        return None
    touch_angle = np.radians(sorted_angles[(gap_idx + 1) % len(sorted_angles)])
    return int(round(center[0] + r * np.cos(touch_angle))), int(round(center[1] + r * np.sin(touch_angle)))


def detect_touch_point(
    bgr_image: np.ndarray,
    hover_height: float,
    min_gap_deg: float = 15.0,
    max_fit_residual_px: float = 8.0,
) -> tuple[int, int] | None:
    """
    Touch point = the point where the visible red/black rim boundary "resumes" (going
    clockwise in image coordinates) after its largest angular gap. Validated against 25
    hand-labeled samples across several bottle placements and all 3 hover heights: this is
    where the ground-truth click landed, within a few px, in the large majority of cases --
    no need to detect the (small, low-contrast) printed arrow at all.

    `hover_height` is ur_left's current hover height in metres above the bottle cap --
    used to predict the rim's apparent pixel radius (see `_predict_rim_radius`), which
    makes the rim circle fit robust to background clutter.

    At some cap tilts, a second, thinner red region becomes visible (a viewing-angle
    artifact, not the main rim edge) and fragments the rim into 3+ angular segments
    instead of 2. When that happens, fitting the circle to *all* red points together can
    become poor enough to fail its own quality check (`max_fit_residual_px`) -- in that
    case, falls back to isolating just the single component with the largest angular span
    (the actual rim, as opposed to the smaller secondary region) and repeats the fit using
    only that. Still returns None (rather than a confident-looking wrong answer) if even
    the fallback's fit is poor, or if neither has a gap clearly bigger than ordinary
    point-to-point spacing (`min_gap_deg`).
    """
    components = _red_components(bgr_image)
    if not components:
        return None
    all_points = np.concatenate(components, axis=0)
    if len(all_points) < 30:
        return None
    expected_radius = _predict_rim_radius(hover_height)

    center, r, residual = _fit_rim_circle(all_points, expected_radius)
    if residual <= max_fit_residual_px:
        touch_point = _touch_point_from_gap(all_points, center, r, min_gap_deg)
        if touch_point is not None:
            return touch_point

    # fallback: isolate the component with the largest angular span around the (possibly
    # still-rough) center estimate above, and redo the fit + gap search using only that.
    main_component = max(
        (comp for comp in components if len(comp) >= 20),
        key=lambda comp: _angular_span_deg(comp, center),
        default=None,
    )
    if main_component is None:
        return None
    center, r, residual = _fit_rim_circle(main_component, expected_radius)
    if residual > max_fit_residual_px:
        return None
    return _touch_point_from_gap(main_component, center, r, min_gap_deg)
