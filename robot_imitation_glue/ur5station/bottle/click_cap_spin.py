"""Click one point on the cap through a record_cap_spin video; the circle it traces gives the cap's true centre.

Pick a feature on the cap that is easy to find again (an edge of the sticker, a letter of the print, a notch
of the rim) and click it on frames spread over the spin. Its positions lie on a circle around the rotation
axis; because the video is head-on, the circle's image centre is where the axis -- the true cap centre --
projects. That pixel is back-projected onto the modelled cap plane and expressed in ur_right's TCP frame,
which gives DEFAULT_TCP_RIGHT_TO_BOTTLE's in-plane x/z (y, the height, stays as modelled).

    python -m robot_imitation_glue.ur5station.bottle.click_cap_spin                 # newest recording
    python -m robot_imitation_glue.ur5station.bottle.click_cap_spin outputs/cap_spin/20261009_143000

Run trim_cap_spin.py first to drop repeated and still frames; the trimmed video is used when present.

Keys: left click = mark the feature on this frame (click again to move it), space / n = forward STEP
frames, b = back STEP frames, u = remove this frame's mark, Enter or q = done. Needs at least 3 marks;
8+ spread over most of a turn is what makes the centre trustworthy.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STEP = 5
OUT_ROOT = Path("outputs/cap_spin")
WINDOW = "click the same cap feature on each frame (space = next, b = back, u = undo, Enter = done)"


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic least-squares circle: centre, radius, rms radial residual."""
    a = np.column_stack([2 * points, np.ones(len(points))])
    b = (points ** 2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    radius = np.sqrt(c + cx ** 2 + cy ** 2)
    residual = np.sqrt(np.mean((np.linalg.norm(points - [cx, cy], axis=1) - radius) ** 2))
    return np.array([cx, cy]), float(radius), float(residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", type=Path, default=None)
    args = parser.parse_args()
    directory = args.recording or sorted(OUT_ROOT.iterdir())[-1]
    meta = json.loads((directory / "meta.json").read_text())

    # trim_cap_spin.py's output when it exists: the same frames minus repeats and the still start/end.
    trimmed = directory / "video_trimmed.mp4"
    video = cv2.VideoCapture(str(trimmed if trimmed.exists() else directory / "video.mp4"))
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)
    print(f"[load] {len(frames)} frames from {directory}")

    clicks: dict[int, tuple[int, int]] = {}
    current = {"index": 0}
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda event, x, y, *_: clicks.__setitem__(current["index"], (x, y))
                         if event == cv2.EVENT_LBUTTONDOWN else None)
    modelled = tuple(int(round(v)) for v in meta["modelled_center_px"])
    while True:
        i = current["index"]
        shown = frames[i].copy()
        cv2.drawMarker(shown, modelled, (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        for j, point in clicks.items():
            cv2.circle(shown, point, 3 if j != i else 6, (0, 255, 0) if j == i else (0, 160, 0), -1)
        cv2.putText(shown, f"frame {i + 1}/{len(frames)}   marks {len(clicks)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord(" "), ord("n")):
            current["index"] = min(i + STEP, len(frames) - 1)
        elif key == ord("b"):
            current["index"] = max(i - STEP, 0)
        elif key == ord("u"):
            clicks.pop(i, None)
        elif key in (13, ord("q"), 27):
            break
    cv2.destroyAllWindows()
    if len(clicks) < 3:
        raise SystemExit(f"only {len(clicks)} marks -- a circle needs at least 3")

    points = np.array(list(clicks.values()), float)
    center, radius, residual = fit_circle(points)
    angles = np.degrees(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    span = 360 - np.diff(np.sort(np.r_[angles, angles.min() + 360])).max()
    print(f"[circle] centre ({center[0]:.1f}, {center[1]:.1f}) px, radius {radius:.1f} px, rms residual "
          f"{residual:.1f} px, marks cover {span:.0f} deg of the turn")
    print(f"[circle] the model put the cap centre at ({modelled[0]}, {modelled[1]}) px: "
          f"{np.linalg.norm(center - modelled):.1f} px apart")

    # Back-project the circle centre onto the modelled cap plane and express it in ur_right's TCP frame.
    intrinsics = np.array(meta["intrinsics"])
    camera = np.array(meta["tcp_left_base_left"]) @ np.array(meta["tcp_left_to_camera"])
    cap = np.array(meta["cap_pose_base_left"])
    ray = camera[:3, :3] @ (np.linalg.inv(intrinsics) @ np.array([center[0], center[1], 1.0]))
    t = cap[:3, 2] @ (cap[:3, 3] - camera[:3, 3]) / (cap[:3, 2] @ ray)
    point = camera[:3, 3] + t * ray
    tcp_right = np.array(meta["base_left_to_base_right"]) @ np.array(meta["tcp_right_base_right"])
    measured = (np.linalg.inv(tcp_right) @ np.append(point, 1.0))[:3] * 1000
    modelled_mm = np.array(meta["tcp_right_to_bottle"])[:3, 3] * 1000
    print(f"[result] cap centre in ur_right's TCP frame: x {measured[0]:.2f}  y {measured[1]:.2f} (height, as "
          f"modelled)  z {measured[2]:.2f} mm")
    print(f"[result] current DEFAULT_TCP_RIGHT_TO_BOTTLE translation: x {modelled_mm[0]:.2f}  y {modelled_mm[1]:.2f}  "
          f"z {modelled_mm[2]:.2f} mm  -> change x by {measured[0] - modelled_mm[0]:+.2f}, z by {measured[2] - modelled_mm[2]:+.2f} mm")
    (directory / "clicks.json").write_text(json.dumps({"clicks": {str(k): v for k, v in clicks.items()},
                                                       "circle_center_px": center.tolist(), "radius_px": radius,
                                                       "rms_residual_px": residual, "span_deg": float(span),
                                                       "cap_center_tcp_right_mm": measured.tolist()}, indent=1))
    print(f"[out] {directory / 'clicks.json'}")


if __name__ == "__main__":
    main()
