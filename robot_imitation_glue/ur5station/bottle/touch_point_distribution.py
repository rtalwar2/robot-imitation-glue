"""Where on the cap did the demonstrations touch? Runs the learned touch-point detector on the first frame of
every episode of the raw expert dataset and plots the distribution relative to the cap centre.

    python -m robot_imitation_glue.ur5station.bottle.touch_point_distribution
    python -m robot_imitation_glue.ur5station.bottle.touch_point_distribution --raw-root <dataset> --out <dir>

Frame 0 of an episode is the hover view the collector planned from (the arm is at the hover pose when
recording starts). The collector did not log the touch pixel it used, so this re-runs the same detector on
that frame: for frames with score >= MIN_SCORE it is exactly the point the collector used; below it the
operator verified or corrected the point at collection time, so those may differ (drawn hollow).

The cap centre is the orange centre sticker, found by colour per frame. Angles are measured about it in the
image, counter-clockwise from the image's +x (right), image y pointing up -- the camera sits rigidly on the
gripper and the hover pose is defined from the cap pose, so a given image angle is a fixed direction on the cap.

Writes to --out (default outputs/touch_point_distribution/):
  touch_points.csv  per episode: touch pixel, score, cap centre, offset, radius, angle
  montage.jpg       every episode's cap crop with its touch point
  overlay.png       all touch points on one reference frame, shifted to its cap centre
  distribution.png  angle histogram (polar) and radius histogram
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "bottle_experiment"))
from touch_point_keypoints.detector import MIN_SCORE, TouchPointKeypointDetector  # noqa: E402

DEFAULT_RAW_ROOT = _REPO_ROOT / "bottle_experiment/datasets/bottle_experiment/expert/bottle_opening_train"
CROP_HALF = 220  # px around the cap centre in the montage
# Orange centre sticker (OpenCV HSV, hue 0-180); the cap's red rim sits at hue ~0-5 and is thin, the
# sticker is a filled disc of ~1600 px at hover.
STICKER_HSV_LOW, STICKER_HSV_HIGH = (3, 140, 120), (16, 255, 255)
STICKER_AREA_PX = (500, 6000)


def find_sticker(image_bgr):
    """Centre (x, y) of the orange cap sticker, or None: the roundest, most filled orange blob."""
    mask = cv2.inRange(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV), STICKER_HSV_LOW, STICKER_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    best, best_fill = None, 0.7  # minimum fill of the bounding circle
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        if not STICKER_AREA_PX[0] <= area <= STICKER_AREA_PX[1] or not 0.7 <= w / h <= 1.4:
            continue
        fill = area / (np.pi * (max(w, h) / 2) ** 2)
        if fill > best_fill:
            best, best_fill = tuple(centroids[i]), fill
    return best


def first_frames(raw_root: Path):
    """{episode: BGR uint8 frame 0} from wrist_image_original."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    table = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "frame_index", "index"]) for f in sorted(raw_root.glob("data/*/*.parquet"))
    )
    starts = table[table.frame_index == 0].sort_values("episode_index")
    dataset = LeRobotDataset(repo_id=None, root=str(raw_root), video_backend="pyav")
    for episode, index in zip(starts.episode_index, starts["index"]):
        rgb = (dataset[int(index)]["wrist_image_original"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        yield int(episode), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "outputs" / "touch_point_distribution")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    detector = TouchPointKeypointDetector()
    rows, crops, reference = [], [], None
    for episode, frame in first_frames(args.raw_root):
        x, y, score = detector.predict(frame)
        centre = find_sticker(frame)
        row = {"episode": episode, "touch_x": x, "touch_y": y, "score": score, "model_only": score >= MIN_SCORE}
        if centre is not None:
            dx, dy = x - centre[0], centre[1] - y  # image y flipped so angles read counter-clockwise
            row |= {"cap_x": centre[0], "cap_y": centre[1], "dx": dx, "dy": dy,
                    "radius_px": float(np.hypot(dx, dy)), "angle_deg": float(np.degrees(np.arctan2(dy, dx)) % 360)}
            if reference is None:
                reference = (frame.copy(), centre)
        else:
            print(f"[warn] episode {episode}: cap sticker not found -- excluded from the cap-relative plots")
        rows.append(row)
        print(f"[ep {episode:3d}] touch ({x:6.1f}, {y:6.1f}) score {score:.2f}"
              + (f"  r {row['radius_px']:5.1f}px  angle {row['angle_deg']:5.1f} deg" if centre else ""))

        # montage tile: crop around the cap centre (or the touch point), touch point marked
        cx, cy = (int(round(c)) for c in (centre or (x, y)))
        tile = cv2.copyMakeBorder(frame, CROP_HALF, CROP_HALF, CROP_HALF, CROP_HALF, cv2.BORDER_CONSTANT)
        tile = tile[cy : cy + 2 * CROP_HALF, cx : cx + 2 * CROP_HALF].copy()
        point = (int(round(x - cx + CROP_HALF)), int(round(y - cy + CROP_HALF)))
        cv2.circle(tile, point, 9, (0, 255, 0), -1 if score >= MIN_SCORE else 2)
        if centre:
            cv2.drawMarker(tile, (CROP_HALF, CROP_HALF), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(tile, f"ep {episode}  {score:.2f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        crops.append(cv2.resize(tile, (220, 220)))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "touch_points.csv", index=False)

    columns = 10
    blank = np.zeros_like(crops[0])
    crops += [blank] * (-len(crops) % columns)
    montage = np.vstack([np.hstack(crops[i : i + columns]) for i in range(0, len(crops), columns)])
    cv2.imwrite(str(args.out / "montage.jpg"), montage)

    located = df.dropna(subset=["angle_deg"])
    if reference is not None and len(located):
        image, (rx, ry) = reference
        for _, r in located.iterrows():
            p = (int(round(rx + r.dx)), int(round(ry - r.dy)))
            cv2.circle(image, p, 6, (0, 255, 0), -1 if r.model_only else 2)
        cv2.drawMarker(image, (int(round(rx)), int(round(ry))), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.imwrite(str(args.out / "overlay.png"), image)

        fig = plt.figure(figsize=(11, 5))
        polar = fig.add_subplot(1, 2, 1, projection="polar")
        bins = np.radians(np.arange(0, 361, 15))
        counts, _ = np.histogram(np.radians(located.angle_deg), bins=bins)
        polar.bar(bins[:-1], counts, width=np.radians(15), align="edge", edgecolor="k")
        polar.set_title(f"touch-point angle about the cap centre (n={len(located)})\n0 deg = image right, counter-clockwise")
        radius = fig.add_subplot(1, 2, 2)
        radius.hist(located.radius_px, bins=20, edgecolor="k")
        radius.set_xlabel("distance from cap centre (px)")
        radius.set_ylabel("episodes")
        radius.set_title("touch-point radius")
        fig.tight_layout()
        fig.savefig(args.out / "distribution.png", dpi=120)

        print(f"\n[summary] {len(df)} episodes, cap centre found in {len(located)}; "
              f"{(~df.model_only).sum()} below MIN_SCORE {MIN_SCORE} (collector may have used a corrected point)")
        print(f"[summary] angle: mean direction {np.degrees(np.angle(np.exp(1j * np.radians(located.angle_deg)).mean())) % 360:.0f} deg, "
              f"range {located.angle_deg.min():.0f}-{located.angle_deg.max():.0f} deg; "
              f"radius median {located.radius_px.median():.0f} px (p5 {located.radius_px.quantile(.05):.0f}, "
              f"p95 {located.radius_px.quantile(.95):.0f})")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
