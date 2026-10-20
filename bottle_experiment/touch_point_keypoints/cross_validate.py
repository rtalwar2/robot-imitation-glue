"""Out-of-fold predictions for every click: how good the detector is, and which clicks look wrong.

Each fold's model (datasets/touch_point_keypoints/models/fold{k}.ckpt, trained by train.sh) predicts the
samples it never saw. Per sample this prints the distance between the prediction and the click, in
ORIGINAL 1280x720 pixels, next to the classical touch_point_detector's distance for comparison. The
largest disagreements are tiled into one image (green = click, red = model) for a person to judge:
a sample goes into outliers.json only when the click is visibly wrong, not merely because the model
disagrees -- the model is wrong on some of them too.

    ~/miniconda3/envs/keypoint-detection/bin/python bottle_experiment/touch_point_keypoints/cross_validate.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import K_FOLDS, OUT_DIR, SOURCE_DIR  # noqa: E402
from detector import TouchPointKeypointDetector  # noqa: E402

N_REVIEW = 24
REVIEW_IMAGE = OUT_DIR / "review_largest_disagreements.jpg"


def main():
    rows = []
    for k in range(K_FOLDS):
        detector = TouchPointKeypointDetector(OUT_DIR / "models" / f"fold{k}.ckpt")
        for image in json.loads((OUT_DIR / f"fold{k}_val.json").read_text())["images"]:
            index = image["id"]
            meta = json.loads((SOURCE_DIR / f"sample_{index:04d}.json").read_text())
            frame = cv2.imread(str(SOURCE_DIR / f"sample_{index:04d}_raw.jpg"))
            x, y, score = detector.predict(frame)
            click = np.asarray(meta["click_xy"], float)
            classical = meta.get("detected_xy")
            rows.append({"index": index, "fold": k, "pred": [x, y], "score": score, "click": click.tolist(),
                         "error_px": float(np.hypot(x - click[0], y - click[1])),
                         "classical_error_px": None if classical is None else float(np.hypot(*(np.asarray(classical) - click)))})

    errors = np.array([r["error_px"] for r in rows])
    classical = np.array([r["classical_error_px"] for r in rows if r["classical_error_px"] is not None])
    print(f"out-of-fold error over {len(rows)} clicks (1280x720 px): median {np.median(errors):.1f}, "
          f"p90 {np.percentile(errors, 90):.1f}, <=10px {np.mean(errors <= 10):.0%}, <=20px {np.mean(errors <= 20):.0%}")
    print(f"classical detector on the same clicks ({len(classical)} with a detection): median "
          f"{np.median(classical):.1f}, p90 {np.percentile(classical, 90):.1f}, <=10px {np.mean(classical <= 10):.0%}")
    rows.sort(key=lambda r: -r["error_px"])
    print("\nlargest disagreements:")
    for r in rows[:N_REVIEW]:
        print(f"  sample {r['index']:4d}  error {r['error_px']:6.1f}px  score {r['score']:.2f}  "
              f"click {r['click']}  model {np.round(r['pred']).astype(int).tolist()}")
    (OUT_DIR / "cross_validation.json").write_text(json.dumps(rows, indent=1))

    tiles = []
    for r in rows[:N_REVIEW]:
        frame = cv2.imread(str(SOURCE_DIR / f"sample_{r['index']:04d}_raw.jpg"))
        cv2.drawMarker(frame, tuple(int(v) for v in r["click"]), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
        cv2.drawMarker(frame, tuple(int(round(v)) for v in r["pred"]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 3)
        cx, cy = np.mean([r["click"], r["pred"]], axis=0).astype(int)
        cx, cy = int(np.clip(cx, 250, 1030)), int(np.clip(cy, 250, 470))
        tile = cv2.resize(frame[cy - 250:cy + 250, cx - 250:cx + 250], (320, 320))
        cv2.putText(tile, f"{r['index']} {r['error_px']:.0f}px", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(REVIEW_IMAGE), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
    print(f"\nreview tiles: {REVIEW_IMAGE}")


if __name__ == "__main__":
    main()
