"""Turn the clicked touch points in lid_touch_point_debug/ into a COCO keypoints dataset for tlpss/keypoint-detection.

Every sample_NNNN_raw.jpg with a click_xy becomes one image with one keypoint ("touch_point"). Frames are
resized from 1280x720 to IMAGE_SIZE (both sides divisible by 32, which the U-Net backbones need), and the
click is scaled with them; predictions are scaled back by the same factors (see detector.py).

Writes, under OUT_DIR:
  images/sample_NNNN.jpg
  all.json                    every usable sample
  fold{k}_train.json / fold{k}_val.json   K_FOLDS cross-validation splits, used to find bad clicks
  final_train.json / final_val.json       the split the deployed model is trained on

Samples listed in outliers.json (next to this file) are left out of every split. That file is written
by hand after looking at the frames cross_validate.py flags -- a click is only dropped once a person
has seen that it is wrong, never because a model disagreed with it.

Run with the keypoint-detection env:
    ~/miniconda3/envs/keypoint-detection/bin/python bottle_experiment/touch_point_keypoints/build_dataset.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO / "lid_touch_point_debug"
OUT_DIR = REPO / "datasets" / "touch_point_keypoints"
OUTLIERS_FILE = Path(__file__).with_name("outliers.json")

IMAGE_SIZE = (640, 352)  # (width, height) fed to the model
K_FOLDS = 5
FINAL_VAL_FRACTION = 0.15
SEED = 0


def coco(entries: list[dict]) -> dict:
    return {
        "categories": [{"supercategory": "cap", "id": 1, "name": "cap", "keypoints": ["touch_point"], "skeleton": []}],
        "images": [{"id": e["id"], "file_name": e["file_name"], "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]}
                   for e in entries],
        "annotations": [{"id": e["id"], "image_id": e["id"], "category_id": 1, "num_keypoints": 1,
                         "keypoints": [e["x"], e["y"], 2]} for e in entries],
    }


def main():
    outliers = set(json.loads(OUTLIERS_FILE.read_text())["exclude"]) if OUTLIERS_FILE.exists() else set()
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    entries = []
    for meta_path in sorted(SOURCE_DIR.glob("sample_*.json")):
        index = int(meta_path.stem.split("_")[1])
        meta = json.loads(meta_path.read_text())
        if not meta.get("click_xy") or index in outliers:
            continue
        image = cv2.imread(str(meta_path.with_name(f"{meta_path.stem}_raw.jpg")))
        height, width = image.shape[:2]
        sx, sy = IMAGE_SIZE[0] / width, IMAGE_SIZE[1] / height
        file_name = f"images/{meta_path.stem}.jpg"
        cv2.imwrite(str(OUT_DIR / file_name), cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA))
        # +0.5: a click at integer pixel (u, v) means that pixel's centre; COCO-float keypoints are continuous.
        entries.append({"id": index, "file_name": file_name,
                        "x": (meta["click_xy"][0] + 0.5) * sx, "y": (meta["click_xy"][1] + 0.5) * sy})

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(entries))
    write = lambda name, subset: (OUT_DIR / name).write_text(json.dumps(coco(subset)))  # noqa: E731
    write("all.json", entries)
    for k, val_idx in enumerate(np.array_split(order, K_FOLDS)):
        val = set(val_idx.tolist())
        write(f"fold{k}_train.json", [e for i, e in enumerate(entries) if i not in val])
        write(f"fold{k}_val.json", [e for i, e in enumerate(entries) if i in val])
    n_val = int(round(FINAL_VAL_FRACTION * len(entries)))
    write("final_train.json", [entries[i] for i in order[n_val:]])
    write("final_val.json", [entries[i] for i in order[:n_val]])
    print(f"{len(entries)} samples ({len(outliers)} excluded as outliers) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
