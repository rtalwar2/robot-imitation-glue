"""Drop repeated and still frames from a record_cap_spin video, so only the spinning remains to click through.

Two kinds of frame go:
  - repeats: the recorder loop runs faster than the camera's 30 fps, so most frames are an exact copy of the
    previous one (first recording: 2151 of 2838). They differ from their predecessor by < DUPLICATE_DIFF grey
    levels inside the cap window, while a genuinely new frame differs by >= 1.8 even in a still scene;
  - still frames: before and after the spin. A new frame is kept only if the cap window changed by more than
    STILL_DIFF against the new frame MOTION_LAG frames earlier or later (a still scene measured 1.2, a spin 4-8).

Writes video_trimmed.mp4 and trimmed_frames.json (the kept frame indices into video.mp4) next to the
recording; click_cap_spin.py uses the trimmed video when it exists.

    python -m robot_imitation_glue.ur5station.bottle.trim_cap_spin [outputs/cap_spin/<time>]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

OUT_ROOT = Path("outputs/cap_spin")
WINDOW_RADIUS_PX = 220  # half-size of the window around the modelled cap centre that motion is measured in
DUPLICATE_DIFF = 1.0
STILL_DIFF = 2.0
MOTION_LAG = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", type=Path, default=None)
    args = parser.parse_args()
    directory = args.recording or sorted(OUT_ROOT.iterdir())[-1]
    meta = json.loads((directory / "meta.json").read_text())
    cx, cy = (int(round(v)) for v in meta["modelled_center_px"])

    video = cv2.VideoCapture(str(directory / "video.mp4"))
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)

    def window(frame):
        crop = frame[max(cy - WINDOW_RADIUS_PX, 0):cy + WINDOW_RADIUS_PX, max(cx - WINDOW_RADIUS_PX, 0):cx + WINDOW_RADIUS_PX]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Repeats on the RAW window: sensor noise is what tells a new still frame from a copy, and a blur would
    # erase exactly that. Motion on a BLURRED window, where that noise is the thing to suppress.
    raw = [window(f) for f in frames]
    new = [0] + [i for i in range(1, len(frames)) if np.abs(raw[i] - raw[i - 1]).mean() >= DUPLICATE_DIFF]
    windows = {i: cv2.GaussianBlur(raw[i], (5, 5), 0) for i in new}
    motion = [max(np.abs(windows[new[j]] - windows[new[k]]).mean()
                  for k in (j - MOTION_LAG, j + MOTION_LAG) if 0 <= k < len(new)) for j in range(len(new))]
    kept = [new[j] for j in range(len(new)) if motion[j] > STILL_DIFF]

    writer = cv2.VideoWriter(str(directory / "video_trimmed.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (frames[0].shape[1], frames[0].shape[0]))
    for i in kept:
        writer.write(frames[i])
    writer.release()
    times = meta.get("frame_times", [])
    (directory / "trimmed_frames.json").write_text(json.dumps({"kept_indices": kept, "kept_times": [times[i] for i in kept] if times else []}))
    span = f", {times[kept[0]] - times[0]:.1f}-{times[kept[-1]] - times[0]:.1f} s into the recording" if times and kept else ""
    print(f"{len(frames)} frames -> {len(new)} new -> {len(kept)} moving{span}; wrote {directory / 'video_trimmed.mp4'}")


if __name__ == "__main__":
    main()
