"""Learned touch-point detector: a tlpss/keypoint-detection model applied to a full 1280x720 wrist frame.

    detector = TouchPointKeypointDetector()     # datasets/touch_point_keypoints/models/final.pt
    x, y, score = detector.predict(bgr_image)   # pixel in the ORIGINAL frame, peak heatmap value in [0, 1]
    if score < MIN_SCORE: ask for a click instead

The frame is resized exactly as build_dataset.py resized the training images, the heatmap peak is refined
to sub-pixel with a local centroid, and the result is scaled back.

A `.pt` model is TorchScript (export_torchscript.py) and needs only torch -- that is what the rig env
loads. A `.ckpt` is the raw Lightning checkpoint and needs the keypoint-detection package (its own
conda env); cross_validate.py uses those.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

IMAGE_SIZE = (640, 352)  # (width, height) -- must match build_dataset.IMAGE_SIZE
REFINE_RADIUS = 2  # heatmap pixels around the peak used for the sub-pixel centroid
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "datasets" / "touch_point_keypoints" / "models" / "final.pt"

# Below this peak score, fall back to a click. Chosen on the 2026-10-09 out-of-fold predictions (270
# clicks): 0.4 accepts 91% of frames at a median 4.1 px / p95 8.0 px error, and lets through only 1 of
# the 10 frames the model got more than 20 px wrong (that one by 31 px).
MIN_SCORE = 0.4


def _load(path: Path, device: str) -> torch.nn.Module:
    if path.suffix == ".pt":
        return torch.jit.load(str(path), map_location=device).eval()
    from keypoint_detection.utils.load_checkpoints import load_from_checkpoint

    return load_from_checkpoint(str(path)).eval().to(device)


class TouchPointKeypointDetector:
    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = _load(Path(model), device)

    @torch.no_grad()
    def heatmap(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(bgr_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return self.model(to_tensor(rgb).unsqueeze(0).to(self.device))[0, 0].float().cpu().numpy()

    def predict(self, bgr_image: np.ndarray) -> tuple[float, float, float]:
        heat = self.heatmap(bgr_image)
        v, u = np.unravel_index(int(np.argmax(heat)), heat.shape)
        v0, v1 = max(v - REFINE_RADIUS, 0), min(v + REFINE_RADIUS + 1, heat.shape[0])
        u0, u1 = max(u - REFINE_RADIUS, 0), min(u + REFINE_RADIUS + 1, heat.shape[1])
        window = heat[v0:v1, u0:u1]
        vv, uu = np.mgrid[v0:v1, u0:u1]
        u_ref, v_ref = (window * uu).sum() / window.sum(), (window * vv).sum() / window.sum()
        # Heatmap pixel (u, v) covers [u, u+1); the click was stored at its pixel centre (+0.5), so undo both.
        height, width = bgr_image.shape[:2]
        x = (u_ref + 0.5) * width / IMAGE_SIZE[0] - 0.5
        y = (v_ref + 0.5) * height / IMAGE_SIZE[1] - 0.5
        return float(x), float(y), float(heat[v, u])
