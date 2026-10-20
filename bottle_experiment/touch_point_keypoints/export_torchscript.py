"""Export the trained touch-point model to TorchScript, so the rig env can run it with plain torch.

The training stack (keypoint-detection, pytorch-lightning<=1.9.4) lives in its own conda env and would
clash with the rig env; a traced model needs neither. Writes models/<split>.pt next to <split>.ckpt.

    ~/miniconda3/envs/keypoint-detection/bin/python bottle_experiment/touch_point_keypoints/export_torchscript.py [final]
"""

import sys
from pathlib import Path

import torch
from keypoint_detection.utils.load_checkpoints import load_from_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import IMAGE_SIZE, OUT_DIR  # noqa: E402


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "final"
    model = load_from_checkpoint(str(OUT_DIR / "models" / f"{split}.ckpt")).eval()
    example = torch.rand(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    # Trace the bare network, not the LightningModule (tracing that asks for a Trainer). Its forward() is
    # exactly sigmoid(unnormalized_model(x)), so the pair below is the same function: heatmaps (N, 1, H, W).
    network = torch.nn.Sequential(model.unnormalized_model, torch.nn.Sigmoid()).eval()
    with torch.no_grad():
        traced = torch.jit.trace(network, example)
        error = (traced(example) - model(example)).abs().max().item()
    out = OUT_DIR / "models" / f"{split}.pt"
    traced.save(str(out))
    print(f"wrote {out} (traced vs eager max abs difference {error:.2e})")


if __name__ == "__main__":
    main()
