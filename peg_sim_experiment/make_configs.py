"""Training configs for the two sim arms, derived from the bottle control's config so the policy is the same one.

    python make_configs.py --dataset peg_train_200

Writes configs/peg_<n>.json (control: action 9) and configs/peg_c_<n>.json (instrumented:
action 9 + seated). Only the keys below change from bottle_generic_100.json: dataset, image size and crop, state
width, action width, audio off (no microphone in the sim), names/paths, and a PRIVATE Hub repo. The two arms differ in
the dataset root and the action width only (asserted).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from peg_env import IMAGE_H, IMAGE_W

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "robot_imitation_glue/ur5station/bottle/configs/bottle_generic_100.json"
OUT = Path(__file__).parent / "configs"
STATE_DIM = 18
CROP = [int(IMAGE_H * 0.9), int(IMAGE_W * 0.9)]  # the bottle crops 216x288 of 240x320


def make(dataset: str, instrumented: bool) -> dict:
    c = copy.deepcopy(json.loads(BASE.read_text()))
    ds = dataset + ("_c" if instrumented else "")
    name = f"peg{'_c' if instrumented else ''}_{dataset.removeprefix('peg_train_')}"
    c["dataset"].update({"repo_id": ds, "root": f"datasets/peg_sim/{ds}/", "video_backend": "pyav"})
    c["job_name"], c["output_dir"] = name, f"outputs/train/peg_sim/{name}"
    c["wandb"]["project"] = "peg_sim_goal"
    p = c["policy"]
    p["input_features"] = {
        "observation.images.wrist_image": {"type": "VISUAL", "shape": [3, IMAGE_H, IMAGE_W]},
        "observation.state": {"type": "STATE", "shape": [STATE_DIM]},
    }
    p["output_features"] = {"action": {"type": "ACTION", "shape": [10 if instrumented else 9]}}
    p["crop_shape"] = CROP
    p["audio_backbone"] = None
    p["normalization_mapping"].pop("AUDIO", None)
    p["push_to_hub"], p["repo_id"], p["private"] = True, f"ramen-noodels/{name}", True  # the bottle repos went public
    return c


def _diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from _diff(a.get(k), b.get(k), f"{prefix}{k}.")
    elif a != b:
        yield prefix[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="peg_train_200")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    control, treat = make(args.dataset, False), make(args.dataset, True)
    diff = set(_diff(control, treat))
    assert diff == {"dataset.repo_id", "dataset.root", "job_name", "output_dir",
                    "policy.output_features.action.shape", "policy.repo_id"}, diff
    for cfg in (control, treat):
        (OUT / f"{cfg['job_name']}.json").write_text(json.dumps(cfg, indent=2))
        print(f"wrote configs/{cfg['job_name']}.json  action {cfg['policy']['output_features']['action']['shape']}")
    print("arms differ only in:", sorted(diff))


if __name__ == "__main__":
    main()
