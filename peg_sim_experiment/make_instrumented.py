"""Build the instrumented arm's dataset: a copy of a collected peg dataset whose `action` is [action(9), seated(1)].

    python make_instrumented.py --src datasets/peg_sim/peg_train_200 --dst datasets/peg_sim/peg_train_200_c

The bottle generic_c mechanism: the sheet's "seated" contact rides as an extra channel of the denoised vector, so
the policy is trained to predict it alongside the actions and LerobotAgent(n_env_action_dims=9) slices it off at
inference -- it is never an input. Everything else (images, state, episode order) is byte-identical to the source,
so the two arms differ only in the action width. The per-episode and global action statistics are recomputed; the
global ones are the RELATIVE-action statistics the policy trains on (write_relative_action_stats passes extra
channels through unchanged).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
CHANNEL, CHANNEL_NAME = 0, "seated"  # index into the recorded `instrumentation` (seated, popped, depth)


def _widen(table: pa.Table, extra: np.ndarray) -> pa.Table:
    action = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
    wide = np.concatenate([action, extra[:, None].astype(np.float32)], 1)
    col = pa.array(list(wide), type=pa.list_(pa.float32(), wide.shape[1]))
    return table.set_column(table.column_names.index("action"), "action", col)


def _episode_stats(values: np.ndarray) -> dict[str, float]:
    out = {"min": values.min(), "max": values.max(), "mean": values.mean(), "std": values.std()}
    out.update({f"q{q:02d}": np.quantile(values, q / 100) for q in (1, 10, 50, 90, 99)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    if args.dst.exists():
        raise SystemExit(f"{args.dst} exists; never overwrite a dataset in place")
    shutil.copytree(args.src, args.dst)

    per_episode: dict[int, np.ndarray] = {}
    for f in sorted((args.dst / "data").glob("*/*.parquet")):
        t = pq.read_table(f)
        sensor = np.stack(t["instrumentation"].to_numpy(zero_copy_only=False))[:, CHANNEL]
        for ep in np.unique(t["episode_index"].to_numpy()):
            per_episode[int(ep)] = sensor[t["episode_index"].to_numpy() == ep]
        pq.write_table(_widen(t, sensor), f)

    for f in sorted((args.dst / "meta" / "episodes").glob("*/*.parquet")):
        t = pq.read_table(f)
        eps = t["episode_index"].to_numpy()
        for name in [c for c in t.column_names if c.startswith("stats/action/") and not c.endswith("/count")]:
            stat = name.rsplit("/", 1)[1]
            rows = [list(v) + [float(_episode_stats(per_episode[int(e)])[stat])]
                    for v, e in zip(t[name].to_pylist(), eps)]
            t = t.set_column(t.column_names.index(name), name, pa.array(rows, type=pa.list_(pa.float64())))
        pq.write_table(t, f)

    info_path = args.dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [info["features"]["action"]["shape"][0] + 1]
    info["features"]["action"]["names"] = list(info["features"]["action"]["names"]) + [CHANNEL_NAME]
    info_path.write_text(json.dumps(info, indent=4))

    sys.path.insert(0, str(REPO))
    from robot_imitation_glue.ur5station.bottle.prepare_datasets_bottle import write_relative_action_stats
    write_relative_action_stats(args.dst)
    frac = np.mean(np.concatenate(list(per_episode.values())))
    print(f"{args.dst.name}: action -> {info['features']['action']['shape'][0]} dims, "
          f"'{CHANNEL_NAME}' = 1 on {100 * frac:.1f}% of frames over {len(per_episode)} episodes")


if __name__ == "__main__":
    main()
