"""Per-run verdict for the leg-by-leg tuning loop: where in the motion did the cap yield?

Usage: check_pop_phase.py sensor_logs/run_NNNN.json [more runs...]

Markers are the fork's own (pose_start, hover_reached, grip_point_reached, push_end, leg_N_end,
retreat_home). Two views of the same data:

* a table of each channel's % of its total baseline->plateau excursion AT each marker, and
* segment attribution, which is the one that answers "which leg opened it": each channel's
  crossing is located threshold-free (baseline = median of the window's first 2 s, plateau =
  median of its last 2 s, open level = baseline + OPEN_FRAC x step) and placed inside the motion
  segment it falls in -- a segment is the interval BETWEEN two consecutive end-markers, so a
  crossing inside (leg_2_end, leg_3_end) happened while leg 3 was executing.

Missing markers are skipped, which is what happens as LEGS is grown one entry at a time.
"""
import json
import os
import sys

import numpy as np

OPEN_FRAC = 0.75  # same fraction derive_thresholds.py uses to call a channel "open"
MIN_STEP_V = 0.10  # below this the channel never opened in this run (run's own noise floor)
HOLD_S = 0.5  # a crossing must stay past the open level this long, else it is a lift-and-recover
WIN = 0.2  # median window at each marker, seconds

# segment each interval is named for, keyed by (start marker, end marker)
SEGMENTS = [
    ("grip_point_reached", "push_end", "push"),
    ("push_end", "leg_2_end", "leg 2"),
    ("leg_2_end", "leg_3_end", "leg 3"),
    ("leg_3_end", "leg_4_end", "leg 4"),
    ("leg_4_end", "leg_5_end", "leg 5"),
    ("leg_5_end", "leg_6_end", "leg 6"),
    ("leg_6_end", "leg_7_end", "leg 7"),
]
MARKER_ORDER = [
    "pose_start",
    "hover_reached",
    "grip_point_reached",
    "push_end",
    "leg_2_end",
    "leg_3_end",
    "leg_4_end",
    "leg_5_end",
    "leg_6_end",
    "leg_7_end",
    "retreat_home",
]


def med(t, V, t0, t1, ci):
    m = (t >= t0) & (t <= t1)
    return float(np.median(V[m, ci])) if m.sum() >= 3 else float("nan")


def attribute(t, V, ci, baseline, plateau, marks):
    """Return (crossing_time, segment, frac_into_segment, peak_rate) for one channel."""
    step = plateau - baseline
    if abs(step) < MIN_STEP_V:
        return None
    open_level = baseline + OPEN_FRAC * step
    v = V[:, ci]
    up = v > open_level
    for i in np.flatnonzero(up):
        end = t[i] + HOLD_S
        if not up[(t >= t[i]) & (t <= end)].all():
            continue  # transient: lifted past the level and fell back (the run_0012 shape)
        tc = t[i]
        # sharpest change in any WIN-wide window around the crossing = flick vs gradual
        lo, hi = max(0, i - int(2 * WIN / max(np.median(np.diff(t)), 1e-3))), min(len(t) - 1, i + 20)
        rate = float(np.max(np.abs(v[lo + 1 : hi + 1] - v[lo : hi - 1 + 1]))) if hi > lo + 1 else 0.0
        seg, frac = "before motion", 0.0
        for a, b, name in SEGMENTS:
            if a in marks and b in marks and marks[a] <= tc <= marks[b]:
                span = marks[b] - marks[a]
                seg, frac = name, ((tc - marks[a]) / span if span > 0 else 1.0)
                break
        else:
            if "leg_6_end" in marks and tc > marks["leg_6_end"]:
                seg = "after leg 6"
        return tc, seg, frac, rate, step
    return None


def report(path):
    d = json.load(open(path))
    S = np.array(d["samples"])
    t, V = S[:, 0], S[:, 1:]
    marks = {}
    for e in d["events"]:
        marks.setdefault(e["name"], e["time"])
    print("=" * 78)
    print(f"{os.path.basename(path)}  split={d.get('split')} pose_slot={d.get('pose_slot')}/{d.get('n_poses')}"
          f"  cap={np.round(d.get('planned_cap_center'), 3).tolist() if d.get('planned_cap_center') else '?'}")
    if "grip_point_reached" not in marks:
        print("  no grip_point_reached -- motion never started")
        return
    grip = marks["grip_point_reached"]
    base = [med(t, V, t.min(), t.min() + 2.0, c) for c in range(3)]
    plat = [med(t, V, t.max() - 2.0, t.max(), c) for c in range(3)]

    order = [k for k in MARKER_ORDER if k in marks]
    print(f"  {'marker':<20}{'t':>7}{'dt':>7}   " + "".join(f"{'ch'+str(c):>9}" for c in range(3)))
    prev = None
    for nm in order:
        tt = marks[nm]
        row = f"  {nm:<20}{tt:7.1f}" + (f"{tt - prev:7.1f}" if prev is not None else "        ")
        prev = tt
        for c in range(3):
            span = plat[c] - base[c]
            v = med(t, V, tt - WIN, tt + WIN, c)
            row += f"{(v - base[c]) / span * 100:8.0f}%" if abs(span) > MIN_STEP_V and v == v else "       --"
        print(row)

    print("  crossing (threshold-free) -> which motion segment did it:")
    fully = []
    for c in range(3):
        r = attribute(t, V, c, base[c], plat[c], marks)
        if r is None:
            print(f"    ch{c}: baseline {base[c]:.2f}V plateau {plat[c]:.2f}V -> NEVER OPENED (step "
                  f"{plat[c] - base[c]:+.2f}V < {MIN_STEP_V}V, sustained)")
            continue
        tc, seg, frac, rate, step = r
        fully.append((tc, seg))
        print(f"    ch{c}: {base[c]:.2f} -> {plat[c]:.2f}V (step {step:+.2f}) crosses at t={tc:.2f} "
              f"during {seg:<12} ({frac * 100:3.0f}% into it)  sharpest {WIN}s jump {rate:.2f}V")
    if len(fully) == 3:
        last_t, last_seg = max(fully)
        print(f"  ==> cap FULLY open during {last_seg} (ch with the latest crossing); "
              f"motion ends at t={marks.get('leg_6_end', float('nan')):.2f}")
    elif fully:
        print(f"  ==> only {len(fully)}/3 channels opened in this run")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
