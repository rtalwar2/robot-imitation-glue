"""Derive cap-sensor thresholds and calibrated ranges from recorded calibration runs.

Calibration runs are `sensor_logs/run_NNNN.json` produced by `open_bottle_demo_analyze_sensors.py`
(samples [t, v0, v1, v2] at ~48 Hz + timestamped motion events). Point this at the runs recorded
under the CURRENT sensor layout; it prints per-channel covered/uncovered bands, margins, proposed
`PER_CHANNEL_THRESHOLDS` and `CALIBRATED_RANGE`, and verifies the `SENSOR_CHECKPOINTS` mapping
(which channel has popped by which stage) — it never writes to any constants file.

How the windows are cut, and why (validated against runs 0003/0005/0006, the ones the constants in
use came from — naive event-relative windows mis-segment them in both directions):

* The covered baseline RISES during the motion itself (gripper position / light scattering inside
  the cap), so the covered ceiling must span the whole pre-pop interval, not just the hover
  window. But channels can pop BEFORE their checkpoint event (mid-leg), and transient light leaks
  can briefly reach the open level before the real pop. Fixed guards around the event therefore
  include open samples in "covered" — or leave the plateau empty when the recording ends at the
  event.
* Each channel's transition is DETECTED from its own trace, threshold-free: baseline = median of
  the first 2 s; plateau level = median of the last `--tail` s; if that step is smaller than the
  run's own noise the channel never opened in this run and is excluded. Crossing = the last sample
  at or below the open level (baseline + `--open-frac` x step). Covered = samples before
  crossing−guard that sit BELOW the open level — pre-pop samples at or above it are momentary
  light leaks, not covered readings, and are excluded from the band but REPORTED (they are the
  re-cover/pre-cross phenomenon the protocol flags). Plateau = samples after crossing+guard
  (clamped to the stream tail when the recording ends near the crossing — thin plateaus are
  flagged; record a second or two with the cap visibly open whenever you can; the demo script's
  `.wait()` on the retreat does this now).
* Bounds are robust: covered ceiling = p99 within the below-open-level subset, plateau floor = p1,
  the proposal is the midpoint of the worst case across runs. Anything a percentile still cannot
  fix shows up as `pre-pop threshold crossings` (covered-window samples at or above the PROPOSED
  threshold): those are checkpoint-killers — a sustained excursion at event-verify time would
  pass a checkpoint before the tab has popped — so weigh them before accepting the constants.
* The detected crossing is printed next to the event time — a mismatch is itself diagnostic of
  where the motion stages sit relative to where the tabs actually pop.

The proposal is advisory, not gospel: the constants in use were eyeballed bands, and this
reproduces them on the legacy runs (S0 lands exactly; S2 within 0.03 V; S1 differs but both sit
inside its 0.4 V separation). Compare the printed bands, plot with --plot, and keep
`hardware/bottle_sensor.py`, the demo script's synced copy, AND `CALIBRATED_RANGE` in
`train_ast_bottle.py` moving together.

Usage:
    python derive_thresholds.py run_0007 run_0008 run_0009
    python derive_thresholds.py --newest 3
    python derive_thresholds.py run_0007 --plot /tmp/cal_plots
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

DEFAULT_LOG_DIR = Path("/home/rtalwar/robot-imitation-glue/bottle_experiment/sensor_logs")
BOTTLE_SENSOR_PY = Path("/home/rtalwar/robot-imitation-glue/robot_imitation_glue/hardware/bottle_sensor.py")

# Channel -> the checkpoint event that gates it, ONE PER CHANNEL (repeat an event when a
# checkpoint gates several channels, as the 2026-10-07 remap does for S1+S2 at leg_3_end).
# Mirrors SENSOR_CHECKPOINTS in hardware/bottle_sensor.py; overridable with --events. Used for
# the mapping check and printed against the detected crossings, NOT for cutting the analysis
# windows (see module docstring).
DEFAULT_TRANSITION_EVENTS = ["leg_2_end", "leg_3_end", "leg_3_end"]


def load_run(path: Path) -> tuple[list[list[float]], dict[str, float]]:
    data = json.loads(path.read_text())
    events: dict[str, float] = {}
    for e in data["events"]:
        if e.get("time") is not None:
            events[e["name"]] = e["time"]  # later occurrences win: retries re-log the boundaries
    return data["samples"], events


def detect_transition(
    samples: list[list[float]], ch: int, tail: float, open_frac: float
) -> tuple[float, float] | None:
    """Threshold-free per-run transition for one channel: (crossing_time, open_level), or None if
    the channel never opened in this run (no step to the stream tail beyond its own noise).

    open_level = baseline + open_frac x (plateau - baseline). It must sit ABOVE the pre-pop
    mid-motion rise (S2's covered baseline climbs ~75% of the way to the plateau before the tab
    actually pops), so a channel counts as opened only once it is essentially at the plateau
    level; samples past open_level before the crossing are genuine light leaks, not slow rise."""
    values = [(s[0], s[ch + 1]) for s in samples]
    base = statistics.median([v for t, v in values if t <= values[0][0] + 2.0])
    plateau = [v for t, v in values if t >= values[-1][0] - tail]
    plateau_med = statistics.median(plateau)
    noise = statistics.pstdev(plateau) if len(plateau) > 2 else 0.005
    step = plateau_med - base
    if step < max(0.05, 5 * noise):
        return None
    open_level = base + open_frac * step
    below = [t for t, v in values if v <= open_level]
    return (below[-1], open_level) if below else None


def measure_run(samples: list[list[float]], args) -> tuple[dict[int, tuple], list[str]]:
    """Per-channel (crossing, covered, plateau, clamped, leaks, open_level, n_pre) for one run."""
    ch_data, notes = {}, []
    for ch in range(3):
        detected = detect_transition(samples, ch, args.tail, args.open_frac)
        if detected is None:
            notes.append(f"S{ch}: never opened (no step to the stream tail) — excluded")
            continue
        crossing, open_level = detected
        covered_all = [s[ch + 1] for s in samples if s[0] < crossing - args.guard]
        covered = [v for v in covered_all if v < open_level]
        leaks = len(covered_all) - len(covered)  # pre-pop excursions to the open level
        plateau = [s[ch + 1] for s in samples if s[0] >= crossing + args.guard]
        clamped = False
        if not plateau:  # recording ends before crossing+guard: fall back to the stream tail
            plateau = [s[ch + 1] for s in samples if s[0] >= max(crossing, samples[-1][0] - args.tail)]
            clamped = True
        if len(covered) < 20 or len(plateau) < 5:
            notes.append(f"S{ch}: unusable windows (covered={len(covered)}, plateau={len(plateau)}) — excluded")
            continue
        ch_data[ch] = (crossing, covered, plateau, clamped, leaks, open_level, len(covered_all))
    return ch_data, notes


def aggregate(windows) -> tuple[dict, dict, dict, dict]:
    per_ch = {ch: [w[3][ch] for w in windows if ch in w[3]] for ch in range(3)}
    ceiling = {ch: max(np.percentile(d[1], 99) for d in per_ch[ch]) for ch in range(3) if per_ch[ch]}
    floor = {ch: min(np.percentile(d[2], 1) for d in per_ch[ch]) for ch in range(3) if per_ch[ch]}
    proposed = {ch: round((float(ceiling[ch]) + float(floor[ch])) / 2, 2) for ch in ceiling}
    return per_ch, ceiling, floor, proposed


def report_proposal(ceiling, floor, proposed, args) -> None:
    print("=== proposal ===")
    for ch in sorted(proposed):
        margin = floor[ch] - ceiling[ch]
        print(
            f"  S{ch}: covered ceiling (p99) {float(ceiling[ch]):.3f} | plateau floor (p1) {float(floor[ch]):.3f}"
            f" | margin {margin:+.3f}V{'  !! THIN' if margin < args.min_margin else ''} -> threshold {proposed[ch]:.2f}"
        )
    live = current_thresholds(BOTTLE_SENSOR_PY)
    if live and len(live) == 3:
        print(f"  in-use (bottle_sensor.py): {[f'{v:.2f}' for v in live]}")


def report_paste(windows, proposed) -> None:
    # The range is the GLOBAL min/max over all samples of the runs where the channel opened --
    # deliberately not the window bounds. normalize_sensor reads raw values frame-by-frame at
    # train/screen time, and demonstration frames contain the transients the analysis windows
    # exclude (a pre-pop overshoot can top the plateau; S1's does by 9 mV). A normalization
    # envelope that misses values the network will see is the one error this paste cannot have.
    lo, hi = {}, {}
    for ch in range(3):
        vals = [s[ch + 1] for _, samples, _, ch_data in windows if ch in ch_data for s in samples]
        lo[ch], hi[ch] = min(vals), max(vals)
    cal_range = tuple((round(float(lo[ch]), 3), round(float(hi[ch]), 3)) for ch in range(3))
    print("\nPaste candidates (bottle_sensor.py + train_ast_bottle.py CALIBRATED_RANGE, together):")
    print(f"PER_CHANNEL_THRESHOLDS = [{proposed[0]}, {proposed[1]}, {proposed[2]}]  # S0, S1, S2, in volts")
    print(f"CALIBRATED_RANGE = ({cal_range[0]}, {cal_range[1]}, {cal_range[2]})  # global min/max over opened runs")


def report_bands(windows, transitions, proposed, args) -> None:
    print("\n--- per-run bands ---")
    for stem, samples, events, ch_data in windows:
        print(f"  {stem} ({len(samples)} samples, {samples[-1][0]:.1f}s):")
        for ch in sorted(ch_data):
            crossing, covered, plateau, clamped, leaks, _open, n_pre = ch_data[ch]
            ev_t = events.get(transitions[ch])
            drift = (
                f" | {transitions[ch]}@{ev_t:.2f}s (crossing {crossing - ev_t:+.2f}s vs event)"
                if ev_t is not None
                else ""
            )
            flags = []
            if leaks:
                flags.append(f"pre-pop light leaks >= open level: {leaks}/{n_pre} ({100 * leaks / n_pre:.1f}%)")
            pre_thr = sum(1 for v in covered if v >= proposed[ch])
            if pre_thr:
                flags.append(f"PRE-POP CROSSINGS >= threshold: {pre_thr}/{n_pre}")
            rec = sum(1 for v in plateau if v < proposed[ch])
            if rec:
                flags.append(f"re-cover dips < threshold: {rec}/{len(plateau)} ({100 * rec / len(plateau):.1f}%)")
            if clamped:
                flags.append("plateau CLAMPED to tail")
            if len(plateau) < args.min_plateau:
                flags.append(f"plateau only {len(plateau)} samples")
            print(
                f"    S{ch}: crossing {crossing:.2f}s{drift}  "
                f"covered[{min(covered):.3f}, p50 {statistics.median(covered):.3f}, p99 {np.percentile(covered, 99):.3f}] n={len(covered)}  "
                f"plateau[{min(plateau):.3f}, p50 {statistics.median(plateau):.3f}, {max(plateau):.3f}] n={len(plateau)}"
                + ("  !! " + "; ".join(flags) if flags else "")
            )


def debounced(samples: list[list[float]], t: float, window: float = 0.3) -> list[float] | None:
    """Mean reading over [t-window, t] — the rule SensorLogger.current_reading uses at the robot's
    checkpoints, so the mapping check judges the signal the checkpoint itself would see."""
    recent = [s[1:] for s in samples if max(0.0, t - window) <= s[0] <= t]
    if not recent:
        return None
    return [sum(col) / len(col) for col in zip(*recent)]


def report_mapping(windows, transitions, proposed) -> None:
    print("\n=== checkpoint mapping (debounced reading, the signal the checkpoint sees, at each event) ===")
    for stem, samples, events, _ in windows:
        notes = []
        for ch, event in enumerate(transitions):
            if event not in events or ch not in proposed:
                notes.append(f"{event}: skipped (no event or no threshold)")
                continue
            reading = debounced(samples, events[event])
            if reading is None:
                notes.append(f"{event}: no samples")
                continue
            if reading[ch] < proposed[ch]:
                notes.append(f"S{ch} NOT open at {event} ({reading[ch]:.3f} < {proposed[ch]:.2f})")
            for later in range(ch + 1, 3):
                # "already open" is only an anomaly if the later channel's OWN gate is a
                # different event; channels sharing a checkpoint (S1+S2 @ leg_3_end) are
                # expected to be open at it.
                if later in proposed and transitions[later] != event and reading[later] >= proposed[later]:
                    notes.append(f"S{later} already open at {event}")
        print(f"  {stem}: {'OK' if not notes else '; '.join(notes)}")


def make_plots(windows, transitions, proposed, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, samples, events, _ in windows:
        t = [s[0] for s in samples]
        fig, ax = plt.subplots(figsize=(12, 4))
        for ch in range(3):
            ax.plot(t, [s[ch + 1] for s in samples], label=f"S{ch}", lw=1.0)
            if ch in proposed:
                ax.axhline(proposed[ch], ls=":", lw=0.8)
        for name in transitions:
            if name in events:
                ax.axvline(events[name], ls="--", lw=0.8)
                ax.text(events[name], ax.get_ylim()[1], f" {name}", rotation=90, fontsize=7, va="top")
        ax.set_title(stem)
        ax.legend()
        fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"plot: {out_dir / (stem + '.png')}")


def current_thresholds(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    match = re.search(r"PER_CHANNEL_THRESHOLDS\s*=\s*\[([^\]]+)\]", path.read_text())
    return [float(v) for v in match.group(1).split(",")] if match else None


def resolve_runs(args) -> list[Path]:
    if args.newest:
        files = sorted(DEFAULT_LOG_DIR.glob("run_*.json"))
        if not files:
            sys.exit(f"no run files in {DEFAULT_LOG_DIR}")
        return files[-args.newest :]
    runs = []
    for entry in args.runs:
        path = Path(entry)
        if not path.exists():
            path = DEFAULT_LOG_DIR / (entry if entry.endswith(".json") else f"{entry}.json")
        if not path.exists():
            sys.exit(f"run not found: {entry} (nor at {path})")
        runs.append(path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive cap-sensor thresholds from calibration runs (see module docstring)."
    )
    parser.add_argument("runs", nargs="*", help="run stems (run_0007) or paths; omit with --newest")
    parser.add_argument("--newest", type=int, default=0, help="take the N most recent run files")
    parser.add_argument("--events", default=",".join(DEFAULT_TRANSITION_EVENTS), help="checkpoint event per channel")
    parser.add_argument("--guard", type=float, default=0.5, help="s excluded around the detected crossing")
    parser.add_argument("--tail", type=float, default=0.25, help="s used for the plateau level / fallback window")
    parser.add_argument("--min-margin", type=float, default=0.15, help="warn below this V between ceiling and floor")
    parser.add_argument("--min-plateau", type=int, default=20, help="warn below this many plateau samples")
    parser.add_argument(
        "--open-frac", type=float, default=0.75, help="fraction of the baseline->plateau step defining 'opened'"
    )
    parser.add_argument("--plot", type=Path, help="write one PNG per run here")
    args = parser.parse_args()

    transitions = [e.strip() for e in args.events.split(",")]
    if len(transitions) != 3:
        sys.exit("expected exactly 3 checkpoint event names (one per channel)")
    runs = resolve_runs(args)
    if not runs:
        sys.exit(f"no runs given; available: {[p.stem for p in sorted(DEFAULT_LOG_DIR.glob('run_*.json'))]}")
    print(f"runs: {', '.join(p.stem for p in runs)}")
    print(
        f"windows: covered = pre-crossing-<{args.guard}s samples below open level | "
        f"plateau >= crossing+{args.guard}s (tail {args.tail}s, open frac {args.open_frac})\n"
    )

    # Every statistic that depends on the PROPOSED threshold (crossing counts, mapping check) needs
    # all runs measured first — hence two passes.
    windows = []  # (stem, samples, events, ch_data)
    for path in runs:
        samples, events = load_run(path)
        ch_data, notes = measure_run(samples, args)
        for note in notes:
            print(f"  !! {path.stem} {note}")
        windows.append((path.stem, samples, events, ch_data))

    per_ch, ceiling, floor, proposed = aggregate(windows)
    missing = [f"S{ch}" for ch in range(3) if ch not in proposed]
    if missing:
        sys.exit("no usable windows for channel(s) " + ", ".join(missing) + " — check the runs")

    report_proposal(ceiling, floor, proposed, args)
    report_paste(windows, proposed)
    report_bands(windows, transitions, proposed, args)
    report_mapping(windows, transitions, proposed)

    if args.plot:
        make_plots(windows, transitions, proposed, args.plot)


if __name__ == "__main__":
    main()
