"""Data-efficiency curves (protocol §1.9 Figure A, per-task descriptive) from outputs/eval/bottle_results.json.

Per arm and level: success rate with a 95% Wilson interval, plus the outcome breakdown. Both arms are evaluated
on the same seeded test poses in the same order, so each level is also compared paired (exact McNemar) and the
four levels together with a Cochran-Mantel-Haenszel test stratified by level -- exploratory for one task; the
pre-registered primary analysis pools across tasks (§1.8).

Usage:
    python -m robot_imitation_glue.ur5station.bottle.plot_curves [--condition id] [--out outputs/figures]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, norm

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "outputs/eval/bottle_results.json"
EPISODE_ORDER = REPO / "datasets/bottle_experiment/prepared/episode_order_pool50.json"
LEVELS = [25, 50, 75, 100]
ARMS = {"generic": "control (generic)", "generic_c": "instrumented (generic_c)"}
COLORS = {"generic": "#1f77b4", "generic_c": "#d62728"}
SUFFIX = "_ddim10_a8"  # the fixed inference settings (§1.8); rows with other settings are not comparable
OUTCOMES = ["success", "manual_stop", "force_abort", "robot_stopped", "timeout"]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def mcnemar_exact(a, b):
    """a, b: bool arrays over the same poses. Returns (b-only, c-only, two-sided exact p)."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    return only_a, only_b, (binomtest(only_a, n, 0.5).pvalue if n else 1.0)


def cmh(tables):
    """Cochran-Mantel-Haenszel (with continuity correction) over 2x2 tables [[a, b], [c, d]], rows = arms."""
    num, var, mh_num, mh_den = 0.0, 0.0, 0.0, 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        num += a - r1 * c1 / n
        var += r1 * r2 * c1 * c2 / (n**2 * (n - 1))
        mh_num += a * d / n
        mh_den += b * c / n
    chi2 = (abs(num) - 0.5) ** 2 / var
    from scipy.stats import chi2 as chi2_dist
    return chi2, chi2_dist.sf(chi2, 1), (mh_num / mh_den if mh_den else float("inf"))


def load(condition):
    rows = json.load(open(RESULTS))
    data = {}
    for arm in ARMS:
        for lvl in LEVELS:
            cfg = f"{arm}_{lvl}{SUFFIX}"
            rs = sorted((r for r in rows if r["config"] == cfg and r["condition"] == condition),
                        key=lambda r: r["pose_index"])
            if rs:
                data[arm, lvl] = rs
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="id")
    ap.add_argument("--out", default=str(REPO / "outputs/figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_eps = json.load(open(EPISODE_ORDER))["levels"]
    data = load(args.condition)

    print(f"## Success rate, {args.condition.upper()} (95% Wilson CI)\n")
    print("| level | episodes | " + " | ".join(ARMS.values()) + " | paired: control-only / instr-only | McNemar p |")
    print("|---|---|" + "---|" * len(ARMS) + "---|---|")
    tables = []
    for lvl in LEVELS:
        cells = []
        for arm in ARMS:
            rs = data.get((arm, lvl), [])
            k, n = sum(r["success"] for r in rs), len(rs)
            lo, hi = wilson(k, n) if n else (np.nan, np.nan)
            cells.append(f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]" if n else "—")
        pair = ""
        if ("generic", lvl) in data and ("generic_c", lvl) in data:
            a = np.array([r["success"] for r in data["generic", lvl]])
            b = np.array([r["success"] for r in data["generic_c", lvl]])
            assert [r["pose_index"] for r in data["generic", lvl]] == [r["pose_index"] for r in data["generic_c", lvl]]
            oa, ob, p = mcnemar_exact(a, b)
            pair = f"{oa} / {ob} | {p:.2f}"
            tables.append([[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]])
        print(f"| {lvl}% | {len(n_eps[str(lvl)])} | " + " | ".join(cells) + f" | {pair} |")
    if len(tables) > 1:
        chi2, p, OR = cmh(tables)
        print(f"\nCMH over levels (control vs instrumented): chi2 = {chi2:.2f}, p = {p:.2f}, "
              f"common odds ratio (control success / instrumented success) = {OR:.2f}")

    print("\n## Outcome breakdown\n")
    print("| config | " + " | ".join(OUTCOMES) + " | of which operator-scored |")
    print("|---|" + "---|" * len(OUTCOMES) + "---|")
    for (arm, lvl), rs in data.items():
        counts = [sum(r["outcome"] == o for r in rs) for o in OUTCOMES]
        op = sum(r.get("success_source") == "operator" for r in rs)
        print(f"| {arm}_{lvl} | " + " | ".join(map(str, counts)) + f" | {op} |")

    # Figure A (single task): success vs % training data
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, arm in enumerate(ARMS):
        lv = [lvl for lvl in LEVELS if (arm, lvl) in data]
        k = np.array([sum(r["success"] for r in data[arm, lvl]) for lvl in lv])
        n = np.array([len(data[arm, lvl]) for lvl in lv])
        lo, hi = wilson(k, n)
        x = np.array(lv) + (i - 0.5) * 1.6  # small dodge so the intervals do not overlap
        ax.errorbar(x, k / n, yerr=[k / n - lo, hi - k / n], fmt="o-", capsize=3, color=COLORS[arm], label=ARMS[arm])
    ax.set_xticks(LEVELS, [f"{lvl}%\n({len(n_eps[str(lvl)])} ep)" for lvl in LEVELS])
    ax.set_xlabel("training data")
    ax.set_ylabel(f"success rate ({args.condition.upper()}, 20 rollouts)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Bottle opening: data efficiency")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"bottle_curve_{args.condition}.{ext}", dpi=200)

    # outcome breakdown, stacked
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels, bottoms = [], None
    keys = [(arm, lvl) for lvl in LEVELS for arm in ARMS if (arm, lvl) in data]
    bottoms = np.zeros(len(keys))
    ocol = {"success": "#2ca02c", "manual_stop": "#7f7f7f", "force_abort": "#ff7f0e", "robot_stopped": "#9467bd",
            "timeout": "#8c564b"}
    for o in OUTCOMES:
        v = np.array([sum(r["outcome"] == o for r in data[k]) for k in keys])
        if v.any():
            ax.bar(range(len(keys)), v, bottom=bottoms, color=ocol[o], label=o)
            bottoms += v
    ax.set_xticks(range(len(keys)), [f"{'ctrl' if a == 'generic' else 'instr'}\n{lvl}%" for a, lvl in keys])
    ax.set_ylabel("rollouts")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"bottle_outcomes_{args.condition}.{ext}", dpi=200)
    print(f"\nfigures: {out}/bottle_curve_{args.condition}.png, bottle_outcomes_{args.condition}.png")


if __name__ == "__main__":
    main()
