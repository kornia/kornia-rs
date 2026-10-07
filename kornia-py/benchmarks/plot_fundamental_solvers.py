"""Plot the focused fundamental RANSAC comparison's recorded JSON results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"F7": "#198754", "F8": "#2864b4"}


def seed_stats(data, mode, backend, budget, threshold):
    rows = [r for r in data["results"] if r["stage"] == "budgets"
            and r["threshold_mode"] == mode and r["backend"] == backend
            and r["budget"] == budget and r["threshold_px"] == threshold]
    per_seed = []
    for seed in sorted({r["seed"] for r in rows}):
        selected = [r for r in rows if r["seed"] == seed]
        error = np.array([np.inf if r["error_deg"] is None else r["error_deg"] for r in selected])
        per_seed.append((np.mean([r["median_ms"] for r in selected]),
                         np.mean(error[:, None] < np.arange(1, 11))))
    return np.std(per_seed, axis=0, ddof=1) if len(per_seed) > 1 else np.zeros(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, engine in enumerate(("generic", "k3d")):
        for col, mode in enumerate(("matched", "tuned")):
            ax = axes[row, col]
            for solver in ("F7", "F8"):
                backend = f"Rust {engine} {solver}"
                points = sorted([r for r in data["summary"] if r["stage"] == "budgets"
                                 and r["threshold_mode"] == mode and r["backend"] == backend],
                                key=lambda r: r["budget"])
                x, y = [r["mean_median_ms"] for r in points], [r["maa"] for r in points]
                sd = np.array([seed_stats(data, mode, backend, r["budget"], r["threshold_px"])
                               for r in points])
                ax.errorbar(x, y, xerr=sd[:, 0], yerr=sd[:, 1], marker="o", capsize=2,
                            color=COLORS[solver], label=f"{solver}, {points[0]['threshold_px']:g} px")
                for r in points:
                    other = next(v for v in data["summary"] if v["stage"] == "budgets"
                                 and v["threshold_mode"] == mode and v["backend"] == f"Rust {engine} {'F8' if solver == 'F7' else 'F7'}"
                                 and v["budget"] == r["budget"])
                    offset = 6 if r["maa"] >= other["maa"] else -14
                    ax.annotate(str(r["budget"]), (r["mean_median_ms"], r["maa"]),
                                xytext=(4, offset), textcoords="offset points", fontsize=8)
            ax.set_title(f"{engine}: {'shared 0.5 px cutoff' if mode == 'matched' else 'independently selected cutoffs'}")
            ax.set_xlabel("Mean per-pair/seed median public-call latency (ms)")
            ax.set_ylabel("Pose mAA (1–10°)")
            ax.grid(alpha=.2)
            ax.legend(loc="best")
    fig.suptitle("Fundamental RANSAC: 7-point vs 8-point\n34 St Peter’s pairs · 3 seeds × 3 interleaved timing rounds · bars: seed SD")
    for suffix in ("png", "svg"):
        fig.savefig(args.out_dir / f"accuracy-latency.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, engine in zip(axes, ("generic", "k3d")):
        for solver in ("F7", "F8"):
            backend = f"Rust {engine} {solver}"
            points = sorted([r for r in data["summary"] if r["stage"] == "sweep" and r["backend"] == backend],
                            key=lambda r: r["threshold_px"])
            ax.plot([r["threshold_px"] for r in points], [r["maa"] for r in points],
                    marker="o", color=COLORS[solver], label=solver)
            selected = data["chosen_thresholds_px"][backend]
            point = next(r for r in points if r["threshold_px"] == selected)
            ax.scatter([selected], [point["maa"]], facecolor=COLORS[solver], edgecolor="black", s=90, zorder=3)
        ax.set_title(engine)
        ax.set_xlabel("Sampson cutoff (physical px)")
        ax.set_ylabel("Pose mAA (1–10°)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Threshold sensitivity at 1,000 draws · black outlines: selected cutoffs")
    for suffix in ("png", "svg"):
        fig.savefig(args.out_dir / f"thresholds.{suffix}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
