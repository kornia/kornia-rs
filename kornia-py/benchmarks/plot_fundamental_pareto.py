"""Plot measured Pareto frontiers for the optimized F7/F8 public APIs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"F7": "#14865b", "F8": "#2864b4"}


def frontier(points):
    """Keep configurations not dominated in latency and pose mAA."""
    return sorted(
        [p for p in points if not any(
            q["latency_ms"] <= p["latency_ms"] and q["maa"] >= p["maa"]
            and (q["latency_ms"] < p["latency_ms"] or q["maa"] > p["maa"])
            for q in points
        )], key=lambda p: p["latency_ms"],
    )


def measured_points(data):
    # F8's "matched" and "tuned" rows use identical settings. Pool their
    # timing observations per pair/seed rather than choosing the faster mode.
    grouped = defaultdict(list)
    rows = data.get("rows")
    if rows is None:
        rows = [{**r, "after": {"error_deg": r["error_deg"], "support": r["support"],
                                "iterations": r["iterations"]},
                 "after_ms": r["timing_samples_ms"]} for r in data["results"]]
    for row in rows:
        if row["stage"] == "budgets":
            key = (row["backend"], row["budget"], row["threshold_px"], row["pair"], row["seed"])
            grouped[key].append(row)
    configs = defaultdict(list)
    for (backend, budget, threshold, pair, seed), rows in grouped.items():
        assert all(r["after"] == rows[0]["after"] for r in rows)
        times = [t for r in rows for t in r["after_ms"]]
        error = rows[0]["after"]["error_deg"]
        maa = np.mean((np.inf if error is None else error) < np.arange(1, 11))
        configs[(backend, budget, threshold)].append((seed, np.median(times), maa))
    points = []
    for (backend, budget, threshold), observations in sorted(configs.items()):
        values = np.array(observations)
        by_seed = np.array([values[values[:, 0] == seed, 1:].mean(axis=0)
                            for seed in np.unique(values[:, 0])])
        mean = values[:, 1:].mean(axis=0)
        sd = by_seed.std(axis=0, ddof=1)
        points.append({"backend": backend, "engine": backend.split()[1],
                       "solver": backend.split()[2], "budget": budget,
                       "threshold_px": threshold, "latency_ms": float(mean[0]),
                       "maa": float(mean[1]), "seed_sd_ms": float(sd[0]),
                       "seed_sd_maa": float(sd[1]), "pair_seed_rows": len(observations)})
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.json.read_text())
    points = measured_points(data)
    metadata = data.get("metadata", {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    engines = sorted({p["engine"] for p in points})
    fig, axes = plt.subplots(1, len(engines), figsize=(6.25 * len(engines), 5.2))
    panels = []
    for ax, engine in zip(np.atleast_1d(axes), engines):
        mode = "shared"
        selected = [p for p in points if p["engine"] == engine
                    and p["threshold_px"] == 0.5]
        best = frontier(selected)
        panels.append({"engine": engine, "mode": mode, "frontier": best})
        for solver in ("F7", "F8"):
            candidates = [p for p in selected if p["solver"] == solver]
            own_front = frontier(candidates)
            ax.scatter([p["latency_ms"] for p in candidates], [p["maa"] for p in candidates],
                       color=COLORS[solver], alpha=0.35, s=25)
            for p in candidates:
                if p not in own_front:
                    ax.annotate(str(p["budget"]), (p["latency_ms"], p["maa"]),
                                xytext=(3, -14), textcoords="offset points",
                                color=COLORS[solver], alpha=0.6, fontsize=8)
            ax.plot([p["latency_ms"] for p in own_front], [p["maa"] for p in own_front],
                    color=COLORS[solver], marker="o", label=solver, lw=1.8)
            for p in own_front:
                ax.errorbar(p["latency_ms"], p["maa"], yerr=p["seed_sd_maa"],
                            color=COLORS[solver], alpha=0.35, capsize=2, lw=0.8)
                ax.annotate(str(p["budget"]), (p["latency_ms"], p["maa"]),
                            xytext=(3, 7 if solver == "F7" else -14),
                            textcoords="offset points", color=COLORS[solver], fontsize=8)
        ax.step([p["latency_ms"] for p in best], [p["maa"] for p in best],
                where="post", color="#333333", ls="--", alpha=0.7, lw=1,
                label="Combined Pareto frontier")
        ax.set_xscale("log")
        low = min(p["latency_ms"] for p in selected) * 0.75
        high = max(p["latency_ms"] for p in selected) * 1.25
        ax.set_xlim(low, high)
        ticks = [v * 10.0 ** exponent for exponent in range(-3, 4) for v in (1, 2, 5)]
        ticks = [v for v in ticks if low <= v <= high]
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        kind = "native" if "FFI excluded" in metadata.get("timing_kind", "") else "public-call"
        ax.set_xlabel(f"Mean {kind} latency (ms, log scale)")
        ax.set_ylabel("Pose mAA (1–10°)")
        ax.set_ylim(min(0.10, min(p["maa"] - p["seed_sd_maa"] for p in selected) - 0.01),
                    max(0.36, max(p["maa"] + p["seed_sd_maa"] for p in selected) + 0.01))
        ax.grid(alpha=0.18, which="both")
        ax.set_title(f"{engine}: shared 0.5 px cutoff")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(metadata.get("plot_title", "Optimized 7-point vs retained 8-point: measured accuracy / time tradeoff"),
                 fontsize=14, y=0.99)
    confidence = metadata.get("confidence", "legacy solver defaults")
    if isinstance(confidence, dict) and len(set(confidence.values())) == 1:
        confidence = next(iter(confidence.values()))
    fig.text(0.5, 0.016,
             f"34 St Peter’s pairs · shared 0.5 px cutoff · confidence {confidence} · {metadata.get('rounds', 5)} timing rounds\n"
             f"Labels: draw caps · bars: seed SD · {metadata.get('timing_kind', 'public calls including FFI')}",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    for suffix in ("png", "svg"):
        output = args.out_dir / f"pareto.{suffix}"
        fig.savefig(output, dpi=180)
        if suffix == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)
    payload = {"metadata": {"input_sha256": hashlib.sha256(args.json.read_bytes()).hexdigest(),
                             "method": "mean per-pair/seed medians; identical settings pooled across threshold-mode labels",
                             "timed_cutoff_px": 0.5,
                             "confidence": confidence,
                             "frontier": "observed configurations only; no interpolation or significance claim"},
               "points": [p for p in points if p["threshold_px"] == 0.5], "panels": panels,
               "global_frontier": frontier([p for p in points if p["threshold_px"] == 0.5])}
    (args.out_dir / "pareto.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["global_frontier"], indent=2))


if __name__ == "__main__":
    main()
