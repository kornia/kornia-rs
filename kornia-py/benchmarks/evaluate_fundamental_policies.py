"""Evaluate and plot-ready summarize the focused native RANSAC policy probe."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from bench_fundamental_solvers import digest, load_items, pose_error, selected_digest, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.999999)
    args = parser.parse_args()
    items, files = load_items(args.data_root, 34, 0.85)
    by_pair = {r["pair"]: r for r in items}
    groups = defaultdict(list)
    for line in args.raw.read_text().splitlines():
        request = json.loads(line)
        assert request["confidence"] == args.confidence
        for r in request["rows"]:
            groups[(request["pair"], request["seed"], request["budget"], r["mode"], r["solver"])].append(r)
    modes = defaultdict(list)
    for (pair, seed, budget, mode, solver), repetitions in groups.items():
        first = repetitions[0]
        assert all(r["model"] == first["model"] and r["mask"] == first["mask"]
                   and r["draws"] == first["draws"] for r in repetitions)
        item = by_pair[pair]
        mask = np.asarray(first["mask"], dtype=bool)
        error, mismatch, rank_ratio = None, 0, None
        if first["model"] is not None:
            f = np.asarray(first["model"]).reshape(3, 3, order="F")
            assert f.shape == (3, 3) and np.isfinite(f).all()
            a = np.c_[item["a"], np.ones(len(mask))]
            b = np.c_[item["b"], np.ones(len(mask))]
            fa, fb = a @ f.T, b @ f
            denominator = np.sum(fa[:, :2] ** 2 + fb[:, :2] ** 2, axis=1)
            residuals = np.divide(np.sum(fa * b, axis=1) ** 2, denominator,
                                  out=np.full(len(mask), np.inf), where=denominator > 0)
            mismatch = int(np.count_nonzero(mask != (residuals < 0.25)))
            assert mismatch == 0, (pair, seed, budget, mode, solver, mismatch)
            singular = np.linalg.svd(f, compute_uv=False)
            rank_ratio = float(singular[-1] / singular[0])
            assert rank_ratio < 1e-10, (pair, seed, budget, mode, solver, rank_ratio)
            value = pose_error(f, item["a"][mask], item["b"][mask], *item["calib"])
            error = float(value) if np.isfinite(value) else None
        times = [r["ns"] / 1e6 for r in repetitions]
        modes[mode].append({"stage": "budgets", "threshold_mode": "matched",
                            "pair": pair, "seed": seed, "budget": budget,
                            "backend": f"Rust generic F{solver}", "solver": f"{solver}point",
                            "api": "generic", "threshold_px": 0.5, "n": len(item["a"]),
                            "median_ms": float(np.median(times)), "timing_samples_ms": times,
                            "error_deg": error, "support": int(mask.sum()),
                            "iterations": first["draws"], "exception": None,
                            "matrix_mask_mismatch": mismatch, "rank_ratio": rank_ratio})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for mode, rows in sorted(modes.items()):
        metadata = {"confidence": args.confidence, "rounds": len(groups[next(k for k in groups if k[3] == mode)]),
                    "matched_threshold_px": 0.5, "mode": mode, "lo_every": 1 if "lo" in mode else 0,
                    "timing_kind": "native Rust driver; FFI excluded", "dataset": "34 St Peter's pairs, 3 seeds",
                    "plot_title": f"Same threshold and confidence: F7 vs F8 with {mode.replace('_', ' ')}",
                    "raw_sha256": digest(args.raw), "probe_sha256": digest(args.probe),
                    "harness_sha256": digest(Path(__file__)),
                    "input_files_sha256": files, "selected_input_sha256": selected_digest(items),
                    "fixed_prefix_method": "diagnostic consensus suppresses only the adaptive-cap count; model score and masks retained" if "fixed" in mode else None}
        payload = {"metadata": metadata, "results": rows, "summary": summarize(rows)}
        (args.out_dir / f"{mode}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        for r in payload["summary"]:
            report.append({"mode": mode, **r})
            print(mode, r["backend"], r["budget"], r["maa"], r["mean_median_ms"], flush=True)
    (args.out_dir / "policy-summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
