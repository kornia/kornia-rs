#!/usr/bin/env python3
"""Focused real-data comparison of 7-point and 8-point fundamental RANSAC.

The input is the precomputed RootSIFT St Peter's Square set used by the
repository's RANSAC benchmark.  It selects the same 34 pairs (seed-zero
without-replacement selection), preserves match order after the common
SNN <= 0.85 filter, and evaluates recovered relative pose with the same
OpenCV convention as Kornia's ``geometry/ransac_cpu.py``.

Only public kornia-rs calls are timed.  The generic API receives a squared
Sampson cutoff; the pose-stack k3d API receives a pixel cutoff.  A seven-point
draw can produce multiple models, so equal draw caps are reported together
with wall-clock time and (where the public API exposes it) actual iterations.

Example (after building/installing the wheel under test)::

    python kornia-py/benchmarks/bench_fundamental_solvers.py \
      --data-root /Users/oldufo/dev/pydegensac/benchmarks/data \
      --json /tmp/fundamental-7pt-vs-8pt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import h5py
import numpy as np


THRESHOLDS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
POSE_THRESHOLDS_DEG = np.arange(1, 11, dtype=np.float64)
VARIANTS = (
    ("Rust generic F7", "generic", "7point"),
    ("Rust generic F8", "generic", "8point"),
    ("Rust k3d F7", "k3d", "7point"),
    ("Rust k3d F8", "k3d", "8point"),
)


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one file without exposing its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_digest(items: list[dict]) -> str:
    """Digest exactly the correspondences and calibration used by this run."""
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(item["pair"].encode())
        for key in ("a", "b"):
            hasher.update(np.ascontiguousarray(item[key]).tobytes())
        for camera in item["calib"]:
            for key in ("K", "R", "T"):
                hasher.update(np.ascontiguousarray(camera[key]).tobytes())
    return hasher.hexdigest()


def load_items(data_root: Path, pairs: int, match_threshold: float) -> tuple[list[dict], dict[str, str]]:
    """Load the recorded St Peter's selection and common SNN-filtered matches."""
    base = data_root / "f_data" / "st_peters_square"
    paths = {name: base / name for name in ("matches.h5", "match_conf.h5", "K1_K2.h5", "R.h5", "T.h5")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing tutorial files: " + ", ".join(missing))
    items: list[dict] = []
    with (
        h5py.File(paths["matches.h5"], "r") as matches_file,
        h5py.File(paths["match_conf.h5"], "r") as confidence_file,
        h5py.File(paths["K1_K2.h5"], "r") as intrinsics_file,
        h5py.File(paths["R.h5"], "r") as rotations_file,
        h5py.File(paths["T.h5"], "r") as translations_file,
    ):
        keys = sorted(matches_file)
        selected = sorted(np.random.default_rng(0).choice(len(keys), min(pairs, len(keys)), replace=False))
        for index in selected:
            pair = keys[index]
            matches = matches_file[pair][()]
            keep = confidence_file[pair][()].reshape(-1) <= match_threshold
            matches = np.ascontiguousarray(matches[keep], dtype=np.float64)
            image_ids = pair.split("-")
            intrinsics = intrinsics_file[pair][()].reshape(2, 3, 3)
            calibration = [
                {
                    "K": np.asarray(intrinsics[side], dtype=np.float64),
                    "R": np.asarray(rotations_file[image_ids[side]][()], dtype=np.float64),
                    "T": np.asarray(translations_file[image_ids[side]][()], dtype=np.float64).reshape(3),
                }
                for side in (0, 1)
            ]
            items.append(
                {
                    "pair": pair,
                    "a": np.ascontiguousarray(matches[:, :2]),
                    "b": np.ascontiguousarray(matches[:, 2:4]),
                    "matches": np.ascontiguousarray(matches[:, :4], dtype=np.float64),
                    "calib": calibration,
                }
            )
    return items, {name: digest(path) for name, path in paths.items()}


def pose_error(fundamental: np.ndarray, a: np.ndarray, b: np.ndarray, calib_a: dict, calib_b: dict) -> float:
    """Return max rotation/unsigned-translation relative-pose error in degrees."""
    if len(a) < 5:
        return float("inf")
    rotation_gt = calib_b["R"] @ calib_a["R"].T
    translation_gt = calib_b["T"] - rotation_gt @ calib_a["T"]
    if np.linalg.norm(translation_gt) < 1e-9:
        return float("inf")
    ka, kb = calib_a["K"], calib_b["K"]
    normalized_a = (a - ka[:2, 2]) / np.array([ka[0, 0], ka[1, 1]])
    normalized_b = (b - kb[:2, 2]) / np.array([kb[0, 0], kb[1, 1]])
    try:
        _, rotation, translation, _ = cv2.recoverPose(kb.T @ fundamental @ ka, normalized_a, normalized_b)
    except cv2.error:
        return float("inf")
    rotation_cos = np.clip((np.trace(rotation @ rotation_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    translation = translation.reshape(-1)
    translation /= np.linalg.norm(translation) + 1e-15
    translation_gt /= np.linalg.norm(translation_gt) + 1e-15
    translation_cos = np.clip(abs(translation @ translation_gt), 0.0, 1.0)
    return float(np.degrees(max(np.arccos(rotation_cos), np.arccos(translation_cos))))


def make_call(kornia_rs, kind: str, solver: str, item: dict, threshold_px: float, budget: int, seed: int, confidence: float | None = None):
    """Invoke one public solver, converting only the generic cutoff convention."""
    if kind == "generic":
        return kornia_rs.ransac.fundamental(
            item["matches"], threshold=threshold_px * threshold_px, max_iters=budget, confidence=0.999 if confidence is None else confidence, seed=seed, solver=solver
        )
    return kornia_rs.k3d.find_fundamental(
        item["a"], item["b"], method=8, ransac_threshold=threshold_px,
        max_iterations=budget, min_inliers=8, seed=seed, solver=solver,
        **({} if confidence is None else {"confidence": confidence}),
    )


def unpack(output, kind: str) -> tuple[np.ndarray | None, np.ndarray, int | None]:
    """Normalize the two public result shapes to F, inlier mask, iterations."""
    if kind == "generic":
        matrix = None if output.model is None else np.asarray(output.model, dtype=np.float64).reshape(3, 3)
        return matrix, np.asarray(output.inliers, dtype=bool), int(output.num_iters)
    matrix, mask = output
    return (None if matrix is None else np.asarray(matrix, dtype=np.float64).reshape(3, 3), np.asarray(mask, dtype=bool).reshape(-1), None)


def evaluate(item: dict, output, kind: str) -> dict:
    """Score one returned model outside the timed section."""
    matrix, inliers, iterations = unpack(output, kind)
    support = int(inliers.sum())
    error = float("inf")
    if matrix is not None and matrix.shape == (3, 3) and np.isfinite(matrix).all() and np.any(matrix) and len(inliers) == len(item["a"]) and support >= 5:
        error = pose_error(matrix, item["a"][inliers], item["b"][inliers], *item["calib"])
    return {"error_deg": None if not np.isfinite(error) else error, "support": support, "iterations": iterations}


def summarize(rows: list[dict]) -> list[dict]:
    """Create mAA/latency summaries while retaining every raw observation."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["threshold_mode"], row["backend"], row["budget"], row["threshold_px"])].append(row)
    result = []
    for (stage, mode, backend, budget, threshold), group in sorted(groups.items()):
        errors = np.asarray([float("inf") if row["error_deg"] is None else row["error_deg"] for row in group])
        timed = [row["median_ms"] for row in group if row["median_ms"] is not None]
        supports = [row["support"] for row in group]
        iterations = [row["iterations"] for row in group if row["iterations"] is not None]
        result.append({
            "stage": stage, "threshold_mode": mode, "backend": backend, "budget": budget, "threshold_px": threshold,
            "maa": float(np.mean(errors[:, None] < POSE_THRESHOLDS_DEG)),
            "mean_median_ms": float(np.mean(timed)) if timed else None,
            "failed": int(np.sum(~np.isfinite(errors))), "rows": len(group),
            "mean_support": float(np.mean(supports)),
            "mean_actual_iterations": float(np.mean(iterations)) if iterations else None,
        })
    return result


def chosen_thresholds(sweep_summary: list[dict]) -> dict[str, float]:
    """Pick the maximum-mAA threshold per solver, then latency then cutoff."""
    candidates: dict[str, list[dict]] = defaultdict(list)
    for row in sweep_summary:
        candidates[row["backend"]].append(row)
    return {name: min(values, key=lambda row: (-row["maa"], row["mean_median_ms"], row["threshold_px"]))["threshold_px"] for name, values in candidates.items()}


def run_stage(
    kornia_rs,
    items: list[dict],
    variants: list[tuple[str, str, str]],
    stage: str,
    configs: list[tuple[str, int, float]],
    seeds: list[int],
    rounds: int,
    rng: np.random.Generator,
    per_variant_thresholds: dict[str, float] | None = None,
    confidence: float | None = None,
) -> list[dict]:
    """Interleave variants while timing only repeated public RANSAC calls."""
    rows: list[dict] = []
    variant_by_name = {entry[0]: entry for entry in variants}
    for item_index, item in enumerate(items):
        for threshold_mode, budget, threshold_px in configs:
            active = list(variants)
            # Compile/allocate outside measurement for every identical public-call shape.
            for name, kind, solver in active:
                try:
                    cutoff = threshold_px if per_variant_thresholds is None else per_variant_thresholds[name]
                    make_call(kornia_rs, kind, solver, item, cutoff, budget, seeds[0], confidence)
                except (ValueError, RuntimeError):
                    pass
            samples: dict[tuple[str, int], list[float]] = defaultdict(list)
            outputs: dict[tuple[str, int], object] = {}
            exceptions: dict[tuple[str, int], str] = {}
            for _ in range(rounds):
                for seed in rng.permutation(seeds):
                    for name, kind, solver in rng.permutation(np.asarray(active, dtype=object)):
                        cutoff = threshold_px if per_variant_thresholds is None else per_variant_thresholds[name]
                        start = time.perf_counter_ns()
                        try:
                            output = make_call(kornia_rs, kind, solver, item, cutoff, budget, int(seed), confidence)
                        except (ValueError, RuntimeError) as exc:
                            output = None
                            exceptions[name, int(seed)] = f"{type(exc).__name__}: {exc}"
                        samples[name, int(seed)].append((time.perf_counter_ns() - start) / 1e6)
                        outputs[name, int(seed)] = output
            for (name, seed), timing in samples.items():
                _, kind, solver = variant_by_name[name]
                cutoff = threshold_px if per_variant_thresholds is None else per_variant_thresholds[name]
                fields = {"error_deg": None, "support": 0, "iterations": None}
                if outputs[name, seed] is not None:
                    fields = evaluate(item, outputs[name, seed], kind)
                rows.append({
                    "stage": stage, "threshold_mode": threshold_mode, "pair": item["pair"], "seed": seed,
                    "backend": name, "solver": solver, "api": kind, "budget": budget, "threshold_px": cutoff,
                    "n": len(item["a"]), "median_ms": float(np.median(timing)), "timing_samples_ms": timing,
                    "exception": exceptions.get((name, seed)), **fields,
                })
        print(f"{stage}: {item_index + 1}/{len(items)} {item['pair']} N={len(item['a'])}", flush=True)
    return rows


def package_version(name: str) -> str | None:
    """Return installed package version when the wheel carries metadata."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=34)
    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stages", default="sweep,budgets", help="comma-separated: sweep,budgets,matched; matched needs no sweep")
    parser.add_argument("--confidence", type=float, help="common sampling confidence for both engines and solvers")
    parser.add_argument("--matched-threshold", type=float, default=0.5)
    parser.add_argument("--budgets", default="64,256,1000,4096")
    args = parser.parse_args()
    if not 0 < args.match_threshold <= 1 or args.rounds < 3 or args.threads < 1:
        parser.error("match threshold must be in (0,1], rounds >= 3, and threads >= 1")
    stages = set(args.stages.split(","))
    if not stages <= {"sweep", "budgets", "matched"} or not stages:
        parser.error("--stages must contain only sweep, budgets, and/or matched")
    if args.confidence is not None and not 0 < args.confidence < 1:
        parser.error("--confidence must be finite and in (0,1)")
    seeds = [int(value) for value in args.seeds.split(",")]
    budgets = [int(value) for value in args.budgets.split(",")]
    if not seeds or min(seeds) < 0 or not budgets or min(budgets) < 1:
        parser.error("seeds must be non-negative and budgets positive")
    import kornia_rs

    cv2.setNumThreads(args.threads)
    items, input_files = load_items(args.data_root, args.pairs, args.match_threshold)
    root = Path(__file__).resolve().parents[2]
    extension = next(Path(kornia_rs.__file__).parent.glob("kornia_rs*.so"), Path(kornia_rs.__file__))
    source_paths = [
        root / "crates/kornia-3d/src/ransac/estimators/fundamental.rs",
        root / "crates/kornia-3d/src/ransac/driver.rs",
        root / "crates/kornia-3d/src/ransac/mod.rs",
        root / "crates/kornia-3d/src/ransac/config.rs",
        root / "crates/kornia-3d/src/pose/fundamental.rs",
        root / "crates/kornia-3d/src/pose/fundamental_7pt.rs",
        root / "crates/kornia-3d/src/pose/twoview.rs",
        root / "kornia-py/src/ransac.rs",
        root / "kornia-py/src/homography.rs",
        root / "kornia-py/src/twoview.rs",
    ]
    metadata = {
        "dataset": "CVPR-2020 RANSAC tutorial / st_peters_square",
        "selection": "sorted(default_rng(0).choice(keys, min(pairs, len(keys)), replace=False))",
        "pairs_requested": args.pairs, "pairs_selected": len(items), "match_snn_cutoff": args.match_threshold,
        "input_files_sha256": input_files, "selected_input_sha256": selected_digest(items),
        "thresholds_px": list(THRESHOLDS), "matched_threshold_px": args.matched_threshold,
        "budgets": budgets, "seeds": seeds, "rounds": args.rounds, "threads": args.threads,
        "confidence": {"generic": 0.999 if args.confidence is None else args.confidence,
                       "k3d": 0.9999 if args.confidence is None else args.confidence},
        "timing": f"median of {args.rounds} interleaved public calls; input construction, warmup and quality excluded",
        "quality": "OpenCV recoverPose on returned F and inliers; max(rotation, sign-invariant translation) error; strict mAA thresholds 1..10 degrees",
        "generic_threshold": "squared physical pixel threshold", "k3d_threshold": "physical pixel threshold",
        "variants": [entry[0] for entry in VARIANTS], "python": sys.version, "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("kornia-rs", "numpy", "h5py", "opencv-python")},
        "rust_extension": str(extension), "rust_extension_sha256": digest(extension),
        "harness_sha256": digest(Path(__file__)),
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in source_paths if path.is_file()},
    }
    rng = np.random.default_rng(8721)
    rows: list[dict] = []
    selected: dict[str, float] = {}
    if "sweep" in stages:
        rows.extend(run_stage(kornia_rs, items, list(VARIANTS), "sweep", [("sweep", 1000, value) for value in THRESHOLDS], seeds, args.rounds, rng, confidence=args.confidence))
        selected = chosen_thresholds(summarize(rows))
    if "budgets" in stages:
        if not selected:
            parser.error("budget stage needs this invocation's sweep stage so thresholds are selected from the same run")
        for budget in budgets:
            # The matched-threshold ablation keeps all four calls interleaved.
            rows.extend(
                run_stage(
                    kornia_rs, items, list(VARIANTS), "budgets",
                    [("matched", budget, args.matched_threshold)], seeds, args.rounds, rng,
                    confidence=args.confidence,
                )
            )
            # The tuned protocol has one independently selected physical cutoff per solver,
            # while its calls remain interleaved with every other solver.
            rows.extend(
                run_stage(
                    kornia_rs, items, list(VARIANTS), "budgets", [("tuned", budget, 0.0)],
                    seeds, args.rounds, rng, per_variant_thresholds=selected,
                    confidence=args.confidence,
                )
            )
    elif "matched" in stages:
        for budget in budgets:
            rows.extend(run_stage(
                kornia_rs, items, list(VARIANTS), "budgets",
                [("matched", budget, args.matched_threshold)], seeds, args.rounds, rng,
                confidence=args.confidence,
            ))
    payload = {"metadata": metadata, "chosen_thresholds_px": selected, "results": rows, "summary": summarize(rows)}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for row in payload["summary"]:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
