#!/usr/bin/env python3
"""Interleaved, same-process comparison of frozen and optimized Rust solvers.

Keep both extension binaries before installing a new wheel. This loads each
under a distinct package name and uses the frozen experiment's cutoffs and
draw budgets. No threshold tuning is performed during this speed comparison.

Example::

    python kornia-py/benchmarks/bench_fundamental_optimization.py \
      --before /tmp/fundamental-opt/before.so --after /tmp/fundamental-opt/after.so \
      --data-root /Users/oldufo/dev/pydegensac/benchmarks/data \
      --reference docs/fundamental-7pt/results.json --json /tmp/optimization.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench_fundamental_solvers import digest, evaluate, load_items, make_call, selected_digest, unpack


def load_extension(name: str, path: Path):
    """Load one PyO3 extension while retaining its required PyInit module name."""
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".kornia_rs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_call(module, *arguments):
    """Represent an expected RANSAC failure without losing its timing."""
    try:
        return make_call(module, *arguments), None
    except (ValueError, RuntimeError) as error:
        kind, _, item, _, _, _ = arguments
        mask = np.zeros(len(item["a"]), dtype=bool)
        output = (None, mask) if kind == "k3d" else types.SimpleNamespace(model=None, inliers=mask, num_iters=0)
        return output, str(error)


def projectively_equal(first, second):
    """Fundamental matrices represent the same geometry up to scale and sign."""
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return np.allclose(first, second, rtol=1e-8, atol=1e-10) or np.allclose(first, -second, rtol=1e-8, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    before = load_extension("frozen", args.before.resolve())
    after = load_extension("optimized", args.after.resolve())
    reference = json.loads(args.reference.read_text())
    items, hashes = load_items(args.data_root, 34, 0.85)
    by_pair = {item["pair"]: item for item in items}
    rows, changes = [], []
    # Replay every prior quality observation, including the complete cutoff
    # sweep. Also time the budget rows, alternating which binary runs first.
    for index, old_row in enumerate(reference["results"]):
        item = by_pair[old_row["pair"]]
        kind = "generic" if "generic" in old_row["backend"] else "k3d"
        solver = "7point" if "F7" in old_row["backend"] else "8point"
        call_args = (kind, solver, item, old_row["threshold_px"], old_row["budget"], old_row["seed"])
        checked = [checked_call(module, *call_args) for module in (before, after)]
        outputs, exceptions = zip(*checked)
        quality = [evaluate(item, output, kind) for output in outputs]
        matrices, masks, iterations = zip(*(unpack(output, kind) for output in outputs))
        same_matrix = (
            matrices[0] is None and matrices[1] is None
            or matrices[0] is not None and matrices[1] is not None
            and projectively_equal(*matrices)
        )
        same_mask = np.array_equal(*masks)
        same_iterations = iterations[0] == iterations[1]
        if not same_matrix or not same_mask or not same_iterations or exceptions[0] != exceptions[1]:
            changes.append({
                "index": index, "pair": old_row["pair"], "backend": old_row["backend"],
                "budget": old_row["budget"], "threshold_px": old_row["threshold_px"],
                "seed": old_row["seed"], "same_matrix": bool(same_matrix),
                "same_mask": bool(same_mask), "same_iterations": same_iterations,
                "before": quality[0], "after": quality[1],
            })
        times = [[], []]
        if old_row["stage"] == "budgets":
            for round_index in range(args.rounds):
                for side in ((0, 1) if (index + round_index) % 2 == 0 else (1, 0)):
                    start = time.perf_counter_ns()
                    checked_call((before, after)[side], *call_args)
                    times[side].append((time.perf_counter_ns() - start) / 1e6)
        rows.append({
            **{key: old_row[key] for key in ("stage", "threshold_mode", "pair", "backend", "budget", "threshold_px", "seed")},
            "before": quality[0], "after": quality[1], "before_ms": times[0], "after_ms": times[1],
            "exceptions": exceptions,
        })
        if index % 400 == 0:
            print(f"{index}/{len(reference['results'])} quality rows; {len(changes)} changed outputs", flush=True)
    groups = defaultdict(list)
    for row in rows:
        if row["before_ms"]:
            groups[(row["backend"], row["threshold_mode"], row["budget"], row["threshold_px"])].append(row)
    summary = []
    for (backend, mode, budget, threshold), group in sorted(groups.items()):
        record = {"backend": backend, "threshold_mode": mode, "budget": budget, "threshold_px": threshold, "rows": len(group)}
        for side in ("before", "after"):
            record[side + "_mean_median_ms"] = float(np.mean([np.median(row[side + "_ms"]) for row in group]))
            errors = np.array([float("inf") if row[side]["error_deg"] is None else row[side]["error_deg"] for row in group])
            record[side + "_maa"] = float(np.mean(errors[:, None] < np.arange(1, 11)))
        record["speedup"] = record["before_mean_median_ms"] / record["after_mean_median_ms"]
        summary.append(record)
    payload = {
        "metadata": {
            "platform": platform.platform(), "python": sys.version, "rounds": args.rounds,
            "before_sha256": digest(args.before), "after_sha256": digest(args.after),
            "script_sha256": digest(Path(__file__)), "reference_sha256": digest(args.reference),
            "input_sha256": selected_digest(items), "data_sha256": hashes,
            "method": "same-process alternating A/B; frozen thresholds, seeds, draw budgets and match order",
        },
        "summary": summary, "changes": changes, "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
