#!/usr/bin/env python3
"""Compare frozen and optimized scoring with exact seeded output checks.

Uses 34 real pairs, three seeds, a shared 0.5 px threshold and confidence
0.999999. Timings include Python dispatch and alternate the two binaries.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from bench_fundamental_optimization import load_extension
from bench_fundamental_solvers import load_items, make_call, selected_digest, unpack

BUDGETS = (64, 256, 1000, 4096, 16384)


def invoke(module, call_args):
    """Retain expected RANSAC failures for equivalence and timing checks."""
    try:
        return make_call(module, *call_args, confidence=0.999999), None
    except (ValueError, RuntimeError) as error:
        return None, str(error)


def identical_outputs(results, kind):
    """Compare matrices, masks, draw counts and exception messages exactly."""
    if results[0][1] != results[1][1]:
        return False
    if any(result[0] is None for result in results):
        return all(result[0] is None for result in results)
    first, second = (unpack(result[0], kind) for result in results)
    same_matrix = (
        first[0] is None and second[0] is None
        or first[0] is not None and second[0] is not None
        and np.array_equal(first[0], second[0])
    )
    return same_matrix and np.array_equal(first[1], second[1]) and first[2] == second[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    modules = [
        load_extension("scoring_before", args.before.resolve()),
        load_extension("scoring_after", args.after.resolve()),
    ]
    items, hashes = load_items(args.data_root, 34, 0.85)
    rows, changes = [], []
    for budget in BUDGETS:
        for item in items:
            for seed in range(3):
                for kind in ("generic", "k3d"):
                    for solver in ("7point", "8point"):
                        call_args = (kind, solver, item, 0.5, budget, seed)
                        results = [invoke(module, call_args) for module in modules]
                        same = identical_outputs(results, kind)
                        entry = {
                            "pair": item["pair"], "seed": seed, "engine": kind,
                            "solver": solver, "budget": budget,
                            "identical": bool(same), "times_ms": [[], []],
                        }
                        if not same:
                            changes.append({
                                key: value for key, value in entry.items()
                                if key != "times_ms"
                            })
                        for repetition in range(args.rounds):
                            order = (0, 1) if (seed + repetition) % 2 == 0 else (1, 0)
                            for side in order:
                                start = time.perf_counter_ns()
                                invoke(modules[side], call_args)
                                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                                entry["times_ms"][side].append(elapsed_ms)
                        rows.append(entry)
        print(f"budget {budget}: {len(rows)} cases, {len(changes)} changes", flush=True)

    summary = []
    for kind in ("generic", "k3d"):
        for solver in ("7point", "8point"):
            for budget in BUDGETS:
                selected = [
                    row for row in rows if row["engine"] == kind
                    and row["solver"] == solver and row["budget"] == budget
                ]
                means = [
                    statistics.mean(
                        statistics.median(row["times_ms"][side]) for row in selected
                    ) for side in (0, 1)
                ]
                summary.append({
                    "engine": kind, "solver": solver, "budget": budget,
                    "before_ms": means[0], "after_ms": means[1],
                    "speedup": means[0] / means[1],
                })
    report = {
        "metadata": {
            "input_sha256": selected_digest(items), "data_sha256": hashes,
            "confidence": 0.999999, "threshold_px": 0.5, "seeds": [0, 1, 2],
            "rounds": args.rounds, "platform": platform.platform(),
            "measurement": "same-process alternating calls; mean per-case median; includes Python FFI",
            "binary_sha256": [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (args.before, args.after)
            ],
        },
        "summary": summary, "changes": changes, "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if changes:
        raise SystemExit(f"Changed outputs: {changes[:5]}")


if __name__ == "__main__":
    main()
