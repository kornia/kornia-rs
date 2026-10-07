"""Prepare deterministic requests for the native fundamental policy audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench_fundamental_solvers import load_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", default="64,256,1000,4096,16384")
    parser.add_argument("--modes", default="count_lo")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.999999)
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or args.rounds < 1:
        parser.error("confidence must be in (0,1) and rounds positive")
    items, _ = load_items(args.data_root, 34, 0.85)
    with args.output.open("w") as output:
        for budget in map(int, args.budgets.split(",")):
            for item in items:
                for seed in range(3):
                    request = {"pair": item["pair"], "matches": np.c_[item["a"], item["b"]].tolist(),
                               "seed": seed, "budget": budget, "confidence": args.confidence,
                               "modes": args.modes.split(","), "rounds": args.rounds}
                    output.write(json.dumps(request, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
