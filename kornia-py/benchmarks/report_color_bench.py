"""Merge kornia (Rust) + OpenCV/VPI (Python) color benchmark JSON into tables.

Inputs are JSON-lines files produced by:
  - `cargo run --example bench_gpu_color_conversions --features gpu-cuda \
     --release -- --json > kornia.jsonl`
  - `python3 bench_color_vs_libs.py --json libs.jsonl`

Output: one markdown table per size — rows are conversions, columns are
variants (min_ms). The final column shows kornia-CUDA(kernel) speedup vs the
best competitor for that op; cells missing a measurement print `-`.

Usage:
    python3 report_color_bench.py kornia.jsonl libs.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

COLUMNS = [
    "kornia-cpu",
    "kornia-cuda-kernel",
    "kornia-cuda-e2e",
    "kornia-cuda-e2e-pinned",
    "opencv-cpu",
    "vpi-cpu",
    "vpi-cuda",
]
COMPETITORS = ["opencv-cpu", "vpi-cpu", "vpi-cuda"]


def main(paths):
    # data[(w,h)][op][variant] = min_ms
    data = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                # Tolerate stray non-JSON lines (cargo/compiler noise).
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                data[(r["width"], r["height"])][r["op"]][r["variant"]] = r["min_ms"]

    for (w, h) in sorted(data):
        ops = data[(w, h)]
        fused_ops = {k: v for k, v in ops.items() if k.startswith("preprocess_")}
        ops = {k: v for k, v in ops.items() if not k.startswith("preprocess_")}
        print(f"\n## {w}x{h} (min ms per call)\n")
        header = ["op", *COLUMNS, "cuda vs best-lib"]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for op in sorted(ops):
            cells = [op]
            for col in COLUMNS:
                v = ops[op].get(col)
                cells.append(f"{v:.4f}" if v is not None else "-")
            cuda = ops[op].get("kornia-cuda-kernel")
            best_lib = min(
                (ops[op][c] for c in COMPETITORS if c in ops[op]), default=None
            )
            if cuda and best_lib:
                cells.append(f"{best_lib / cuda:.2f}x")
            else:
                cells.append("-")
            print("| " + " | ".join(cells) + " |")

        if fused_ops:
            print(f"\n**Fused camera preprocessing** (frame → 640×640 CHW tensor, one kernel)\n")
            print("| pipeline | fused | chained (decode + preprocess) | speedup |")
            print("|---|---|---|---|")
            for op in sorted(fused_ops):
                f = fused_ops[op].get("fused")
                c = fused_ops[op].get("chained")
                if f and c:
                    print(f"| {op} | **{f:.4f}** | {c:.4f} | {c / f:.2f}x |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
