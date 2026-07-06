"""Merge kornia (Rust) + OpenCV/VPI (Python) color benchmark JSON into tables.

Inputs are JSON-lines files produced by:
  - `cargo run --example bench_gpu_color_conversions --features cuda \
     --release -- --json > kornia.jsonl`
  - `python3 bench_color_vs_libs.py --json libs.jsonl`

Output: one markdown table per size — rows are conversions, columns are
variants (min_ms). The final column shows kornia-CUDA(kernel) speedup vs the
best competitor for that op; cells missing a measurement print `-`.

Usage:
    python3 report_color_bench.py kornia.jsonl libs.jsonl
"""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from collections import defaultdict


def _run(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return ""


def system_info_header():
    """Markdown provenance block: everything needed to attribute the numbers
    to a specific machine, software stack, and power state."""
    rows = []

    def add(k, v):
        if v:
            rows.append((k, v))

    add("Date (UTC)", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"))
    add("Host", platform.node())
    add("Machine", _run("cat /proc/device-tree/model 2>/dev/null | tr -d '\\0'"))
    add("Kernel / arch", f"{platform.release()} {platform.machine()}")
    add("CPU", _run("lscpu | grep -m1 'Model name' | sed 's/.*: *//'")
        + f" x{_run('nproc')}")
    add("L4T", _run("head -c 200 /etc/nv_tegra_release 2>/dev/null | head -1"))
    add("GPU", _run("cat /sys/devices/platform/*.gpu/of_node/compatible 2>/dev/null | tr -d '\\0'")
        or "integrated (see L4T)")
    add("CUDA", _run(
        "python3 -c \"import json;print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'])\" 2>/dev/null"
    ))
    add("Power mode", _run("nvpmodel -q 2>/dev/null | grep -m1 'NV Power Mode' | sed 's/.*: *//'"))
    add("rustc", _run("rustc -V"))
    add("OpenCV (py)", _run("python3 -c 'import cv2;print(cv2.__version__)' 2>/dev/null"))
    add("VPI", _run(
        "python3 -c \"import glob,sys;sys.path.insert(0,next(iter(glob.glob('/opt/nvidia/vpi*/lib/*/python')),''));import vpi;print(vpi.__version__)\" 2>/dev/null"
    ))
    add("Git commit", _run("git rev-parse --short=10 HEAD 2>/dev/null")
        + (" (dirty)" if _run("git status --porcelain 2>/dev/null") else ""))

    out = ["# CUDA color conversion benchmark", "", "## System", "",
           "| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return chr(10).join(out) + chr(10)

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

    print(system_info_header())

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
