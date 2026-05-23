#!/usr/bin/env python3
"""Baseline: upstream XFeat unmodified on CPU.

Driver contract documented in ../README.md.
"""

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True,
                    help="Path to an existing accelerated_features checkout (cloned by regen.py).")
    ap.add_argument("--fixture", type=Path, required=True,
                    help="tests/fixtures/v1/input.png")
    ap.add_argument("--platform", required=True,
                    help="Platform tag (e.g. 'jetson-orin', 'x86_64-avx2').")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    sys.path.insert(0, str(args.upstream))
    import torch
    import numpy as np
    from PIL import Image
    from modules.xfeat import XFeat  # type: ignore

    xfeat = XFeat(weights=str(args.upstream / "weights" / "xfeat.pt"))
    getattr(xfeat, "eval")()

    img = Image.open(args.fixture).convert("L")
    img_np = np.array(img, dtype=np.uint8)
    h, w = img_np.shape
    h_a = (h // 32) * 32
    w_a = (w // 32) * 32

    with torch.no_grad():
        x = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        x = torch.nn.functional.interpolate(x, size=(h_a, w_a), mode="bilinear", align_corners=False)

        # Warmup.
        for _ in range(args.warmup):
            xfeat.detectAndCompute(x, top_k=4096)

        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            xfeat.detectAndCompute(x, top_k=4096)
            samples.append((time.perf_counter() - t0) * 1000.0)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns kibibytes; macOS returns bytes. Normalize to MB.
    rss_mb = rss / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0)

    result = {
        "baseline": "pytorch_eager",
        "platform": args.platform,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(0.95 * len(samples))],
        "p99_ms": sorted(samples)[int(0.99 * len(samples))],
        "iterations": args.iters,
        "warmup": args.warmup,
        "rss_mb": rss_mb,
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "kernel": platform.release(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[pytorch_eager] median={result['median_ms']:.2f}ms p95={result['p95_ms']:.2f}ms -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
