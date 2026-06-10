#!/usr/bin/env python3
"""Baseline: upstream XFeat exported to ONNX, run through ONNX Runtime CPU EP.

See pytorch_eager.py for the driver contract.
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
    ap.add_argument("--onnx-model", type=Path, required=True,
                    help="Path to a pre-exported xfeat.onnx (see ../README.md for export flow).")
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    img = Image.open(args.fixture).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0
    h, w = img_np.shape
    h_a = (h // 32) * 32
    w_a = (w // 32) * 32
    img_np = img_np[:h_a, :w_a][np.newaxis, np.newaxis, :, :]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # match Rust single-thread measurement
    session = ort.InferenceSession(str(args.onnx_model), sess_options=so,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(args.warmup):
        session.run(None, {input_name: img_np})

    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: img_np})
        samples.append((time.perf_counter() - t0) * 1000.0)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1024.0

    result = {
        "baseline": "onnxruntime",
        "platform": args.platform,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(0.95 * len(samples))],
        "p99_ms": sorted(samples)[int(0.99 * len(samples))],
        "iterations": args.iters,
        "warmup": args.warmup,
        "rss_mb": rss_mb,
        "env": {
            "onnxruntime": ort.__version__,
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "kernel": platform.release(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[onnxruntime] median={result['median_ms']:.2f}ms p95={result['p95_ms']:.2f}ms -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
