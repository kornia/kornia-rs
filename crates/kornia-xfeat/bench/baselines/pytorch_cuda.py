#!/usr/bin/env python3
"""Baseline: upstream XFeat unmodified on CUDA GPU.

Driver contract documented in bench/baselines/README.md.

Jetson-specific note: on Jetson Orin the GPU is integrated (unified memory).
rss_mb via resource.getrusage reflects CPU-side RSS only; it under-counts true
GPU memory usage because the CUDA UVM allocations do not always appear in the
Linux RSS counter. This is expected behaviour on Tegra/iGPU platforms.
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
    ap = argparse.ArgumentParser(
        description="Benchmark upstream XFeat on CUDA (PyTorch eager)."
    )
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination JSON file (parent dirs created automatically).")
    ap.add_argument("--upstream", type=Path, required=True,
                    help="Path to an existing accelerated_features checkout "
                         "(cloned by regen.py or manually).")
    ap.add_argument("--fixture", type=Path, required=True,
                    help="Grayscale or RGB test image, e.g. tests/fixtures/v1/input.png.")
    ap.add_argument("--platform", required=True,
                    help="Platform tag written verbatim into the JSON, "
                         "e.g. 'jetson-orin-cuda'.")
    ap.add_argument("--warmup", type=int, default=100,
                    help="Number of warm-up iterations (not timed). Default 100.")
    ap.add_argument("--iters", type=int, default=1000,
                    help="Number of timed iterations. Default 1000.")
    ap.add_argument("--device", default="cuda:0",
                    help="PyTorch device string. Default 'cuda:0'.")
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Imports — torch must come after sys.path manipulation so the upstream
    # modules package is importable.
    # ------------------------------------------------------------------ #
    sys.path.insert(0, str(args.upstream))
    import torch
    import numpy as np
    from PIL import Image
    from modules.xfeat import XFeat  # type: ignore[import]

    # Guard: this driver targets CUDA only.  Exit code 2 lets the harness
    # distinguish "CUDA not available" from a crash (exit code 1 / exception).
    if not torch.cuda.is_available():
        print(
            "[pytorch_cuda] ERROR: torch.cuda.is_available() returned False. "
            "Install a CUDA-enabled PyTorch build and rerun.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ------------------------------------------------------------------ #
    # Model setup
    # ------------------------------------------------------------------ #
    weights_path = str(args.upstream / "weights" / "xfeat.pt")
    xfeat = XFeat(weights=weights_path)
    xfeat.cuda().eval()

    # ------------------------------------------------------------------ #
    # Pre-process: grayscale → float32 tensor, aligned to 32-px grid
    # ------------------------------------------------------------------ #
    img_pil = Image.open(args.fixture).convert("L")
    img_np = np.array(img_pil, dtype=np.uint8)
    h, w = img_np.shape
    h_a = (h // 32) * 32
    w_a = (w // 32) * 32

    with torch.no_grad():
        x = (
            torch.from_numpy(img_np)  # type: ignore[attr-defined]
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )
        x = torch.nn.functional.interpolate(
            x, size=(h_a, w_a), mode="bilinear", align_corners=False
        )
        x = x.to(args.device)

        # ------------------------------------------------------------------ #
        # Warm-up: allow the JIT, cuDNN auto-tuner, and CUDA driver to settle.
        # ------------------------------------------------------------------ #
        for _ in range(args.warmup):
            xfeat.detectAndCompute(x, top_k=4096)
            torch.cuda.synchronize()

        # ------------------------------------------------------------------ #
        # Timed measurement
        # synchronize() before t0 ensures GPU is idle when we start the clock;
        # synchronize() after the call ensures GPU work is complete before we
        # stop the clock.  Without this, perf_counter measures only kernel
        # *launch* latency, not actual execution time.
        # ------------------------------------------------------------------ #
        samples: list[float] = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            xfeat.detectAndCompute(x, top_k=4096)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

    # RSS in Linux kibibytes → MB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0)

    result = {
        "baseline": "pytorch_cuda",
        "platform": args.platform,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(0.95 * len(samples))],
        "p99_ms": sorted(samples)[int(0.99 * len(samples))],
        "iterations": args.iters,
        "warmup": args.warmup,
        "rss_mb": rss_mb,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "kernel": platform.release(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"[pytorch_cuda] median={result['median_ms']:.2f}ms "
        f"p95={result['p95_ms']:.2f}ms "
        f"gpu={result['env']['gpu']} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
