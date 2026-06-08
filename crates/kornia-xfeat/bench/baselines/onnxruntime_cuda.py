#!/usr/bin/env python3
"""Baseline: upstream XFeat ONNX model through ONNX Runtime CUDA EP.

Driver contract documented in bench/baselines/README.md.

Pre-requisite — export the ONNX model once before running this script:

    python3 - <<'EOF'
    import sys, torch
    sys.path.insert(0, '<upstream>')
    from modules.xfeat import XFeat
    model = XFeat(weights='<upstream>/weights/xfeat.pt').eval()
    dummy = torch.zeros(1, 1, 480, 640)
    torch.onnx.export(
        model, (dummy,),
        'xfeat_480x640.onnx',
        opset_version=17,
        input_names=['image'],
        output_names=['feats', 'keypoints', 'reliability'],
        dynamic_axes={'image': {2: 'H', 3: 'W'}},
    )
    EOF

Notes:
- XFeat's forward() returns (feats, k1, h1), NOT post-processed keypoints;
  the ONNX graph covers only the neural-network forward pass (no NMS / sampling).
  This is consistent with pytorch_cuda.py's detectAndCompute call, which wraps
  the same forward pass.
- opset_version=17 is required because aten::unfold is used internally; earlier
  opsets do not support it and the export will fail.
- If the exported model contains ops that the CUDA EP does not support, ONNX
  Runtime will silently fall back those individual nodes to the CPU EP.  The
  benchmark is still valid in that case; the measured latency will simply reflect
  the mixed-device execution graph.
- Jetson Orin note: the CUDAExecutionProvider on Jetson targets the integrated
  GPU (iGPU).  There is no separate discrete GPU memory — CUDA UVM is shared
  with system RAM.  rss_mb from resource.getrusage therefore under-counts true
  GPU memory usage on this platform, exactly as in pytorch_cuda.py.
- Jetson wheel: use the onnxruntime-gpu wheel built against CUDA 11.4+ from the
  JetPack 5.x feed (not the x86 pypi wheel).
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
        description="Benchmark pre-exported XFeat ONNX model via ONNX Runtime CUDA EP."
    )
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination JSON file (parent dirs created automatically).")
    ap.add_argument("--onnx-model", type=Path, required=True,
                    help="Path to a pre-exported xfeat.onnx "
                         "(see module docstring for the export command).")
    ap.add_argument("--fixture", type=Path, required=True,
                    help="Grayscale or RGB test image, e.g. tests/fixtures/v1/input.png.")
    ap.add_argument("--platform", required=True,
                    help="Platform tag written verbatim into the JSON, "
                         "e.g. 'jetson-orin-cuda'.")
    ap.add_argument("--warmup", type=int, default=100,
                    help="Number of warm-up iterations (not timed). Default 100.")
    ap.add_argument("--iters", type=int, default=1000,
                    help="Number of timed iterations. Default 1000.")
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Imports
    # ------------------------------------------------------------------ #
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    # Guard: this driver targets the CUDA EP only.  Exit code 2 lets the
    # harness distinguish "CUDA EP not available" from a crash (exit code 1).
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print(
            "[onnxruntime_cuda] ERROR: CUDAExecutionProvider is not available in "
            f"this onnxruntime build ({ort.__version__}). "
            "Install the onnxruntime-gpu wheel and rerun.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ------------------------------------------------------------------ #
    # Pre-process: match pytorch_cuda.py's interpolation exactly.
    # PIL.Image.resize uses BILINEAR on the L-channel image, then we convert
    # to float32 and add the batch + channel dimensions.
    # ------------------------------------------------------------------ #
    img_pil = Image.open(args.fixture).convert("L")
    h, w = img_pil.height, img_pil.width
    h_a = (h // 32) * 32
    w_a = (w // 32) * 32

    # PIL resize: note argument order is (width, height).
    img_resized = img_pil.resize((w_a, h_a), Image.Resampling.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    # Shape: (1, 1, h_a, w_a) — NCHW expected by XFeat ONNX graph.
    img_np = img_np[np.newaxis, np.newaxis, :, :]

    # ------------------------------------------------------------------ #
    # Build ONNX Runtime session with CUDA EP as first provider.
    # ORT falls back to CPU EP only for nodes the CUDA EP cannot handle;
    # step 2 above already ensures the CUDA EP is present in this build.
    # intra_op_num_threads=1 matches the single-thread Rust measurement.
    # ------------------------------------------------------------------ #
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.onnx_model),
        sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # ------------------------------------------------------------------ #
    # Warm-up
    # ------------------------------------------------------------------ #
    for _ in range(args.warmup):
        session.run(None, {input_name: img_np})

    # ------------------------------------------------------------------ #
    # Timed measurement
    # ORT's CUDAExecutionProvider is synchronous from Python: session.run()
    # blocks until all GPU kernels have completed, so no explicit
    # CUDA synchronize call is needed here.
    # ------------------------------------------------------------------ #
    samples: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: img_np})
        samples.append((time.perf_counter() - t0) * 1000.0)

    # RSS in Linux kibibytes → MB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0)

    result = {
        "baseline": "onnxruntime_cuda",
        "platform": args.platform,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[int(0.95 * len(samples))],
        "p99_ms": sorted(samples)[int(0.99 * len(samples))],
        "iterations": args.iters,
        "warmup": args.warmup,
        "rss_mb": rss_mb,
        "env": {
            "onnxruntime": ort.__version__,
            "cuda_ep": "CUDAExecutionProvider",
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "kernel": platform.release(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"[onnxruntime_cuda] median={result['median_ms']:.2f}ms "
        f"p95={result['p95_ms']:.2f}ms "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
