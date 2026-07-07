"""Color-conversion benchmark: OpenCV (CPU) and NVIDIA VPI (CPU + CUDA).

Companion to the Rust example `bench_cuda_color_conversions` (kornia CPU/CUDA
numbers). Emits the SAME JSON-lines schema so `report_color_bench.py` can merge
both:

    {"op": ..., "width": ..., "height": ..., "variant": ...,
     "min_ms": ..., "p50_ms": ..., "p95_ms": ...}

Variants here: `opencv-cpu`, `vpi-cpu`, `vpi-cuda`.

Fairness protocol (mirrors the Rust side):
  - identical sizes and a fixed-seed uint8 source
  - warmup, then every call timed individually; report min/p50/p95
  - VPI: the image is wrapped once and converted once as warmup so the H2D
    upload is cached; each timed call is convert + stream sync — comparable to
    kornia's `cuda-kernel` (device-resident) variant, plus VPI driver overhead.
  - value conventions differ across libraries (e.g. kornia HSV scales H from
    [0,360) to [0,255]; cv2 uses [0,180] for u8) — timings are still
    comparable; outputs are NOT asserted equal here.

Usage:
    python3 bench_color_vs_libs.py [--json out.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# ── VPI probe (same pattern as bench_orb.py) ─────────────────────────────────
_vpi_path = os.environ.get("KORNIA_VPI_PYPATH") or next(
    iter(glob.glob("/opt/nvidia/vpi*/lib/*/python")), None
)
if _vpi_path:
    sys.path.insert(0, _vpi_path)
try:
    import vpi

    HAVE_VPI = True
except ImportError:
    vpi = None
    HAVE_VPI = False

try:
    import cv2

    HAVE_CV2 = True
except ImportError:
    cv2 = None
    HAVE_CV2 = False

SIZES = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "min_ms": samples[0],
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[min(int(len(samples) * 0.95), len(samples) - 1)],
    }


def emit(rows, op, w, h, variant, stats):
    rows.append(
        {"op": op, "width": w, "height": h, "variant": variant, **stats}
    )
    print(
        f"{op:>26} {w}x{h:<9} {variant:>12} "
        f"min {stats['min_ms']:9.4f} ms  p50 {stats['p50_ms']:9.4f} ms"
    )


def cv2_cases(rgb, rgba, gray, rgb_f32, w, h):
    """(op, callable) pairs for the installed OpenCV (CPU-only build)."""
    n = w * h
    yuyv = np.random.default_rng(42).integers(0, 256, (h, w, 2), dtype=np.uint8)
    nv12 = np.random.default_rng(42).integers(
        0, 256, (h * 3 // 2, w), dtype=np.uint8
    )
    sepia_m = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        dtype=np.float32,
    )
    return [
        ("gray_from_rgb_u8", lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)),
        ("bgr_from_rgb_u8", lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
        ("rgba_from_rgb_u8", lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)),
        ("rgb_from_rgba_u8", lambda: cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)),
        ("ycbcr_from_rgb_u8", lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)),
        ("rgb_from_ycbcr_u8", lambda: cv2.cvtColor(rgb, cv2.COLOR_YCrCb2RGB)),
        ("sepia_from_rgb_u8", lambda: cv2.transform(rgb, sepia_m)),
        ("apply_colormap_jet_u8", lambda: cv2.applyColorMap(gray, cv2.COLORMAP_JET)),
        (
            "rgb_from_bayer_rggb_u8",
            lambda: cv2.cvtColor(gray, cv2.COLOR_BayerBG2RGB),
        ),
        ("rgb_from_yuyv_u8", lambda: cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)),
        ("rgb_from_nv12_u8", lambda: cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)),
        ("hsv_from_rgb_f32", lambda: cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2HSV)),
        ("lab_from_rgb_f32", lambda: cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2Lab)),
    ]


def vpi_cases(rgb, w, h):
    """(op, src_format, dst_format) triples VPI can express.

    VPI's format-conversion coverage is narrow; unsupported combos are probed
    with try/except and skipped (reported as N/A downstream).
    """
    return [
        ("gray_from_rgb_u8", vpi.Format.RGB8, vpi.Format.Y8_ER),
        ("bgr_from_rgb_u8", vpi.Format.RGB8, vpi.Format.BGR8),
        ("rgba_from_rgb_u8", vpi.Format.RGB8, vpi.Format.RGBA8),
        ("nv12_from_rgb_u8", vpi.Format.RGB8, vpi.Format.NV12_ER),
        ("rgb_from_nv12_u8", vpi.Format.NV12_ER, vpi.Format.RGB8),
    ]


def run_vpi(rows, rgb, w, h):
    for backend, variant in [
        (vpi.Backend.CPU, "vpi-cpu"),
        (vpi.Backend.CUDA, "vpi-cuda"),
    ]:
        for op, src_fmt, dst_fmt in vpi_cases(rgb, w, h):
            try:
                if src_fmt == vpi.Format.RGB8:
                    src = vpi.asimage(rgb, vpi.Format.RGB8)
                else:
                    # Build the non-RGB source by converting once on CUDA.
                    with vpi.Backend.CUDA:
                        src = vpi.asimage(rgb, vpi.Format.RGB8).convert(src_fmt)

                def convert():
                    with backend:
                        _ = src.convert(dst_fmt)
                    vpi.Stream.current.sync()

                convert()  # warmup / support probe
                stats = timed(convert, warmup=5, iters=30)
                emit(rows, op, w, h, variant, stats)
            except Exception as e:  # noqa: BLE001 — probe-and-skip is the point
                print(f"{op:>26} {w}x{h:<9} {variant:>12} N/A ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="write JSON lines here")
    args = ap.parse_args()

    rows = []
    rng = np.random.default_rng(42)
    for w, h in SIZES:
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        gray = rng.integers(0, 256, (h, w), dtype=np.uint8)
        rgb_f32 = (rgb.astype(np.float32) / 255.0).copy()

        if HAVE_CV2:
            for op, fn in cv2_cases(rgb, rgba, gray, rgb_f32, w, h):
                iters = 20 if w >= 3840 else 50
                emit(rows, op, w, h, "opencv-cpu", timed(fn, warmup=5, iters=iters))
        if HAVE_VPI:
            run_vpi(rows, rgb, w, h)
        print()

    if args.json:
        with open(args.json, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {args.json}")


if __name__ == "__main__":
    main()
