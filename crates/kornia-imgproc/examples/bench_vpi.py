#!/usr/bin/env python3
"""GPU throughput benchmark: kornia-rs CUDA kernels vs NVIDIA VPI CUDA backend.

Measures wall-clock latency for resize, warp_affine, and warp_perspective at
1920×1080 and 3840×2160, comparing kornia-rs direct CUDA kernel launches
against VPI's CUDA backend.

Usage (from repo root):
    # Build kornia-rs timing binary first:
    cargo build --features cuda --release --example bench_cuda_imgproc

    # Run benchmark:
    python3 crates/kornia-imgproc/examples/bench_vpi.py

VPI availability note
---------------------
VPI ships with NVIDIA JetPack SDK. On Jetson:
    sudo apt install libnvvpi3 vpi3-dev python3-vpi3
On x86 without JetPack, VPI is not available and this script will exit early.

Benchmark methodology
---------------------
- Warm-up: WARMUP iterations discarded before measurement.
- Measurement: ITERS iterations, report mean ± 1σ.
- VPI timing: Python time.perf_counter around VPI call + explicit GPU sync.
  This includes Python/VPI API overhead but excludes H2D/D2H transfers
  (VPI manages device-resident images internally once uploaded).
- kornia-rs timing: reported from the companion Rust bench binary
  (bench_cuda_imgproc) which uses CUDA events for kernel-only timing.
- Transfer overhead is reported separately when available.

Format note
-----------
VPI CUDA backend uses U8 images for most operations (RGB8 format).
kornia-rs operates on F32 images. The throughput numbers are directly
comparable (pixels/second), but the format difference means numerical
output will differ — see check_correctness_vpi.py for that comparison.
"""

import subprocess
import sys
import time

import numpy as np

try:
    import vpi
    HAS_VPI = True
except ImportError:
    HAS_VPI = False

WARMUP = 30
ITERS = 200

# Resolutions to benchmark
RESOLUTIONS = [
    (1920, 1080, "1080p"),
    (3840, 2160, "4K"),
]

# ---------------------------------------------------------------------------
# VPI benchmark helpers
# ---------------------------------------------------------------------------

def make_vpi_image(w: int, h: int) -> "vpi.Image":
    """Create a random U8 RGB VPI image pre-uploaded to CUDA device."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    with vpi.Backend.CUDA:
        img = vpi.asimage(arr, vpi.Format.RGB8)
    img.sync()
    return img


def bench_vpi_op(fn, warmup: int = WARMUP, iters: int = ITERS) -> tuple[float, float]:
    """Run fn() warmup+iters times, return (mean_ms, std_ms) of iters measurements."""
    for _ in range(warmup):
        out = fn()
        out.sync()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        out.sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(samples)
    return float(a.mean()), float(a.std())


# ---------------------------------------------------------------------------
# Benchmark: resize
# ---------------------------------------------------------------------------

def bench_resize():
    print("\n=== resize (bilinear, 2× downscale) ===")
    print(f"{'Resolution':<12}  {'VPI mean (ms)':>14}  {'VPI σ (ms)':>10}  "
          f"{'VPI Mpix/s':>11}")

    for src_w, src_h, label in RESOLUTIONS:
        dst_w, dst_h = src_w // 2, src_h // 2
        src_img = make_vpi_image(src_w, src_h)

        mean_ms, std_ms = bench_vpi_op(
            lambda: vpi.rescale(src_img, (dst_w, dst_h), vpi.Interp.LINEAR)
        )
        mpix = (dst_w * dst_h) / (mean_ms * 1e-3) / 1e6
        print(f"  {label:<10}  {mean_ms:>14.3f}  {std_ms:>10.3f}  {mpix:>11.1f}")


# ---------------------------------------------------------------------------
# Benchmark: warp affine
# ---------------------------------------------------------------------------

def bench_warp_affine():
    print("\n=== warp_affine bilinear (30° rotation) ===")
    print(f"{'Resolution':<12}  {'VPI mean (ms)':>14}  {'VPI σ (ms)':>10}  "
          f"{'VPI Mpix/s':>11}")

    for src_w, src_h, label in RESOLUTIONS:
        src_img = make_vpi_image(src_w, src_h)

        cx, cy = src_w / 2.0, src_h / 2.0
        angle = 30.0 * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        m = np.array([
            [cos_a, sin_a, (1 - cos_a) * cx - sin_a * cy],
            [-sin_a, cos_a, sin_a * cx + (1 - cos_a) * cy],
        ], dtype=np.float64)

        mean_ms, std_ms = bench_vpi_op(
            lambda: vpi.warpaffine(src_img, m, (src_w, src_h), vpi.Interp.LINEAR)
        )
        mpix = (src_w * src_h) / (mean_ms * 1e-3) / 1e6
        print(f"  {label:<10}  {mean_ms:>14.3f}  {std_ms:>10.3f}  {mpix:>11.1f}")


# ---------------------------------------------------------------------------
# Benchmark: warp perspective
# ---------------------------------------------------------------------------

def bench_warp_perspective():
    print("\n=== warp_perspective bilinear (30° rotation, affine-embedded homography) ===")
    print(f"{'Resolution':<12}  {'VPI mean (ms)':>14}  {'VPI σ (ms)':>10}  "
          f"{'VPI Mpix/s':>11}")

    for src_w, src_h, label in RESOLUTIONS:
        src_img = make_vpi_image(src_w, src_h)

        cx, cy = src_w / 2.0, src_h / 2.0
        angle = 30.0 * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        h3x3 = np.array([
            [cos_a, sin_a, (1 - cos_a) * cx - sin_a * cy],
            [-sin_a, cos_a, sin_a * cx + (1 - cos_a) * cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        mean_ms, std_ms = bench_vpi_op(
            lambda: vpi.perspwarp(src_img, h3x3, (src_w, src_h), vpi.Interp.LINEAR)
        )
        mpix = (src_w * src_h) / (mean_ms * 1e-3) / 1e6
        print(f"  {label:<10}  {mean_ms:>14.3f}  {std_ms:>10.3f}  {mpix:>11.1f}")


# ---------------------------------------------------------------------------
# kornia-rs numbers from Rust bench binary
# ---------------------------------------------------------------------------

BENCH_BIN = "./target/release/examples/bench_cuda_imgproc"


def try_print_kornia_numbers():
    """Run the Rust bench binary and print its output, if available."""
    try:
        result = subprocess.run(
            [BENCH_BIN],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print("\n=== kornia-rs CUDA kernel timings (from Rust bench) ===")
            print(result.stdout)
        else:
            print(f"\nkornia-rs bench binary failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print(
            f"\nkornia-rs bench binary not found ({BENCH_BIN}).\n"
            "Build with: cargo build --features cuda --release --example bench_cuda_imgproc\n"
            "For now, run `cargo bench --features cuda` and compare criterion output manually."
        )
    except subprocess.TimeoutExpired:
        print("\nkornia-rs bench timed out.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_env():
    try:
        import vpi
        print(f"VPI version : {vpi.__version__}")
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        print(f"GPU         : {result.stdout.strip()}")
    except FileNotFoundError:
        pass
    print(f"Warmup iters: {WARMUP}   Measurement iters: {ITERS}")


if __name__ == "__main__":
    if not HAS_VPI:
        print(
            "VPI not available (import vpi failed).\n"
            "VPI ships with NVIDIA JetPack SDK. Install on Jetson:\n"
            "  sudo apt install libnvvpi3 vpi3-dev python3-vpi3\n"
            "On x86, install via NVIDIA SDK Manager with JetPack components.\n\n"
            "Exiting — no VPI numbers to report."
        )
        sys.exit(0)

    print_env()

    try:
        bench_resize()
        bench_warp_affine()
        bench_warp_perspective()
    except Exception as exc:
        print(f"\nBenchmark error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try_print_kornia_numbers()

    print(
        "\nNote: VPI timings include Python/VPI API call overhead (~0.1–0.5 ms).\n"
        "kornia-rs kernel-only timings (from criterion / CUDA events) exclude this.\n"
        "For a fair comparison use the H2D+kernel+D2H column from kornia-rs bench."
    )
