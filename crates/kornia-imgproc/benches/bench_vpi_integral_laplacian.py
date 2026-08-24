#!/usr/bin/env python3
"""VPI Integral & Laplacian bench"""
import time
import numpy as np

try:
    import vpi
    HAS_VPI = True
except ImportError:
    HAS_VPI = False

SIZES = [(1920, 1080, "1080p"), (3840, 2160, "4K")]
REPS = 100
WARMUP = 30

def make_vpi_image(w: int, h: int) -> "vpi.Image":
    arr = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    with vpi.Backend.CUDA:
        img = vpi.asimage(arr, vpi.Format.U8)
    img.sync()
    return img

def bench_vpi_op(fn, warmup: int = WARMUP, iters: int = REPS) -> tuple[float, float]:
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

def bench_laplacian():
    print("\n=== vpi laplacian ===")
    if not hasattr(vpi, 'laplacian'):
        print("vpi.laplacian not found in this VPI version.")
        return
    for w, h, label in SIZES:
        src_img = make_vpi_image(w, h)
        mean_ms, std_ms = bench_vpi_op(
            lambda: vpi.laplacian(src_img, 3)
        )
        mpix = (w * h) / (mean_ms * 1e-3) / 1e6
        print(f"  {label:<10}  {mean_ms:>14.3f}  {std_ms:>10.3f}  {mpix:>11.1f}")

def bench_integral():
    print("\n=== vpi integral ===")
    if not hasattr(vpi, 'integral_image') and not hasattr(vpi, 'integral'):
        print("vpi.integral_image not found in this VPI version.")
        return
    for w, h, label in SIZES:
        src_img = make_vpi_image(w, h)
        fn = getattr(vpi, 'integral_image', getattr(vpi, 'integral', None))
        mean_ms, std_ms = bench_vpi_op(
            lambda: fn(src_img)
        )
        mpix = (w * h) / (mean_ms * 1e-3) / 1e6
        print(f"  {label:<10}  {mean_ms:>14.3f}  {std_ms:>10.3f}  {mpix:>11.1f}")

if __name__ == "__main__":
    if not HAS_VPI:
        print("VPI not available.")
    else:
        bench_laplacian()
        bench_integral()
