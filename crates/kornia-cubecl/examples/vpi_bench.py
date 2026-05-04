#!/usr/bin/env python3
"""VPI bilinear rescale bench on Jetson Orin Nano — same sizes as our cubecl bench."""
import time
import numpy as np
import vpi

SIZES = [
    (512, 256, 256, 128),
    (1024, 512, 512, 256),
    (2048, 1024, 1024, 512),
    (4096, 2048, 2048, 1024),
    (8192, 4096, 4096, 2048),
    (1920, 1080, 960, 540),
]
REPS = 10
WARMUP = 3

def median(xs):
    s = sorted(xs)
    return s[len(s)//2]

print("# VPI Rescale bench — bilinear, RGB8, on Jetson Orin Nano")
print(f"# Reps={REPS}, warmup={WARMUP}, reporting min/median/mean μs and Mpix/s (median).\n")
print(f"{'src→dst':<26}{'arm':<22}{'min(μs)':>11}{'med(μs)':>11}{'mean(μs)':>11}{'Mpix/s':>10}")
print("-" * 91)

for src_w, src_h, dst_w, dst_h in SIZES:
    dst_pix = dst_w * dst_h
    label = f"{src_w}x{src_h}→{dst_w}x{dst_h}"

    # Random RGB8 input — same seed as our cubecl bench (best-effort match)
    rng = np.random.default_rng(0xC0FFEE)
    src_np = rng.integers(0, 256, (src_h, src_w, 3), dtype=np.uint8)

    # CUDA backend kernel-only (warm device buffers)
    with vpi.Backend.CUDA:
        src_img = vpi.asimage(src_np, vpi.Format.RGB8)
        dst_img = vpi.Image((dst_w, dst_h), vpi.Format.RGB8)

        # Warmup
        for _ in range(WARMUP):
            src_img.rescale(dst_img, interp=vpi.Interp.LINEAR)
        vpi.Stream.default.sync()

        # Timed
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            src_img.rescale(dst_img, interp=vpi.Interp.LINEAR)
            vpi.Stream.default.sync()
            times.append(time.perf_counter() - t0)
        mn, md, mu = min(times), median(times), sum(times)/len(times)
        mpix = dst_pix / md / 1e6
        print(f"{label:<26}{'vpi_cuda_kernel':<22}{mn*1e6:>11.1f}{md*1e6:>11.1f}{mu*1e6:>11.1f}{mpix:>10.1f}")

    # CPU backend kernel-only — VPI's CPU is single-thread scalar
    with vpi.Backend.CPU:
        src_img = vpi.asimage(src_np, vpi.Format.RGB8)
        dst_img = vpi.Image((dst_w, dst_h), vpi.Format.RGB8)
        for _ in range(WARMUP):
            src_img.rescale(dst_img, interp=vpi.Interp.LINEAR)
        vpi.Stream.default.sync()

        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            src_img.rescale(dst_img, interp=vpi.Interp.LINEAR)
            vpi.Stream.default.sync()
            times.append(time.perf_counter() - t0)
        mn, md, mu = min(times), median(times), sum(times)/len(times)
        mpix = dst_pix / md / 1e6
        print(f"{label:<26}{'vpi_cpu_kernel':<22}{mn*1e6:>11.1f}{md*1e6:>11.1f}{mu*1e6:>11.1f}{mpix:>10.1f}")

    print()
