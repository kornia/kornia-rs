#!/usr/bin/env python3
"""OpenCV findContours bench — same fixtures as bench_contours_min.rs, identical seed."""
import time
import numpy as np
import cv2

SIZES = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
REPS = 20
WARMUP = 5


def make_filled_square(w, h):
    mw, mh = w // 8, h // 8
    d = np.zeros((h, w), dtype=np.uint8)
    d[mh:h - mh, mw:w - mw] = 255  # OpenCV expects 0/255
    return d


def make_hollow_square(w, h):
    ow, oh = w // 8, h // 8
    iw, ih = w // 4, h // 4
    d = np.zeros((h, w), dtype=np.uint8)
    d[oh:h - oh, ow:w - ow] = 255
    d[ih:h - ih, iw:w - iw] = 0
    return d


def make_noise(w, h, seed=0xC0FFEE):
    """Replicates bench_contours.rs LCG bit-for-bit."""
    state = seed & ((1 << 64) - 1)
    out = np.empty(w * h, dtype=np.uint8)
    M = 6364136223846793005 & ((1 << 64) - 1)
    A = 1442695040888963407 & ((1 << 64) - 1)
    MASK = (1 << 64) - 1
    for i in range(w * h):
        state = (state * M + A) & MASK
        out[i] = ((state >> 33) & 1) * 255  # 0 or 255 for OpenCV
    return out.reshape(h, w)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def run_one(label, w, h, data, mode, method):
    # Warmup
    for _ in range(WARMUP):
        cv2.findContours(data, mode, method)
    samples = []
    for _ in range(REPS):
        t = time.perf_counter()
        cv2.findContours(data, mode, method)
        samples.append(time.perf_counter() - t)
    mn, md = min(samples), median(samples)
    mu = sum(samples) / len(samples)
    pix_per_s = (w * h) / md / 1e6
    print(f"opencv,{label},{w}x{h},{mn*1e6:.1f},{md*1e6:.1f},{mu*1e6:.1f},{pix_per_s:.1f}")


def main():
    print("# CSV: impl,fixture,size,min_us,med_us,mean_us,Mpix_per_s_median")
    print("impl,fixture,size,min_us,med_us,mean_us,Mpix_s")
    for w, h in SIZES:
        run_one("filled_square_external_simple", w, h, make_filled_square(w, h),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for w, h in SIZES:
        run_one("hollow_square_external_simple", w, h, make_hollow_square(w, h),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for w, h in SIZES:
        run_one("sparse_noise_external_simple", w, h, make_noise(w, h),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for w, h in SIZES:
        run_one("filled_square_external_none", w, h, make_filled_square(w, h),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


if __name__ == "__main__":
    main()
