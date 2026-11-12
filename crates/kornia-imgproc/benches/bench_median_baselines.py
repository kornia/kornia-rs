"""Baselines for `median_blur` (see `bench_median.rs`).

Times `cv2.medianBlur` (`BORDER_REPLICATE`, the only border it offers) on
the same images as the Rust bench: `u8`, `u16` and `f32`, k = 3 and 5,
640x480 and 1280x720, on one thread and on all of OpenCV's threads. `dst`
is pre-allocated, the most favourable setup for OpenCV.

Run with any python3 that has numpy and opencv-python:

    python3 bench_median_baselines.py
"""

import time

import cv2
import numpy as np

SIZES = [(640, 480), (1280, 720)]
KSIZES = [3, 5]
ROUNDS = 5
ITERS = 50


def images(w: int, h: int) -> dict:
    """Same data as `bench_median.rs`: a multiplicative hash of the index."""
    hashed = np.arange(w * h, dtype=np.uint64) * np.uint64(2654435761)
    u16 = ((hashed >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16)
    return {
        "u8": ((hashed >> np.uint64(24)) & np.uint64(0xFF)).astype(np.uint8).reshape(h, w),
        "u16": u16.reshape(h, w),
        "f32": (u16.astype(np.float32) * np.float32(1e-3)).reshape(h, w),
    }


def best_ms(fn) -> float:
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        best = min(best, (time.perf_counter() - t0) * 1e3 / ITERS)
    return best


def main() -> None:
    threads = cv2.getNumThreads()
    print(f"opencv {cv2.__version__}, numpy {np.__version__}, threads 1 / {threads}")
    print(f"best of {ROUNDS} rounds x {ITERS} calls, ms per call")
    for w, h in SIZES:
        for name, src in images(w, h).items():
            for ksize in KSIZES:
                dst = np.empty_like(src)
                line = f"{w}x{h} {name}_k{ksize}: "
                for n in (1, threads):
                    cv2.setNumThreads(n)
                    ms = best_ms(lambda: cv2.medianBlur(src, ksize, dst=dst))
                    line += f"opencv[{n}t] {ms:.3f} ms  "
                print(line.rstrip())
    cv2.setNumThreads(threads)


if __name__ == "__main__":
    main()
