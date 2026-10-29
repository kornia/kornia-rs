"""Baselines for `median_blur_depth` (see `bench_depth.rs`).

Times `cv2.medianBlur` (`BORDER_REPLICATE`, the only border it offers) on
the same synthetic depth images as the Rust bench: two planar surfaces with
noise and ~10 % holes (`0` for `u16`, NaN for `f32`), and the hole-free
variant (`_dense`, holes replaced by `1`) that `cv2.medianBlur` handles
like for like, since it has no invalid marker. `u16` and `f32`, k = 3 and
5, 640x480 and 1280x720, on one thread and on all of OpenCV's threads.
`dst` is pre-allocated, the most favourable setup for OpenCV.

Run with any python3 that has numpy and opencv-python:

    python3 bench_depth_baselines.py
"""

import time

import cv2
import numpy as np

SIZES = [(640, 480), (1280, 720)]
KSIZES = [3, 5]
ROUNDS = 5
ITERS = 50


def depth_u16(w: int, h: int) -> np.ndarray:
    """Same fixture as `depth_u16` in the Rust bench, bit for bit."""
    state = 0x9E3779B97F4A7C15
    mask64 = (1 << 64) - 1
    out = np.empty(w * h, dtype=np.uint16)
    for i in range(w * h):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask64
        r = state >> 33
        x, y = i % w, i // w
        if r % 10 == 0:
            out[i] = 0
        else:
            base = 900 if x < w // 2 else 2400
            out[i] = base + x // 4 + y // 4 + (r >> 28)
    return out.reshape(h, w)


def depth_f32(depth: np.ndarray) -> np.ndarray:
    """The u16 fixture in metres, holes as NaN (as `depth_f32` in Rust)."""
    out = depth.astype(np.float32) * np.float32(1e-3)
    out[depth == 0] = np.nan
    return out


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
        holey = depth_u16(w, h)
        dense = np.maximum(holey, 1)
        inputs = [
            ("u16", "", holey),
            ("f32", "", depth_f32(holey)),
            ("u16", "_dense", dense),
            ("f32", "_dense", depth_f32(dense)),
        ]
        for ksize in KSIZES:
            for dtype, tag, src in inputs:
                dst = np.empty_like(src)
                line = f"{w}x{h} median_{dtype}_k{ksize}{tag}: "
                for n in (1, threads):
                    cv2.setNumThreads(n)
                    ms = best_ms(lambda: cv2.medianBlur(src, ksize, dst=dst))
                    line += f"opencv[{n}t] {ms:.3f} ms  "
                print(line.rstrip())
    cv2.setNumThreads(threads)


if __name__ == "__main__":
    main()
