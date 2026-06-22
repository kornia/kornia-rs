"""OpenCV cv2.resize benchmark — nearest-neighbor and bilinear, f32 RGB.

Run:
    python3 crates/kornia-imgproc/examples/bench_opencv_resize.py

Requires: opencv-python  (pip install opencv-python)
"""

import time
import cv2
import numpy as np

WARMUP = 50
ITERS = 200

# Each entry: (src_w, src_h), (dst_w, dst_h)
CASES = [
    ((1024, 1024), (512, 512)),
    ((512, 512), (1024, 1024)),
    ((1920, 1080), (960, 540)),
    ((1920, 1080), (3840, 2160)),
    ((3840, 2160), (1920, 1080)),
]

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
}


def bench(src, dst_size, interp_flag):
    # warmup
    for _ in range(WARMUP):
        cv2.resize(src, dst_size, interpolation=interp_flag)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        cv2.resize(src, dst_size, interpolation=interp_flag)
    elapsed = time.perf_counter() - t0
    return elapsed * 1e3 / ITERS  # ms per iter


print(f"OpenCV {cv2.__version__}  —  f32 RGB resize benchmark")
print(f"Warmup {WARMUP}, timed {ITERS} iters\n")

for name, flag in INTERP.items():
    print(f"=== {name} ===")
    print(f"  {'Source → Dest':<28}  {'ms/iter':>8}  {'GB/s':>8}")
    print("  " + "-" * 48)
    for (sw, sh), (dw, dh) in CASES:
        src = np.random.rand(sh, sw, 3).astype(np.float32)
        ms = bench(src, (dw, dh), flag)
        # bytes = dst pixels * 3 channels * 4 bytes (f32)
        gb = (dh * dw * 3 * 4) / (ms * 1e-3) / 1e9
        label = f"{sw}x{sh}→{dw}x{dh}"
        print(f"  {label:<28}  {ms:>8.3f}  {gb:>8.1f}")
    print()
