"""Device interop benchmarks: quantify the zero-copy hand-off vs a copy-based API.

Run (needs a CUDA build + torch-CUDA):
    python kornia-py/benchmarks/bench_cuda_interop.py

Reports, per image size:
- DLPack export (device Image -> torch): the zero-copy hand-off cost.
- D2H + H2D round-trip (what a copy-based `.download()`/upload API costs).
- gray_from_rgb on device (GPU) vs numpy (CPU).
And:
- the fused preprocessor serving throughput (run_into, no per-frame alloc).
- the imgproc residency-dispatcher's FIXED overhead, isolated by running a
  color op at a tiny (1x1) size against a near-zero-cost PyO3 call floor.
"""

import time

import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("torch not installed — interop bench needs torch-CUDA")

import kornia_rs
from kornia_rs.image import Image

if not (kornia_rs.cuda.is_available() and torch.cuda.is_available()):
    raise SystemExit("no CUDA device / torch-CUDA")

ip = kornia_rs.imgproc


def bench(fn, n=200, warm=30):
    """Mean microseconds per call, GPU-synchronized, after warmup."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


def main() -> None:
    print(f"{'size':>7} {'MB':>5} | {'dlpack(zc)':>11} {'D2H+H2D copy':>13} {'speedup':>8}"
          f" | {'gray GPU':>9} {'gray CPU':>9}")
    for h, w, label in [(480, 640, "VGA"), (1080, 1920, "1080p"), (2160, 3840, "4K")]:
        a = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
        dev = Image.from_numpy(a).to_cuda()
        mb = a.nbytes / 1e6
        t_zc = bench(lambda: torch.from_dlpack(dev))
        t_cpy = bench(lambda: torch.from_numpy(dev.numpy()).cuda())
        t_gpu = bench(lambda: ip.gray_from_rgb(dev))
        t_cpu = bench(lambda: ip.gray_from_rgb(a), n=100)
        print(f"{label:>7} {mb:5.1f} | {t_zc:9.1f}us {t_cpy:11.1f}us {t_cpy / max(t_zc, 1e-6):6.0f}x"
              f" | {t_gpu:7.1f}us {t_cpu:7.1f}us")

    # Fused serving path: preallocate once, run_into each frame (zero per-frame alloc).
    pre = kornia_rs.cuda.CudaPreprocessor(mode="letterbox", format="rgb", f16=True)
    out = pre.alloc_output(640, 640)
    frame = np.random.default_rng(0).integers(0, 256, (1280 * 720 * 3,), dtype=np.uint8)
    t_run = bench(lambda: pre.run_into(out, frame, 1280, 720), n=300)
    print(f"\npreprocess 720p -> 640x640 f16 letterbox (run_into): "
          f"{t_run:.1f}us/frame  ({1e6 / t_run:.0f} fps)")

    # imgproc dispatcher overhead: bench a color op at 1x1 (kernel cost ~0) so
    # the timing is dominated by the numpy|Image residency check + PyO3 call
    # itself, then compare against a near-zero-cost baseline PyO3 call (no
    # dispatch, no array work) to see how much the dispatcher adds on top of
    # the call floor every pyo3 function already pays.
    def ns(fn, n=20000, warm=2000):
        for _ in range(warm):
            fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n * 1e9

    tiny = np.zeros((1, 1, 3), np.uint8)
    t_floor = ns(kornia_rs.cuda.is_available)
    t_dispatch = ns(lambda: ip.gray_from_rgb(tiny))
    print(f"\ndispatcher overhead: baseline PyO3 call {t_floor:.0f}ns, "
          f"gray_from_rgb(1x1 numpy) {t_dispatch:.0f}ns "
          f"(+{t_dispatch - t_floor:.0f}ns over floor) — "
          f"negligible next to any real image's kernel time (>80us, see above).")


if __name__ == "__main__":
    main()
