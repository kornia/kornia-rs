"""Bench resize across interpolation modes vs OpenCV."""
import numpy as np
import cv2

from kornia_rs.image import Image

from _bench import bench as _bench_fn


def bench(fn, n=None, warmup=None):
    """Backwards-compat shim around benchmarks/_bench.py — reports min ms.

    The shared helper auto-tunes iteration count to a 1s budget; the legacy
    n / warmup args are accepted but ignored. min_ms is the right number for
    sub-millisecond ops; mean is biased high by GC/scheduler noise.
    """
    r = _bench_fn(fn, target_seconds=1.0, min_iters=100)
    return r.min_ms


modes = [
    ("bilinear", "bilinear", cv2.INTER_LINEAR),
    ("bicubic",  "bicubic",  cv2.INTER_CUBIC),
    ("lanczos",  "lanczos",  cv2.INTER_LANCZOS4),
    ("nearest",  "nearest",  cv2.INTER_NEAREST),
]

for (h, w) in [(480, 640), (1080, 1920)]:
    data = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.frombuffer(data)  # zero-copy wrap
    for tw, th, desc in [
        (w // 2, h // 2, "0.5x"),
        (224, 224, "224²"),
        (w * 2, h * 2, "2x up"),
    ]:
        print(f"\n{w}x{h} → {tw}x{th} ({desc})")
        for name, kmode, cvmode in modes:
            k = bench(lambda: img.resize(width=tw, height=th, interpolation=kmode))
            c = bench(lambda: cv2.resize(data, (tw, th), interpolation=cvmode))
            ratio = c / k if k > 0 else 0
            flag = "✓" if k < c else "✗"
            print(f"  {name:10s} k={k:7.3f}ms  cv={c:7.3f}ms  ratio={ratio:.2f}x {flag}")
