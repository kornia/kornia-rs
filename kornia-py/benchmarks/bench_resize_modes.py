"""Bench resize across interpolation modes vs OpenCV."""
import time
import numpy as np
import cv2

from kornia_rs.image import Image

def bench(fn, n=150, warmup=8):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


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
