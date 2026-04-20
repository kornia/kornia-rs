"""Bench resize across interpolation modes vs OpenCV."""
import time, numpy as np, cv2, kornia_rs as K

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
    for tw, th, desc in [(w // 2, h // 2, "0.5x"), (224, 224, "224²")]:
        print(f"\n{w}x{h} → {tw}x{th} ({desc})")
        for name, kmode, cvmode in modes:
            k = bench(lambda: K.imgproc.resize(data, (th, tw), kmode, antialias=False))
            c = bench(lambda: cv2.resize(data, (tw, th), interpolation=cvmode))
            ratio = c / k if k > 0 else 0
            flag = "✓" if k < c else "✗"
            print(f"  {name:10s} k={k:7.3f}ms  cv={c:7.3f}ms  ratio={ratio:.2f}x {flag}")
