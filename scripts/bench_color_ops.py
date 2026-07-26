#!/usr/bin/env python3
"""Color-op section of the per-op 10x table: GPU-capable conversions vs cv2 (+VPI where an equivalent exists)."""
import time, statistics as st
import numpy as np
from kornia_rs import imgproc
from kornia_rs.cuda import Stream
from kornia_rs.image import Image

H, W = 1080, 1920
rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
imgf = rng.random((H, W, 3), dtype=np.float32)
stream = Stream.new()
d_u8 = Image.from_numpy(img).to_cuda(stream)
d_f32 = Image.from_numpy(imgf).to_cuda(stream)

def gpu(fn, warm=60, iters=200, rounds=3):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm): fn()
        stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        stream.synchronize()
        best = min(best, (time.perf_counter()-t0)*1e3/iters)
    return best

def cpu(fn, warm=10, iters=60, rounds=2):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm): fn()
        ts=[]
        for _ in range(iters):
            t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e3)
        best=min(best, st.median(ts))
    return best

import cv2
rows = []
def add(name, gfn, cfn):
    try:
        g = gpu(gfn)
    except Exception as e:
        print(f"{name}: GPU err {e}"); return
    c = cpu(cfn) if cfn else float("nan")
    rows.append((name, g, c, c/g if c==c else float("nan")))

add("gray_from_rgb",  lambda: imgproc.gray_from_rgb(d_u8),  lambda: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
add("bgr_from_rgb",   lambda: imgproc.bgr_from_rgb(d_u8),   lambda: cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
add("rgba_from_rgb",  lambda: imgproc.rgba_from_rgb(d_u8),  lambda: cv2.cvtColor(img, cv2.COLOR_RGB2RGBA))
add("hsv_from_rgb",   lambda: imgproc.hsv_from_rgb(d_f32),  lambda: cv2.cvtColor(imgf, cv2.COLOR_RGB2HSV))
add("lab_from_rgb",   lambda: imgproc.lab_from_rgb(d_f32),  lambda: cv2.cvtColor(imgf, cv2.COLOR_RGB2Lab))
add("ycbcr_from_rgb", lambda: imgproc.ycbcr_from_rgb(d_u8), lambda: cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))

add("sepia_from_rgb", lambda: imgproc.sepia_from_rgb(d_u8), None)

print(f"\nColor ops, 1080p (sustained GPU vs cv2 CPU, ms)")
print(f"{'op':16s} {'gpu':>8s} {'cv2':>8s} {'x-cv2':>7s}")
for n,g,c,x in rows:
    print(f"{n:16s} {g:8.3f} {c:8.3f} {x:7.1f}")
