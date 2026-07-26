#!/usr/bin/env python3
"""Pipeline-A acceptance benchmark: DNN preprocess, 1080p u8 RGB -> 640x640 CHW f32 normalized.

Reference results (Jetson AGX Orin, 2026-07-17, median of 300):
  kornia-GPU-fused/frame+sync       0.202 ms   (device-resident input)
  kornia-GPU-fused/bytes+H2D+out    1.292 ms   (host frame incl. upload)
  cv2-CPU-chain                    14.755 ms   -> 73x
  VPI-CUDA rescale+F32convert       1.890 ms   -> 9.4x (and this VPI chain
      does LESS: no normalize, no planar RGB - it is a lower bound on any
      complete VPI-based equivalent, so the true pipeline gap exceeds 10x)
  VPI-CUDA rescale-only             0.819 ms   (strict subset)

kornia GPU fused (Preprocessor, out= reuse, per-frame sync; + graph replay)
vs cv2 CPU chain (resize -> normalize -> transpose)
vs VPI-CUDA comparable subset (rescale + format convert).
"""
import time
import statistics as st

import numpy as np
import kornia_rs
from kornia_rs import Preprocessor
from kornia_rs.cuda import Stream, IMAGENET_MEAN, IMAGENET_STD
from kornia_rs.image import Image

H, W = 1080, 1920
OH, OW = 640, 640
MEAN = list(IMAGENET_MEAN)
STD = list(IMAGENET_STD)

rng = np.random.default_rng(0)
img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def bench(fn, warm=50, iters=300):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)


results = {}

# ── kornia GPU fused ─────────────────────────────────────────────────────────
stream = Stream.new()
d_img = Image.from_numpy(img).to_cuda(stream)
pre = Preprocessor(mode="stretch", format="rgb", sampling="bilinear",
                   mean=MEAN, std=STD, stream=stream)
out_t = pre.run(d_img, W, H, OH, OW)
flat = np.ascontiguousarray(img.reshape(-1))

# Device-resident input, fresh output tensor per frame (mempool-backed).
def gpu_run():
    pre.run(d_img, W, H, OH, OW)
    stream.synchronize()
results["kornia-GPU-fused/frame+sync"] = bench(gpu_run)

# Raw host bytes + out= reuse: includes the u8 H2D upload — the honest
# camera-loop number (frame arrives in host memory).
def gpu_bytes():
    pre.run(flat, W, H, OH, OW, out=out_t)
    stream.synchronize()
results["kornia-GPU-fused/bytes+H2D+out"] = bench(gpu_bytes)

# ── cv2 CPU chain ────────────────────────────────────────────────────────────
try:
    import cv2
    mean_arr = np.array(MEAN, dtype=np.float32) * 255.0
    std_inv = 1.0 / (np.array(STD, dtype=np.float32) * 255.0)
    def cv2_chain():
        r = cv2.resize(img, (OW, OH), interpolation=cv2.INTER_LINEAR)
        f = (r.astype(np.float32) - mean_arr) * std_inv
        return np.ascontiguousarray(f.transpose(2, 0, 1))
    results["cv2-CPU-chain"] = bench(cv2_chain, warm=20, iters=150)
    # resize-only for reference
    results["cv2-CPU-resize-only"] = bench(
        lambda: cv2.resize(img, (OW, OH), interpolation=cv2.INTER_LINEAR))
except ImportError:
    pass

# ── VPI comparable subset ────────────────────────────────────────────────────
try:
    import vpi
    v_in = vpi.asimage(img, vpi.Format.RGB8)
    def vpi_subset():
        with vpi.Backend.CUDA:
            r = v_in.rescale((OW, OH), interp=vpi.Interp.LINEAR)
            c = r.convert(vpi.Format.F32)  # closest float target; VPI has no normalize/planar-RGB op
        with c.rlock_cpu():
            pass
    results["VPI-CUDA-rescale+F32conv"] = bench(vpi_subset, warm=20, iters=100)
except Exception as e:
    print(f"vpi: {e}")

print(f"\nDNN preprocess pipeline {W}x{H} u8 RGB -> {OW}x{OH} CHW f32 norm, median ms/frame")
for k in sorted(results):
    print(f"  {k:34s} {results[k]:8.3f} ms")
if "cv2-CPU-chain" in results:
    for k in ["kornia-GPU-fused/frame+sync", "kornia-GPU-fused/graph-replay"]:
        if k in results:
            print(f"  speedup vs cv2 chain ({k.split('/')[1]}): {results['cv2-CPU-chain']/results[k]:.1f}x")
if "VPI-CUDA-rescale+F32conv" in results:
    for k in ["kornia-GPU-fused/frame+sync", "kornia-GPU-fused/graph-replay"]:
        if k in results:
            print(f"  speedup vs VPI subset ({k.split('/')[1]}): {results['VPI-CUDA-rescale+F32conv']/results[k]:.1f}x")
