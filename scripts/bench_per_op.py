#!/usr/bin/env python3
"""Per-op 10x acceptance table: every u8 GPU op vs cv2 CPU and VPI-CUDA.

Acceptance criterion (Edgar, 2026-07-17): every op >= 10x vs BOTH baselines.
Run with locked clocks (`sudo jetson_clocks`) — unlocked DVFS swings both
sides by +/-30-60%, larger than several targets.

Reference snapshot (Jetson AGX Orin, 2026-07-17, UNLOCKED clocks, min-of-rounds):
  op                  gpu-pf   cv2      vpi     x-cv2  x-vpi
  resize-nearest      0.177    0.746    1.429    4.2    8.1
  resize-bilinear     0.254    2.505    1.829    9.9    7.2
  resize-bicubic      0.643    7.313    1.837   11.4    2.9
  resize-lanczos      1.206   13.262     n/a    11.0    n/a
  warp-affine         1.012   11.869    1.771   11.7    1.8
  warp-perspective    1.587   16.136    2.384   10.2    1.5
  dilate              0.227    0.615    2.055    2.7    9.0
  erode               0.243    0.618    1.652    2.5    6.8
Physics note: bicubic/affine/perspective vs VPI-CUDA cap at ~2-3x — both
sides saturate the same DRAM with competent kernels; byte-exactness is
retained by decision (no approximate fast modes).

Modes: per-frame (op + sync each frame) and sustained (N-deep enqueue, one
sync, amortized). min-of-3 rounds against unlocked DVFS.
1080p u8; resize ops go 1080p->720p, warps/morphology are full-frame.
"""
import time
import statistics as st

import numpy as np
from kornia_rs import imgproc
from kornia_rs.cuda import Stream
from kornia_rs.image import Image

H, W = 1080, 1920
DH, DW = 720, 1280
AFF = [0.87, -0.5, 400.0, 0.5, 0.87, -100.0]
PSP = [0.9, 0.12, 40.0, -0.08, 1.05, -20.0, 6.0e-5, -4.5e-5, 1.0]

rng = np.random.default_rng(0)
img3 = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
img1 = rng.integers(0, 256, size=(H, W, 1), dtype=np.uint8)

stream = Stream.new()
d3 = Image.from_numpy(img3).to_cuda(stream)
d1 = Image.from_numpy(img1).to_cuda(stream)


def per_frame(fn, warm=80, iters=200, rounds=3):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            stream.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        best = min(best, st.median(ts))
    return best


def sustained(fn, warm=80, iters=200, rounds=3):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm):
            fn()
        stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        stream.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3 / iters)
    return best


def cpu_bench(fn, warm=10, iters=60, rounds=2):
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        best = min(best, st.median(ts))
    return best


ops = {}

# kornia GPU ops (out= reuse where supported)
for mode in ["nearest", "bilinear", "bicubic", "lanczos"]:
    out = imgproc.resize(d3, (DH, DW), mode)
    fn = (lambda m=mode, o=out: imgproc.resize(d3, (DH, DW), m, out=o))
    ops[f"resize-{mode}"] = {"gpu_pf": per_frame(fn), "gpu_sus": sustained(fn)}

out_a = imgproc.warp_affine(d3, AFF, (H, W), "bilinear")
fn = lambda: imgproc.warp_affine(d3, AFF, (H, W), "bilinear", out=out_a)
ops["warp-affine"] = {"gpu_pf": per_frame(fn), "gpu_sus": sustained(fn)}

out_p = imgproc.warp_perspective(d3, PSP, (H, W), "bilinear")
fn = lambda: imgproc.warp_perspective(d3, PSP, (H, W), "bilinear", out=out_p)
ops["warp-perspective"] = {"gpu_pf": per_frame(fn), "gpu_sus": sustained(fn)}

for op in ["dilate", "erode"]:
    f = getattr(imgproc, op)
    fn = lambda f=f: f(d1, kernel="box", size=(3, 3))
    ops[op] = {"gpu_pf": per_frame(fn), "gpu_sus": sustained(fn)}

# cv2 CPU baselines
try:
    import cv2
    cvm = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
           "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
    for mode, flag in cvm.items():
        ops[f"resize-{mode}"]["cv2"] = cpu_bench(
            lambda f=flag: cv2.resize(img3, (DW, DH), interpolation=f))
    m_a = np.array(AFF, dtype=np.float64).reshape(2, 3)
    m_p = np.array(PSP, dtype=np.float64).reshape(3, 3)
    ops["warp-affine"]["cv2"] = cpu_bench(lambda: cv2.warpAffine(img3, m_a, (W, H)))
    ops["warp-perspective"]["cv2"] = cpu_bench(lambda: cv2.warpPerspective(img3, m_p, (W, H)))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ops["dilate"]["cv2"] = cpu_bench(lambda: cv2.dilate(img1, se))
    ops["erode"]["cv2"] = cpu_bench(lambda: cv2.erode(img1, se))
except ImportError:
    pass

# VPI-CUDA baselines (completion forced via rlock)
try:
    import vpi
    v3 = vpi.asimage(img3, vpi.Format.RGB8)
    v1 = vpi.asimage(img1[:, :, 0])
    def vb(fn, warm=15, iters=60, rounds=2):
        best = float("inf")
        for _ in range(rounds):
            for _ in range(warm):
                fn()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1e3)
            best = min(best, st.median(ts))
        return best
    vint = {"nearest": vpi.Interp.NEAREST, "bilinear": vpi.Interp.LINEAR,
            "bicubic": vpi.Interp.CATMULL_ROM}
    for mode, interp in vint.items():
        def f(i=interp):
            with vpi.Backend.CUDA:
                o = v3.rescale((DW, DH), interp=i)
            with o.rlock_cpu():
                pass
        ops[f"resize-{mode}"]["vpi"] = vb(f)
    def f_p():
        with vpi.Backend.CUDA:
            o = v3.perspwarp(np.array(PSP, dtype=np.float32).reshape(3, 3))
        with o.rlock_cpu():
            pass
    ops["warp-perspective"]["vpi"] = vb(f_p)
    m33 = np.array([[AFF[0], AFF[1], AFF[2]], [AFF[3], AFF[4], AFF[5]], [0, 0, 1]],
                   dtype=np.float32)
    def f_a():
        with vpi.Backend.CUDA:
            o = v3.perspwarp(m33)
        with o.rlock_cpu():
            pass
    ops["warp-affine"]["vpi"] = vb(f_a)
    se = np.ones((3, 3), dtype=np.uint8)
    def f_d():
        with vpi.Backend.CUDA:
            o = v1.dilate(se)
        with o.rlock_cpu():
            pass
    ops["dilate"]["vpi"] = vb(f_d)
    def f_e():
        with vpi.Backend.CUDA:
            o = v1.erode(se)
        with o.rlock_cpu():
            pass
    ops["erode"]["vpi"] = vb(f_e)
except Exception as e:
    print(f"vpi partial: {e}")

hdr = f"{'op':18s} {'gpu-pf':>8s} {'gpu-sus':>8s} {'cv2':>8s} {'vpi':>8s} {'x-cv2':>7s} {'x-vpi':>7s}"
print("\nPer-op table, 1080p u8 (ms; x = baseline / gpu-per-frame)")
print(hdr)
for name, d in ops.items():
    cv = d.get("cv2", float("nan"))
    vp = d.get("vpi", float("nan"))
    xc = cv / d["gpu_pf"] if cv == cv else float("nan")
    xv = vp / d["gpu_pf"] if vp == vp else float("nan")
    print(f"{name:18s} {d['gpu_pf']:8.3f} {d['gpu_sus']:8.3f} {cv:8.3f} {vp:8.3f} {xc:7.1f} {xv:7.1f}")
