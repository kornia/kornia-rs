#!/usr/bin/env python3
"""Full-kernel sweep across the wider CV ecosystem.

Runs the same operation matrix as `bench_cuda_imgproc` (the Rust sweep that
covers kornia's CPU and CUDA paths) against every other backend we can reach:

    OpenCV 5 CPU      - opencv 5.x, no CUDA
    OpenCV 5 CUDA     - opencv 5.x cv2.cuda.*
    PyTorch CUDA      - torch.nn.functional
    NVIDIA VPI 3      - vpi.Backend.CUDA

Emits one markdown table so the numbers can be pasted straight into
benchmarks.md alongside the kornia columns.

Backends that are missing, or that do not implement a given operation, produce
`n/a` rather than a blank or a guess.

Usage
-----
    # point at an OpenCV 5 build that has CUDA
    export OPENCV5_PATH=$HOME/.local/opencv5-cuda/lib/python3.10/dist-packages
    python3 bench_ecosystem_sweep.py [--iters 100] [--warmup 30]

The OpenCV build must be importable by the interpreter running this script.
If it was compiled against NumPy 1.x, run inside a venv with `numpy<2`.
"""
import argparse
import os
import sys
import time

import numpy as np

# ── backend discovery ────────────────────────────────────────────────────────
if os.environ.get("OPENCV5_PATH"):
    sys.path.insert(0, os.environ["OPENCV5_PATH"])

try:
    import cv2
    HAS_CV = True
    CV_VER = cv2.__version__
    HAS_CV_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception as e:                                    # noqa: BLE001
    HAS_CV = HAS_CV_CUDA = False
    CV_VER = f"unavailable ({e})"

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = torch.cuda.is_available()
    TORCH_VER = torch.__version__
except Exception as e:                                    # noqa: BLE001
    HAS_TORCH = False
    TORCH_VER = f"unavailable ({e})"

try:
    import vpi
    HAS_VPI = True
    VPI_VER = getattr(vpi, "__version__", "3.x")
except Exception as e:                                    # noqa: BLE001
    HAS_VPI = False
    VPI_VER = f"unavailable ({e})"

NA = "n/a"


# ── timing ───────────────────────────────────────────────────────────────────
def timed(fn, sync, warmup, iters):
    """Minimum per-call wall time in ms. Minimum, not mean: it is the sample
    least contaminated by unrelated work on the host."""
    try:
        for _ in range(warmup):
            fn()
        sync()
    except Exception:                                     # noqa: BLE001
        return None
    best = float("inf")
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def nosync():
    pass


def cuda_sync():
    cv2.cuda.Stream_Null().waitForCompletion()


def torch_sync():
    torch.cuda.synchronize()


# ── case matrix, matched to bench_cuda_imgproc ───────────────────────────────
FHD = (1920, 1080)
UHD = (3840, 2160)


def cases():
    """(label, interp, resolution_text, builder) where builder returns a dict of
    backend -> zero-arg callable, or None where that backend cannot do it."""
    out = []

    # ---- resize -------------------------------------------------------------
    for (sw, sh), (dw, dh) in ((FHD, (960, 540)), (UHD, FHD)):
        res = f"{sw}x{sh}->{dw}x{dh}"
        for interp, cv_flag, t_mode, vpi_interp in (
            ("bilinear", "INTER_LINEAR", "bilinear", "LINEAR"),
            ("nearest", "INTER_NEAREST", "nearest", "NEAREST"),
            ("bicubic", "INTER_CUBIC", "bicubic", "CUBIC"),
        ):
            out.append(("resize (f32)", interp, res,
                        _resize_builder(sw, sh, dw, dh, cv_flag, t_mode, vpi_interp, np.float32)))
            out.append(("resize (u8)", interp, res,
                        _resize_builder(sw, sh, dw, dh, cv_flag, t_mode, vpi_interp, np.uint8)))

    # ---- warps --------------------------------------------------------------
    for (w, h) in (FHD, UHD):
        res = f"{w}x{h}"
        out.append(("warp_affine (30deg, f32)", "bilinear", res, _warp_affine_builder(w, h, np.float32)))
        out.append(("warp_affine (30deg, u8)", "bilinear", res, _warp_affine_builder(w, h, np.uint8)))
        out.append(("warp_perspective (30deg, f32)", "bilinear", res, _warp_persp_builder(w, h, np.float32)))
        out.append(("remap (f32)", "bilinear", res, _remap_builder(w, h, np.float32)))

    # ---- colour -------------------------------------------------------------
    for (w, h) in (FHD, UHD):
        res = f"{w}x{h}"
        out.append(("gray_from_rgb (u8)", "", res, _cvt_builder(w, h, np.uint8, "COLOR_RGB2GRAY", 3, 1)))
        out.append(("gray_from_rgb (f32)", "", res, _cvt_builder(w, h, np.float32, "COLOR_RGB2GRAY", 3, 1)))
        out.append(("bgr_from_rgb (u8)", "", res, _cvt_builder(w, h, np.uint8, "COLOR_RGB2BGR", 3, 3)))
        out.append(("hsv_from_rgb (f32)", "", res, _cvt_builder(w, h, np.float32, "COLOR_RGB2HSV", 3, 3)))
        out.append(("ycc_from_rgb (u8)", "", res, _cvt_builder(w, h, np.uint8, "COLOR_RGB2YCrCb", 3, 3)))

    # ---- filters and morphology --------------------------------------------
    for (w, h) in (FHD, UHD):
        res = f"{w}x{h}"
        out.append(("gaussian_blur (3x3, u8)", "", res, _gauss_builder(w, h, np.uint8, 3)))
        out.append(("gaussian_blur (5x5, f32)", "", res, _gauss_builder(w, h, np.float32, 5)))
        out.append(("box_blur (3x3, u8)", "", res, _box_builder(w, h)))
        out.append(("dilate (3x3, u8)", "", res, _morph_builder(w, h, "dilate")))
        out.append(("erode (3x3, u8)", "", res, _morph_builder(w, h, "erode")))
        out.append(("sobel (3x3, f32)", "", res, _sobel_builder(w, h)))
        out.append(("laplacian (3x3, u8)", "", res, _laplacian_builder(w, h)))
        out.append(("integral (u8)", "", res, _integral_builder(w, h)))
    return out


# ── per-op builders ──────────────────────────────────────────────────────────
def _src(w, h, dtype, ch=3):
    if dtype == np.uint8:
        return np.random.randint(0, 256, (h, w, ch), dtype=np.uint8)
    return np.random.rand(h, w, ch).astype(np.float32)


def _gpumat(a):
    g = cv2.cuda_GpuMat()
    g.upload(a)
    return g


def _torch_img(a):
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda()
    return t.float()


def _vpi_img(a):
    return vpi.asimage(np.ascontiguousarray(a))


def _vpi_pair():
    """VPI 3 ops take keyword-only backend/interp/border/stream. Verified against
    VPI 3.2.4 on a Jetson Orin Nano; the positional form used elsewhere in this
    repo raises TypeError on that version."""
    return dict(backend=vpi.Backend.CUDA, border=vpi.Border.ZERO)


def _resize_builder(sw, sh, dw, dh, cv_flag, t_mode, vpi_interp, dtype):
    def build():
        a = _src(sw, sh, dtype)
        d = {}
        if HAS_CV:
            f = getattr(cv2, cv_flag)
            d["cv_cpu"] = (lambda: cv2.resize(a, (dw, dh), interpolation=f), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            f = getattr(cv2, cv_flag)
            d["cv_cuda"] = (lambda: cv2.cuda.resize(g, (dw, dh), interpolation=f), cuda_sync)
        if HAS_TORCH:
            t = _torch_img(a)
            kw = {} if t_mode == "nearest" else {"align_corners": False}
            d["torch"] = (lambda: F.interpolate(t, size=(dh, dw), mode=t_mode, **kw), torch_sync)
        if HAS_VPI:
            try:
                v = _vpi_img(a)
                itp = getattr(vpi.Interp, vpi_interp)
                s = vpi.Stream()
                def run(v=v, itp=itp, s=s):
                    v.rescale((dw, dh), interp=itp, stream=s, **_vpi_pair())
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _warp_affine_builder(w, h, dtype):
    def build():
        a = _src(w, h, dtype)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0) if HAS_CV else None
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            d["cv_cuda"] = (lambda: cv2.cuda.warpAffine(g, m, (w, h), flags=cv2.INTER_LINEAR), cuda_sync)
        if HAS_TORCH:
            t = _torch_img(a)
            th = torch.tensor([[[0.866, -0.5, 0.0], [0.5, 0.866, 0.0]]], device="cuda")
            def run(t=t, th=th):
                g = F.affine_grid(th, list(t.shape), align_corners=False)
                return F.grid_sample(t, g, mode="bilinear", align_corners=False)
            d["torch"] = (run, torch_sync)
        if HAS_VPI:
            try:
                v = _vpi_img(a)
                s = vpi.Stream()
                Ha = np.array([[0.866, -0.5, 0.0],
                               [0.5, 0.866, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
                def run(v=v, s=s, Ha=Ha):
                    v.perspwarp(Ha, interp=vpi.Interp.LINEAR, stream=s, **_vpi_pair())
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _warp_persp_builder(w, h, dtype):
    def build():
        a = _src(w, h, dtype)
        d = {}
        H = np.array([[0.866, -0.5, 60.0], [0.5, 0.866, -40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.warpPerspective(a, H, (w, h), flags=cv2.INTER_LINEAR), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            d["cv_cuda"] = (lambda: cv2.cuda.warpPerspective(g, H, (w, h), flags=cv2.INTER_LINEAR), cuda_sync)
        if HAS_VPI:
            try:
                v = _vpi_img(a)
                s = vpi.Stream()
                def run(v=v, s=s):
                    v.perspwarp(H, interp=vpi.Interp.LINEAR, stream=s, **_vpi_pair())
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _remap_builder(w, h, dtype):
    def build():
        a = _src(w, h, dtype)
        mx = (np.tile(np.arange(w, dtype=np.float32), (h, 1)) * 0.98).astype(np.float32)
        my = (np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) * 0.98).astype(np.float32)
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.remap(a, mx, my, interpolation=cv2.INTER_LINEAR), nosync)
        if HAS_CV_CUDA:
            g, gx, gy = _gpumat(a), _gpumat(mx), _gpumat(my)
            d["cv_cuda"] = (lambda: cv2.cuda.remap(g, gx, gy, interpolation=cv2.INTER_LINEAR), cuda_sync)
        return d
    return build


def _cvt_builder(w, h, dtype, code, cin, cout):
    def build():
        a = _src(w, h, dtype, cin)
        d = {}
        if HAS_CV:
            c = getattr(cv2, code)
            d["cv_cpu"] = (lambda: cv2.cvtColor(a, c), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            c = getattr(cv2, code)
            d["cv_cuda"] = (lambda: cv2.cuda.cvtColor(g, c), cuda_sync)
        if HAS_VPI and code == "COLOR_RGB2GRAY" and dtype == np.uint8:
            try:
                v = _vpi_img(a)
                s = vpi.Stream()
                def run(v=v, s=s):
                    v.convert(vpi.Format.U8, backend=vpi.Backend.CUDA, stream=s)
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _gauss_builder(w, h, dtype, k):
    def build():
        a = _src(w, h, dtype)
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.GaussianBlur(a, (k, k), 0), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            typ = cv2.CV_8UC3 if dtype == np.uint8 else cv2.CV_32FC3
            try:
                flt = cv2.cuda.createGaussianFilter(typ, typ, (k, k), 0)
                d["cv_cuda"] = (lambda: flt.apply(g), cuda_sync)
            except Exception:                             # noqa: BLE001
                pass
        if HAS_VPI and dtype == np.uint8:
            try:
                v = _vpi_img(a[:, :, 0])          # VPI filters want single-channel
                s = vpi.Stream()
                def run(v=v, s=s, k=k):
                    v.gaussian_filter(k, 1.5, stream=s, **_vpi_pair())
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _box_builder(w, h):
    def build():
        a = _src(w, h, np.uint8)
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.blur(a, (3, 3)), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            try:
                flt = cv2.cuda.createBoxFilter(cv2.CV_8UC3, cv2.CV_8UC3, (3, 3))
                d["cv_cuda"] = (lambda: flt.apply(g), cuda_sync)
            except Exception:                             # noqa: BLE001
                pass
        if HAS_VPI:
            try:
                v = _vpi_img(a[:, :, 0])
                s = vpi.Stream()
                def run(v=v, s=s):
                    v.box_filter(3, stream=s, **_vpi_pair())
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _morph_builder(w, h, kind):
    def build():
        a = _src(w, h, np.uint8)
        ker = np.ones((3, 3), np.uint8)
        d = {}
        if HAS_CV:
            fn = cv2.dilate if kind == "dilate" else cv2.erode
            d["cv_cpu"] = (lambda: fn(a, ker), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            op = cv2.MORPH_DILATE if kind == "dilate" else cv2.MORPH_ERODE
            try:
                flt = cv2.cuda.createMorphologyFilter(op, cv2.CV_8UC3, ker)
                d["cv_cuda"] = (lambda: flt.apply(g), cuda_sync)
            except Exception:                             # noqa: BLE001
                pass
        if HAS_VPI:
            try:
                v = _vpi_img(a[:, :, 0])
                s = vpi.Stream()
                K = np.ones((3, 3), np.uint8)
                op = "dilate" if kind == "dilate" else "erode"
                def run(v=v, s=s, K=K, op=op):
                    getattr(v, op)(K, stream=s, **_vpi_pair())
                d["vpi"] = (run, s.sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _sobel_builder(w, h):
    def build():
        a = _src(w, h, np.float32, 1)[:, :, 0]
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            try:
                flt = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 1, 0, 3)
                d["cv_cuda"] = (lambda: flt.apply(g), cuda_sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


def _laplacian_builder(w, h):
    def build():
        a = _src(w, h, np.uint8, 1)[:, :, 0]
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.Laplacian(a, cv2.CV_8U, ksize=3), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            try:
                flt = cv2.cuda.createLaplacianFilter(cv2.CV_8U, cv2.CV_8U, 3)
                d["cv_cuda"] = (lambda: flt.apply(g), cuda_sync)
            except Exception:                             # noqa: BLE001
                pass
        # VPI 3.2.4 exposes no laplacian_filter; the honest answer is n/a.
        return d
    return build


def _integral_builder(w, h):
    def build():
        a = _src(w, h, np.uint8, 1)[:, :, 0]
        d = {}
        if HAS_CV:
            d["cv_cpu"] = (lambda: cv2.integral(a), nosync)
        if HAS_CV_CUDA:
            g = _gpumat(a)
            try:
                d["cv_cuda"] = (lambda: cv2.cuda.integral(g), cuda_sync)
            except Exception:                             # noqa: BLE001
                pass
        return d
    return build


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    print(f"<!-- bench_ecosystem_sweep  warmup={args.warmup}  iters={args.iters} -->")
    print(f"<!-- OpenCV {CV_VER} (cuda={HAS_CV_CUDA})  PyTorch {TORCH_VER}  VPI {VPI_VER} -->")
    print()
    print("| Operation | Interp | Resolution | OpenCV5 CPU | OpenCV5 CUDA | PyTorch | VPI 3 |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")

    for name, interp, res, build in cases():
        try:
            backends = build()
        except Exception as e:                            # noqa: BLE001
            print(f"| {name} | {interp} | {res} | setup failed: {e} | | | |", file=sys.stderr)
            continue
        cells = []
        for key in ("cv_cpu", "cv_cuda", "torch", "vpi"):
            entry = backends.get(key)
            if entry is None:
                cells.append(NA)
                continue
            fn, sync = entry
            ms = timed(fn, sync, args.warmup, args.iters)
            cells.append(f"{ms:.3f}" if ms is not None else NA)
        print(f"| {name} | {interp} | {res} | " + " | ".join(cells) + " |")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
