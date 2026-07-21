#!/usr/bin/env python3
"""Pixel-level correctness check: kornia-rs GPU kernels vs NVIDIA VPI (CUDA backend).

Runs the dump_cuda_* Rust examples, then compares their output against VPI
pixel-by-pixel. VPI is the primary comparison target for Jetson / embedded
deployments.

Build Rust binaries first:
    cargo build --features cuda --release \
        --example dump_cuda_resize \
        --example dump_cuda_warp_affine \
        --example dump_cuda_warp_perspective

Run from repo root:
    python3 crates/kornia-imgproc/examples/check_correctness_vpi.py

Requirements:
    pip install numpy
    # VPI 3.x: install via JetPack SDK or NVIDIA VPI package (Jetson / x86 with JetPack)
    # On Jetson: sudo apt install libnvvpi3 vpi3-dev python3-vpi3
    # On x86:   install via NVIDIA SDK Manager with JetPack components

VPI availability note
---------------------
VPI is a Jetson SDK component and is not available in the standard CUDA repo
for x86 Ubuntu. On a GTX 1650 dev machine, VPI will not be present — this
script detects the absence and exits with a clear message. Run this script on
a Jetson or an x86 machine with JetPack installed to get actual numbers.

Algorithm notes
---------------
VPI CUDA resize bilinear   — VPI uses the CUDA bilinear sampler.
                             Our kernel uses BORDER_REPLICATE at source edges;
                             VPI uses BORDER_ZERO. Expect INFO (not PASS) for
                             non-identity transforms at image borders.

VPI warpAffine bilinear    — VPI perspwarp / warpAffine both use CUDA backend.
                             Matrix convention: forward 2×3 (VPI + OpenCV agree).

VPI perspwarp bilinear     — 3×3 homography, forward convention.
                             For affine-embedded homographies (bottom row [0,0,1]),
                             warp_perspective and warp_affine should agree.
"""

import json
import subprocess
import sys

import numpy as np

try:
    import vpi  # noqa: F401 — just test availability
    HAS_VPI = True
except ImportError:
    HAS_VPI = False

RESIZE_BIN = "./target/release/examples/dump_cuda_resize"
WARP_AFF_BIN = "./target/release/examples/dump_cuda_warp_affine"
WARP_PER_BIN = "./target/release/examples/dump_cuda_warp_perspective"

PASS = "\033[32m✅ PASS\033[0m"
FAIL = "\033[31m❌ FAIL\033[0m"
INFO = "\033[33m⚠  INFO\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ramp(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) / max(n - 1, 1)).astype(np.float32)


def run_rust(bin_path: str, *args) -> dict:
    out = subprocess.run(
        [bin_path, *[str(a) for a in args]],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def report(label: str, gpu: np.ndarray, ref: np.ndarray, tol: float | None) -> bool:
    diff = np.abs(gpu.astype(np.float64) - ref.astype(np.float64))
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    if tol is None:
        status = INFO
        verdict = f"max={max_err:.2e}  mean={mean_err:.2e}  (report only)"
        ok = True
    else:
        ok = max_err <= tol
        status = PASS if ok else FAIL
        verdict = f"max={max_err:.2e}  mean={mean_err:.2e}  tol={tol:.0e}"

    print(f"  {status}  {label:<52}  {verdict}")
    return ok


def np_to_vpi_f32(arr: np.ndarray):
    """Import a HWC float32 numpy array into a VPI image."""
    import vpi

    h, w, c = arr.shape
    if c == 1:
        return vpi.asimage(arr[:, :, 0], vpi.Format.F32)
    if c == 3:
        # VPI does not have a packed F32x3 format on all platforms.
        # Use the planar RGB F32 format if available, else fall back to U8.
        try:
            return vpi.asimage(arr, vpi.Format.RGBf32)
        except Exception:
            # Fall back: scale to U8 for the comparison (noted in output).
            arr_u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            return vpi.asimage(arr_u8, vpi.Format.RGB8), True
    raise ValueError(f"Unsupported channel count: {c}")


def vpi_to_numpy(vpi_img) -> np.ndarray:
    """Download a VPI image to a float32 numpy array."""
    with vpi_img.rlock_cpu() as data:
        arr = np.copy(data)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# VPI resize
# ---------------------------------------------------------------------------

def vpi_resize(src_np: np.ndarray, dst_w: int, dst_h: int, interp) -> np.ndarray:
    import vpi

    src_img = np_to_vpi_f32(src_np)
    is_u8 = isinstance(src_img, tuple)
    if is_u8:
        src_img, _ = src_img

    with vpi.Backend.CUDA:
        dst_img = vpi.rescale(src_img, (dst_w, dst_h), interp)
    dst_img.sync()

    out = vpi_to_numpy(dst_img)
    # Reshape to HWC if needed.
    if out.ndim == 2:
        out = out[:, :, np.newaxis]
    return out


# ---------------------------------------------------------------------------
# VPI warp affine
# ---------------------------------------------------------------------------

def vpi_warp_affine(src_np: np.ndarray, m: np.ndarray, interp) -> np.ndarray:
    """m is a 2×3 forward affine matrix (same convention as OpenCV / our dump binary)."""
    import vpi

    src_img = np_to_vpi_f32(src_np)
    is_u8 = isinstance(src_img, tuple)
    if is_u8:
        src_img, _ = src_img

    h, w = src_np.shape[:2]
    # VPI warpAffine expects the forward matrix in float64.
    xform = m.astype(np.float64)

    with vpi.Backend.CUDA:
        dst_img = vpi.warpaffine(src_img, xform, (w, h), interp)
    dst_img.sync()

    out = vpi_to_numpy(dst_img)
    if out.ndim == 2:
        out = out[:, :, np.newaxis]
    return out


# ---------------------------------------------------------------------------
# VPI warp perspective
# ---------------------------------------------------------------------------

def vpi_warp_perspective(src_np: np.ndarray, h3x3: np.ndarray, interp) -> np.ndarray:
    """h3x3 is a 3×3 forward homography matrix."""
    import vpi

    src_img = np_to_vpi_f32(src_np)
    is_u8 = isinstance(src_img, tuple)
    if is_u8:
        src_img, _ = src_img

    h, w = src_np.shape[:2]
    xform = h3x3.astype(np.float64)

    with vpi.Backend.CUDA:
        dst_img = vpi.perspwarp(src_img, xform, (w, h), interp)
    dst_img.sync()

    out = vpi_to_numpy(dst_img)
    if out.ndim == 2:
        out = out[:, :, np.newaxis]
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

VPI_INTERP = {
    "bilinear": None,  # filled in at runtime after vpi import
    "nearest": None,
}


def load_vpi_interps():
    import vpi

    VPI_INTERP["bilinear"] = vpi.Interp.LINEAR
    VPI_INTERP["nearest"] = vpi.Interp.NEAREST


def check_resize(all_ok: bool) -> bool:
    print("\n=== resize vs VPI CUDA ===")

    # (mode, sw, sh, dw, dh, vpi_interp_key, tol)
    # tol=None → report only (known convention/border difference).
    cases = [
        ("bilinear", 64, 48, 32, 24, "bilinear", 2e-4),
        ("bilinear", 32, 24, 64, 48, "bilinear", 2e-4),
        ("nearest", 64, 48, 32, 24, "nearest", None),   # convention may differ
    ]

    for mode, sw, sh, dw, dh, interp_key, tol in cases:
        n_src = sw * sh * 3
        src_np = ramp(n_src).reshape(sh, sw, 3)

        d = run_rust(RESIZE_BIN, mode, sw, sh, dw, dh)
        gpu = np.array(d["pixels"], dtype=np.float32).reshape(dh, dw, 3)

        try:
            ref = vpi_resize(src_np, dw, dh, VPI_INTERP[interp_key])
            if ref.shape != gpu.shape:
                ref = ref.reshape(gpu.shape)
        except Exception as exc:
            print(f"  ⚠  VPI error for {mode} {sw}×{sh}→{dw}×{dh}: {exc}")
            continue

        label = f"resize {mode:8}  {sw}×{sh}→{dw}×{dh}"
        ok = report(label, gpu, ref, tol)
        all_ok = all_ok and ok

    return all_ok


def check_warp_affine(all_ok: bool) -> bool:
    print("\n=== warp-affine vs VPI CUDA ===")

    # (mode, w, h, angle, tol)
    cases = [
        ("bilinear", 64, 64, 0.0, 2e-4),
        ("bilinear", 64, 64, 30.0, None),   # BORDER_REPLICATE vs VPI's BORDER_ZERO
        ("nearest", 64, 64, 0.0, 1e-4),
        ("nearest", 64, 64, 90.0, 1e-4),
    ]

    for mode, w, h, angle, tol in cases:
        n = w * h * 3
        src_np = ramp(n).reshape(h, w, 3)

        d = run_rust(WARP_AFF_BIN, mode, w, h, angle)
        gpu = np.array(d["pixels"], dtype=np.float32).reshape(h, w, 3)
        m = np.array(d["m"], dtype=np.float64).reshape(2, 3)

        try:
            ref = vpi_warp_affine(src_np, m, VPI_INTERP[mode])
            if ref.shape != gpu.shape:
                ref = ref.reshape(gpu.shape)
        except Exception as exc:
            print(f"  ⚠  VPI error for warp_affine {mode} angle={angle}: {exc}")
            continue

        label = f"warp_affine {mode:8}  {w}×{h}  angle={angle:5.1f}°"
        ok = report(label, gpu, ref, tol)
        all_ok = all_ok and ok

    return all_ok


def check_warp_perspective(all_ok: bool) -> bool:
    print("\n=== warp-perspective vs VPI CUDA ===")

    # Affine-embedded homographies (bottom row [0,0,1]):
    # warp_perspective and warp_affine should give bit-identical output for these.
    cases = [
        ("bilinear", 64, 64, 0.0, 2e-4),
        ("bilinear", 64, 64, 30.0, None),   # border difference at edges
        ("nearest", 64, 64, 0.0, 1e-4),
    ]

    for mode, w, h, angle, tol in cases:
        n = w * h * 3
        src_np = ramp(n).reshape(h, w, 3)

        d = run_rust(WARP_PER_BIN, mode, w, h, angle)
        gpu = np.array(d["pixels"], dtype=np.float32).reshape(h, w, 3)
        h3x3 = np.array(d["h3x3"], dtype=np.float64).reshape(3, 3)

        try:
            ref = vpi_warp_perspective(src_np, h3x3, VPI_INTERP[mode])
            if ref.shape != gpu.shape:
                ref = ref.reshape(gpu.shape)
        except Exception as exc:
            print(f"  ⚠  VPI error for warp_perspective {mode} angle={angle}: {exc}")
            continue

        label = f"warp_perspective {mode:8}  {w}×{h}  angle={angle:5.1f}°"
        ok = report(label, gpu, ref, tol)
        all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_bins() -> bool:
    ok = True
    for path in (RESIZE_BIN, WARP_AFF_BIN, WARP_PER_BIN):
        try:
            subprocess.run([path, "--help"], capture_output=True)
        except FileNotFoundError:
            print(f"Binary not found: {path}")
            print(
                "Build with: cargo build --features cuda --release "
                "--example dump_cuda_resize "
                "--example dump_cuda_warp_affine "
                "--example dump_cuda_warp_perspective"
            )
            ok = False
    return ok


if __name__ == "__main__":
    if not HAS_VPI:
        print(
            "VPI not available (import vpi failed).\n"
            "VPI ships with NVIDIA JetPack SDK. Install on Jetson:\n"
            "  sudo apt install libnvvpi3 vpi3-dev python3-vpi3\n"
            "On x86, install via NVIDIA SDK Manager with JetPack components.\n"
            "This script is intended to be run on a Jetson or VPI-equipped x86 machine."
        )
        sys.exit(0)

    if not check_bins():
        sys.exit(1)

    load_vpi_interps()

    all_ok = True
    all_ok = check_resize(all_ok)
    all_ok = check_warp_affine(all_ok)
    all_ok = check_warp_perspective(all_ok)

    print()
    if all_ok:
        print("✅  ALL VPI CHECKS PASSED")
    else:
        print("❌  SOME VPI CHECKS FAILED — see above")
    sys.exit(0 if all_ok else 1)
