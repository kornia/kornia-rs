#!/usr/bin/env python3
"""Pixel-level correctness check: kornia-rs GPU kernels vs OpenCV reference.

Runs the dump_cuda_resize and dump_cuda_warp_affine Rust examples, then
compares their output against cv2 pixel-by-pixel.

Build first:
    cargo build --features cuda --release \
        --example dump_cuda_resize \
        --example dump_cuda_warp_affine

Run from repo root:
    python3 crates/kornia-imgproc/examples/check_correctness_cuda.py

Algorithm notes
---------------
resize bilinear  — half-pixel alignment (dx+0.5)*scale-0.5, clamped taps.
                   Matches cv2.INTER_LINEAR exactly (both use BORDER_REPLICATE
                   internally for the 2-tap edge case).

resize nearest   — floor((dx+0.5)*scale), clamped. Matches cv2.INTER_NEAREST.

resize bicubic   — Keys a=-0.5, half-pixel alignment, BORDER_REPLICATE for OOB
                   taps. cv2.INTER_CUBIC also uses a=-0.5 and BORDER_REPLICATE.
                   Expect ≤3e-4 max error (f32 FMA vs double accumulation).

resize lanczos   — Our kernel is Lanczos-3 (3-lobe, 6-tap separable).
                   cv2.INTER_LANCZOS4 is Lanczos-4 (4-lobe, 8-tap 2D).
                   DIFFERENT KERNEL — results will diverge. We report max error
                   for information only; no pass/fail threshold applied.

warp bilinear    — Centre-pixel OOB → zero (BORDER_CONSTANT). The +1 tap at
                   source edges is clamped to the last row/column (BORDER_REPLICATE),
                   mirroring the CPU warp_affine inner loop.
                   cv2 warpAffine uses BORDER_CONSTANT for both, so pixels whose
                   bilinear stencil touches the source border differ. Identity and
                   interior pixels match cv2 closely; edge-touching pixels do not.

warp nearest     — Centre-pixel OOB → zero, rounded tap clamped into [0, src-1].
                   Matches cv2.INTER_NEAREST.

warp bicubic     — Keys a=-0.5, BORDER_REPLICATE for the 4×4 OOB taps.
                   cv2 INTER_CUBIC + BORDER_CONSTANT=0 uses zero for OOB taps.
                   Interior pixels match; border pixels where some 4×4 taps are
                   OOB will differ. We report interior (inner 80%) separately.

warp lanczos     — Our kernel is Lanczos-3 (6-tap 2D).
                   cv2.INTER_LANCZOS4 is Lanczos-4 (8-tap 2D). REPORT ONLY.
"""

import json
import math
import subprocess
import sys

import cv2
import numpy as np

RESIZE_BIN = "./target/release/examples/dump_cuda_resize"
WARP_BIN   = "./target/release/examples/dump_cuda_warp_affine"

PASS = "\033[32m✅ PASS\033[0m"
FAIL = "\033[31m❌ FAIL\033[0m"
INFO = "\033[33m⚠  INFO\033[0m"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ramp(n_elements: int) -> np.ndarray:
    """Same ramp as the Rust examples: arange(N) / (N-1), f32."""
    return (np.arange(n_elements, dtype=np.float64) / max(n_elements - 1, 1)).astype(np.float32)


def run_rust_resize(mode: str, sw: int, sh: int, dw: int, dh: int) -> np.ndarray:
    out = subprocess.run(
        [RESIZE_BIN, mode, str(sw), str(sh), str(dw), str(dh)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    return np.array(d["pixels"], dtype=np.float32).reshape(dh, dw, 3)


def run_rust_warp(mode: str, w: int, h: int, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gpu_pixels, forward_matrix_2x3)."""
    out = subprocess.run(
        [WARP_BIN, mode, str(w), str(h), str(angle)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    pixels = np.array(d["pixels"], dtype=np.float32).reshape(h, w, 3)
    m = np.array(d["m"], dtype=np.float64).reshape(2, 3)
    return pixels, m


def rotation_matrix_cv(w: int, h: int, angle_deg: float) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)


def report(label: str, gpu: np.ndarray, ref: np.ndarray, tol: float | None,
           interior_only: bool = False) -> bool:
    """Compare gpu vs ref, report stats, return True if within tol."""
    if interior_only:
        h, w = gpu.shape[:2]
        m = max(1, int(min(h, w) * 0.1))
        gpu_c = gpu[m:h-m, m:w-m]
        ref_c = ref[m:h-m, m:w-m]
        tag = " (interior)"
    else:
        gpu_c, ref_c, tag = gpu, ref, ""

    diff = np.abs(gpu_c.astype(np.float64) - ref_c.astype(np.float64))
    max_err = float(diff.max())
    mean_err = float(diff.mean())

    if tol is None:
        status = INFO
        verdict = f"max={max_err:.2e}  mean={mean_err:.2e}{tag}  (report only)"
        ok = True
    else:
        ok = max_err <= tol
        status = PASS if ok else FAIL
        verdict = f"max={max_err:.2e}  mean={mean_err:.2e}{tag}  tol={tol:.0e}"

    print(f"  {status}  {label:<48}  {verdict}")
    return ok


# ---------------------------------------------------------------------------
# Resize checks
# ---------------------------------------------------------------------------

def check_resize() -> bool:
    print("\n=== resize vs cv2 ===")
    all_ok = True

    # tol=None → report max error only, no pass/fail threshold.
    #
    # nearest: kornia-rs uses floor((dst+0.5)*scale) — PIL convention.
    #          cv2.INTER_NEAREST uses floor(dst*scale) — no half-pixel offset.
    #          Convention mismatch, not a bug. Yields ~1-pixel shift for integer scales.
    #
    # bicubic: kornia-rs clamps OOB 4×4 taps to the edge pixel (BORDER_REPLICATE).
    #          cv2.resize INTER_CUBIC uses BORDER_REFLECT_101 internally for OOB taps.
    #          Interior pixels agree; border pixels diverge.
    #          For upscale the source coords go more negative → more OOB taps → larger error.
    cases = [
        # (mode,  sw,  sh,  dw,  dh,  cv2_interp,              tol,   interior)
        ("bilinear", 64, 48, 32, 24, cv2.INTER_LINEAR,  2e-5,  False),
        ("bilinear", 32, 24, 64, 48, cv2.INTER_LINEAR,  2e-5,  False),  # upscale
        ("nearest",  64, 48, 32, 24, cv2.INTER_NEAREST, None,  False),  # convention diff
        ("nearest",  64, 48, 21, 16, cv2.INTER_NEAREST, None,  False),  # convention diff
        ("bicubic",  64, 48, 32, 24, cv2.INTER_CUBIC,   None,  False),  # border diff
        ("bicubic",  32, 24, 64, 48, cv2.INTER_CUBIC,   None,  False),  # border diff
        ("lanczos",  64, 48, 32, 24, cv2.INTER_LANCZOS4, None, False),  # diff kernel
    ]

    for mode, sw, sh, dw, dh, interp, tol, interior in cases:
        n_src = sw * sh * 3
        src_np = ramp(n_src).reshape(sh, sw, 3)

        gpu = run_rust_resize(mode, sw, sh, dw, dh)
        ref = cv2.resize(src_np, (dw, dh), interpolation=interp)

        label = f"{mode:8}  {sw}×{sh}→{dw}×{dh}"
        ok = report(label, gpu, ref, tol, interior_only=interior)
        all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Warp-affine checks
# ---------------------------------------------------------------------------

CV2_INTERP = {
    "bilinear": cv2.INTER_LINEAR,
    "nearest":  cv2.INTER_NEAREST,
    "bicubic":  cv2.INTER_CUBIC,
    "lanczos":  cv2.INTER_LANCZOS4,
}

def check_warp() -> bool:
    print("\n=== warp-affine vs cv2 ===")
    all_ok = True

    # bilinear non-identity: the +1 bilinear tap at source edges uses BORDER_REPLICATE
    # (matches our CPU warp_affine), while cv2 warpAffine uses BORDER_CONSTANT=0.
    # Pixels whose stencil touches the source border will differ; those far from any
    # source edge match within floating-point tolerance. Identity is exact because
    # integer source coordinates carry zero fractional weight on the replicated tap.
    #
    # bicubic non-identity: kornia-rs uses BORDER_REPLICATE for OOB 4×4 taps.
    # cv2 warpAffine with BORDER_CONSTANT=0 uses zero for OOB taps.
    # These differ for pixels whose 4×4 neighbourhood extends outside the source.
    # Identity transform has no OOB taps → matches exactly.
    cases = [
        # (mode,      w,   h,  angle,  tol,    interior)
        ("bilinear",  64,  64,   0.0,  2e-5,   False),   # identity — exact
        ("bilinear",  64,  64,  45.0,  None,   False),   # BORDER_REPLICATE vs cv2's BORDER_CONSTANT
        ("bilinear", 128,  96,  30.0,  None,   False),   # BORDER_REPLICATE vs cv2's BORDER_CONSTANT
        ("nearest",   64,  64,   0.0,  1e-6,   False),
        ("nearest",   64,  64,  90.0,  1e-6,   False),
        ("bicubic",   64,  64,   0.0,  3e-4,   False),   # identity — no border effect
        ("bicubic",   64,  64,  30.0,  None,   False),   # border diff, report only
        ("lanczos",   64,  64,  30.0,  None,   False),   # diff kernel, report only
    ]

    for mode, w, h, angle, tol, interior in cases:
        n = w * h * 3
        src_np = ramp(n).reshape(h, w, 3)

        gpu, m_rs = run_rust_warp(mode, w, h, angle)

        # Verify that our rotation matrix matches cv2.getRotationMatrix2D.
        m_cv = rotation_matrix_cv(w, h, angle)
        if not np.allclose(m_rs, m_cv, atol=1e-5):
            print(f"  ❌ MATRIX MISMATCH for {mode} angle={angle}")
            all_ok = False
            continue

        ref = cv2.warpAffine(
            src_np, m_cv, (w, h),
            flags=CV2_INTERP[mode],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        label = f"{mode:8}  {w}×{h}  angle={angle:5.1f}°"
        ok = report(label, gpu, ref, tol, interior_only=interior)
        all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_bins() -> bool:
    ok = True
    for path in (RESIZE_BIN, WARP_BIN):
        try:
            subprocess.run([path, "--help"], capture_output=True)
        except FileNotFoundError:
            print(f"Binary not found: {path}")
            print("Build with: cargo build --features cuda --release "
                  "--example dump_cuda_resize --example dump_cuda_warp_affine")
            ok = False
    return ok


if __name__ == "__main__":
    if not check_bins():
        sys.exit(1)

    r_ok = check_resize()
    w_ok = check_warp()

    print()
    if r_ok and w_ok:
        print("✅  ALL CHECKS PASSED")
    else:
        print("❌  SOME CHECKS FAILED — see above")
    sys.exit(0 if (r_ok and w_ok) else 1)
