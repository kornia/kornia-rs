"""Resize + fused-preprocess comparison: kornia-rs vs OpenCV vs PIL (vs NVIDIA
VPI-CPU when importable), across resolutions — wall time AND byte fidelity.

Run:  python3 kornia-py/benchmarks/bench_resize_vs_libs.py

Reports, per resolution case:
  * p50 wall time (ms) for nearest / bilinear / lanczos (AA + non-AA) per lib,
    plus the full model-preprocess chain (resize + /255 + mean/std + HWC->CHW
    f32): each lib + numpy vs kornia's fused `resize_normalize_to_tensor`.
  * Byte fidelity: identical%, max, and mean abs diff vs the reference with the
    same semantics (PIL for center-convention nearest and AA lanczos, cv2 for
    bilinear), plus rounding accuracy vs a float64 bilinear ground truth
    (torch, antialias=False) when torch is importable.

Semantics map (why the pairs are what they are):
  * kornia nearest        == PIL NEAREST (center convention; cv2 uses floor)
  * kornia bilinear       == cv2 INTER_LINEAR (max 1 LSB; kornia is Q14,
                             cv2 Q11 — vs f64 truth kornia rounds MORE bytes
                             correctly)
  * kornia lanczos AA     == PIL LANCZOS (widened kernel on downscale)
  * kornia lanczos no-AA  ~= cv2 INTER_LANCZOS4 (fixed kernel; 3-lobe vs
                             4-lobe, so bytes differ by design)
"""
import time

import kornia_rs
import numpy as np
import cv2
from PIL import Image as PILImage
from kornia_rs.image import Image

MEAN = np.array(kornia_rs.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(kornia_rs.IMAGENET_STD, dtype=np.float32)
CASES = [
    ((3840, 2160), (1920, 1080)),
    ((1920, 1080), (640, 480)),
    ((1280, 720), (512, 512)),
    ((640, 480), (224, 224)),
]


def timeit(iters, f):
    for _ in range(3):
        f()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        f()
        t.append((time.perf_counter() - t0) * 1e3)
    t.sort()
    return t[len(t) // 2], t[min(len(t) - 1, int(len(t) * 0.99))]


def p50(iters, f):
    return timeit(iters, f)[0]


def np_chain(r):
    return np.ascontiguousarray(
        (((r.astype(np.float32) / 255.0) - MEAN) / STD).transpose(2, 0, 1)
    )


def fidelity(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return f"{float((d == 0).mean()) * 100:5.1f}% ident, max {int(d.max()):3d}, mean {float(d.mean()):.3f}"


def main():
    rng = np.random.default_rng(42)
    for (sw, sh), (dw, dh) in CASES:
        src = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
        img = Image(src)
        pil = PILImage.fromarray(src)
        it = 20 if sw >= 3840 else 50
        print(f"\n== {sw}x{sh} -> {dw}x{dh} (RGB u8, p50 ms) ==")
        rows = [
            ("nearest        cv2", p50(it, lambda: cv2.resize(src, (dw, dh), interpolation=cv2.INTER_NEAREST))),
            ("nearest     kornia", p50(it, lambda: img.resize(dw, dh, "nearest").numpy())),
            ("bilinear       cv2", p50(it, lambda: cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR))),
            ("bilinear    kornia", p50(it, lambda: img.resize(dw, dh, "bilinear").numpy())),
            ("lanczos4(noAA) cv2", p50(it // 2, lambda: cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LANCZOS4))),
            ("lanczos noAA kornia", p50(it // 2, lambda: img.resize(dw, dh, "lanczos", False).numpy())),
            ("lanczos AA     PIL", p50(4, lambda: np.asarray(pil.resize((dw, dh), PILImage.LANCZOS)))),
            ("lanczos AA  kornia", p50(it // 2, lambda: img.resize(dw, dh, "lanczos").numpy())),
            ("chain cv2+numpy   ", p50(8, lambda: np_chain(cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)))),
            ("chain kornia FUSED", p50(it, lambda: img.resize_normalize_to_tensor(dw, dh, list(MEAN), list(STD)))),
        ]
        for name, ms in rows:
            print(f"  {name:22} {ms:8.3f}")

        # Batched fused chain (dataloader shape): 8 images per call, p50 & p99
        # per image. Real-time systems are judged at p99, so report it.
        from kornia_rs.pipeline import resize_normalize_to_tensor_batch

        batch = [src] * 8
        b50, b99 = timeit(
            max(4, it // 4),
            lambda: resize_normalize_to_tensor_batch(batch, (dh, dw), list(MEAN), list(STD)),
        )
        f50, f99 = timeit(it, lambda: img.resize_normalize_to_tensor(dw, dh, list(MEAN), list(STD)))
        print(f"  {'chain FUSED p50/p99':22} {f50:8.3f} / {f99:.3f}")
        print(f"  {'batch x8 per-img p50/p99':26} {b50 / 8:6.3f} / {b99 / 8:.3f}")

        print("  -- byte fidelity --")
        print("  nearest  vs PIL     ", fidelity(img.resize(dw, dh, "nearest").numpy(), np.asarray(pil.resize((dw, dh), PILImage.NEAREST))))
        print("  bilinear vs cv2     ", fidelity(img.resize(dw, dh, "bilinear").numpy(), cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)))
        print("  lanczosAA vs PIL    ", fidelity(img.resize(dw, dh, "lanczos").numpy(), np.asarray(pil.resize((dw, dh), PILImage.LANCZOS))))
        try:
            import torch
            import torch.nn.functional as F

            x = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0).double()
            t = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False, antialias=False)
            t = t.squeeze(0).permute(1, 2, 0).numpy()

            def truth_score(o):
                err = np.abs(o.astype(np.float64) - t)
                return f"{float((err <= 0.5).mean()) * 100:5.1f}% correctly rounded, mean err {float(err.mean()):.4f}"

            print("  bilinear kornia vs f64 truth", truth_score(img.resize(dw, dh, "bilinear").numpy()))
            print("  bilinear cv2    vs f64 truth", truth_score(cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)))
        except ImportError:
            pass


if __name__ == "__main__":
    main()
