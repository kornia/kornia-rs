"""Generate cv2.medianBlur reference vectors for kornia's `median_blur`.

Seeded numpy inputs (u8, u16 and f32; 1 and 3 channels; shapes from 1x1 to
wider than one 256-column kernel strip) filtered by `cv2.medianBlur` with
ksize 3 and 5 (BORDER_REPLICATE, the only border it offers). The f32
inputs mix both signs, +-inf and repeated values; no NaN and no -0.0,
whose order OpenCV leaves unspecified. Output: flat little-endian binaries
named `<dtype>_c<channels>_<H>x<W>.{src,k3,k5}`, parsed by
`tests/opencv_median.rs` (no manifest needed).

Run from the repository root:

    python3 crates/kornia-imgproc/tests/data/opencv_median/generate.py
"""
import cv2
import numpy as np

OUT = "crates/kornia-imgproc/tests/data/opencv_median"
SHAPES = [(1, 1), (3, 2), (6, 7), (5, 40), (4, 300)]  # (H, W)


def pattern(rng, dtype, shape):
    if dtype == "u8":
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if dtype == "u16":
        v = rng.integers(0, 65536, size=shape, dtype=np.uint16)
        v.flat[::11] = 7  # ties
        return v
    v = (rng.standard_normal(shape) * 100.0).astype(np.float32)
    v.flat[::11] = 2.5  # ties
    v.flat[5::23] = np.inf
    v.flat[9::29] = -np.inf
    v[v == 0] = 1.0  # no signed zeros
    return v


rng = np.random.default_rng(1240)
count = 0
for dtype in ("u8", "u16", "f32"):
    for cn in (1, 3):
        for h, w in SHAPES:
            src = pattern(rng, dtype, (h, w, cn))
            src_cv = src[..., 0] if cn == 1 else src  # cv2 wants HxW for 1ch
            key = f"{dtype}_c{cn}_{h}x{w}"
            np.ascontiguousarray(src).tofile(f"{OUT}/{key}.src")
            for k in (3, 5):
                dst = cv2.medianBlur(np.ascontiguousarray(src_cv), k)
                np.ascontiguousarray(dst).tofile(f"{OUT}/{key}.k{k}")
            count += 1

print(f"{count} inputs x 2 ksizes, cv2 {cv2.__version__}, numpy {np.__version__}")
