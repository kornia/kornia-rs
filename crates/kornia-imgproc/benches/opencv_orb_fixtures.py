"""Write `tests/data/orb_opencv.bin`: OpenCV `cv2.ORB` reference outputs for the
`OrbCvDetector` parity tests in kornia-imgproc (`features/orb/cv.rs`).

Usage: python3 opencv_orb_fixtures.py <kornia-rs root>

Input: `tests/data/mh01_frame1.png` (752x480). Two parts:

A. `cv2.ORB_create(1000, 1.2, 8, 31, 0, 2, HARRIS_SCORE, 31, 20)` on the full
   frame: every keypoint and descriptor.
B. The same detector with 500 features on a 376x240 crop, plus its stages:
   each pyramid level (`cv2.resize(prev, size, INTER_LINEAR_EXACT)`, chained
   as ORB does), `cv2.FastFeatureDetector_create(20, True)` corners of each
   level, and a 64-bit FNV-1a hash of each level's 7x7 sigma-2 blur
   (`cv2.sepFilter2D` with `getGaussianKernel(7, 2, CV_32F)`, the float path
   ORB's in-place blur on its padded pyramid takes) over the rows and columns
   `[12, size - 12)` that BRIEF can read.

Before writing, the script checks that a numpy emulation of ORB's stages
(border filter, retainBest(2n) on FAST, 7x7 Harris, retainBest(n), intensity
centroid with `fastAtan2`, rBRIEF on the sepFilter2D blur) reproduces
`cv2.ORB` exactly on both images, so a kornia mismatch points at one stage.

Layout, little endian:
  keypoint record: f32 x, f32 y, u32 octave, f32 angle (deg), f32 response, u8[32] descriptor
  u32 n, record[n]                                   part A
  u32 crop_x, crop_y, crop_w, crop_h, u32 n_levels   part B
  per level: u32 w, u32 h, u8[h][w] level, u32 n_fast, (u16 x, u16 y, u8 score)[n_fast], u64 blur hash
  u32 n, record[n]                                   ORB on the crop
"""

import ctypes
import re
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

F32 = np.float32
CROP = (188, 120, 376, 240)
RING = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
        (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]
UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
BLUR_MARGIN = 12
LIBM = ctypes.CDLL("libm.so.6")
for fn in (LIBM.cosf, LIBM.sinf):
    fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]


def orb_levels(img, n_levels, scale_factor=1.2):
    levels = [img]
    for lv in range(1, n_levels):
        scale = F32(np.power(np.float64(F32(scale_factor)), lv))
        inv = F32(1.0) / scale
        size = (int(np.rint(F32(img.shape[1]) * inv)), int(np.rint(F32(img.shape[0]) * inv)))
        levels.append(cv2.resize(levels[-1], size, interpolation=cv2.INTER_LINEAR_EXACT))
    return levels


def fast_nms(a, t=20):
    """FAST-9 corners with cornerScore and 3x3 strict non-max suppression, as cv2.FAST."""
    h, w = a.shape
    a = a.astype(np.int32)
    c = a[3:h - 3, 3:w - 3]
    d = np.stack([a[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx] - c for dx, dy in RING])
    dd = np.concatenate([d, d[:9]])
    bright = np.full(c.shape, -999)
    dark = np.full(c.shape, -999)
    for k in range(16):
        arc = dd[k:k + 9]
        bright = np.maximum(bright, arc.min(0))
        dark = np.maximum(dark, (-arc).min(0))
    corner = (bright > t) | (dark > t)
    s = np.zeros((h, w), np.int32)
    s[3:h - 3, 3:w - 3] = np.where(corner, np.maximum(bright, dark) - 1, 0)
    p = np.pad(s, 1)
    keep = s > 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                keep &= s > p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    ys, xs = np.nonzero(keep)
    return xs, ys, s[ys, xs]


def features_per_level(n, n_levels, scale_factor=1.2):
    factor = F32(1.0 / np.float64(F32(scale_factor)))
    nd = F32(n) * (F32(1) - factor) / (F32(1) - F32(np.power(np.float64(factor), float(n_levels))))
    out, total = [], 0
    for _ in range(n_levels - 1):
        v = int(np.rint(nd))
        out.append(v)
        total += v
        nd = F32(nd * factor)
    return out + [max(n - total, 0)]


def harris(a, x, y, k=F32(0.04)):
    a = a.astype(np.int64)
    sa = sb = sc = 0
    for yy in range(y - 3, y + 4):
        for xx in range(x - 3, x + 4):
            ix = (a[yy, xx + 1] - a[yy, xx - 1]) * 2 + (a[yy - 1, xx + 1] - a[yy - 1, xx - 1]) + (a[yy + 1, xx + 1] - a[yy + 1, xx - 1])
            iy = (a[yy + 1, xx] - a[yy - 1, xx]) * 2 + (a[yy + 1, xx - 1] - a[yy - 1, xx - 1]) + (a[yy + 1, xx + 1] - a[yy - 1, xx + 1])
            sa += ix * ix
            sb += iy * iy
            sc += ix * iy
    scale = F32(1) / (F32(28) * F32(255))
    ssq = scale * scale * scale * scale
    fa, fb, fc = F32(sa), F32(sb), F32(sc)
    return ((fa * fb - fc * fc) - (k * (fa + fb)) * (fa + fb)) * ssq


R2D = F32(180 / np.pi)
P1, P3, P5, P7 = (F32(v) * R2D for v in (0.9997878412794807, -0.3258083974640975, 0.1555786518463281, -0.04432655554792128))


def fast_atan2(y, x):
    y, x = F32(y), F32(x)
    ax, ay, eps = abs(x), abs(y), F32(2.220446049250313e-16)
    if ax >= ay:
        c = ay / (ax + eps)
        c2 = c * c
        a = (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c
    else:
        c = ax / (ay + eps)
        c2 = c * c
        a = F32(90) - (((P7 * c2 + P5) * c2 + P3) * c2 + P1) * c
    if x < 0:
        a = F32(180) - a
    if y < 0:
        a = F32(360) - a
    return a


def ic_angle(a, x, y):
    a = a.astype(np.int64)
    m01 = 0
    m10 = sum(u * a[y, x + u] for u in range(-15, 16))
    for v in range(1, 16):
        vs = 0
        for u in range(-UMAX[v], UMAX[v] + 1):
            vp, vm = a[y + v, x + u], a[y - v, x + u]
            vs += vp - vm
            m10 += u * (vp + vm)
        m01 += v * vs
    return fast_atan2(m01, m10)


def pattern(root):
    src = (root / "crates/kornia-imgproc/src/features/orb/pattern.rs").read_text()

    def arr(name):
        body = src.split(f"const {name}")[1].split("];\n")[0]
        return np.array([[int(a), int(b)] for a, b in re.findall(r"\[(-?\d+), (-?\d+)\]", body)])

    return arr("POS0"), arr("POS1")


def brief(blurred, x, y, deg, pat):
    ang = F32(deg) * F32(np.pi / np.float64(F32(180)))
    ca, sa = F32(LIBM.cosf(ang)), F32(LIBM.sinf(ang))
    out = np.zeros(32, np.uint8)

    def val(px, py):
        xx = F32(px) * ca - F32(py) * sa
        yy = F32(px) * sa + F32(py) * ca
        return int(blurred[y + int(np.rint(yy)), x + int(np.rint(xx))])

    for j in range(256):
        if val(*pat[0][j]) < val(*pat[1][j]):
            out[j // 8] |= 1 << (j % 8)
    return out


def blur(level):
    k = cv2.getGaussianKernel(7, 2, cv2.CV_32F)
    return cv2.sepFilter2D(level, cv2.CV_8U, k, k, borderType=cv2.BORDER_REFLECT_101)


def fnv1a(region):
    h = 0xCBF29CE484222325
    for b in region.tobytes():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def emulate(img, n, pat):
    """numpy ORB; keys (octave, x, y) -> (response, angle, descriptor)."""
    levels = orb_levels(img, 8)
    out = {}
    for lv, (level, budget) in enumerate(zip(levels, features_per_level(n, 8))):
        h, w = level.shape
        xs, ys, sc = fast_nms(level)
        m = (xs >= 31) & (xs < w - 31) & (ys >= 31) & (ys < h - 31)
        xs, ys, sc = xs[m], ys[m], sc[m]
        if len(sc) > 2 * budget:
            m = sc >= np.sort(sc)[::-1][2 * budget - 1]
            xs, ys, sc = xs[m], ys[m], sc[m]
        hr = np.array([harris(level, x, y) for x, y in zip(xs, ys)], F32)
        if len(hr) > budget:
            m = hr >= np.sort(hr)[::-1][budget - 1]
            xs, ys, hr = xs[m], ys[m], hr[m]
        blurred = blur(level)
        for x, y, r in zip(xs.tolist(), ys.tolist(), hr):
            a = ic_angle(level, x, y)
            out[(lv, x, y)] = (r, a, brief(blurred, x, y, a, pat))
    return out


def run_cv(img, n):
    orb = cv2.ORB_create(n, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 20)
    kps, desc = orb.detectAndCompute(img, None)
    return kps, desc


def check(kps, desc, emu, what):
    got = {}
    for k, d in zip(kps, desc):
        s = F32(np.power(np.float64(F32(1.2)), k.octave))
        got[(k.octave, int(np.rint(k.pt[0] / s)), int(np.rint(k.pt[1] / s)))] = (k.response, k.angle, d)
    assert set(got) == set(emu), f"{what}: keypoint sets differ"
    for key, (r, a, d) in got.items():
        er, ea, ed = emu[key]
        assert (F32(r), F32(a)) == (er, ea) and np.array_equal(d, ed), f"{what}: {key}"
    print(f"{what}: numpy emulation == cv2.ORB on {len(got)} keypoints")


def records(kps, desc):
    out = struct.pack("<I", len(kps))
    for k, d in zip(kps, desc):
        out += struct.pack("<ffIff", k.pt[0], k.pt[1], k.octave, k.angle, k.response) + d.tobytes()
    return out


def main(root: Path) -> None:
    cv2.setNumThreads(1)
    data = root / "tests" / "data"
    img = cv2.imread(str(data / "mh01_frame1.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (480, 752)
    pat = pattern(root)

    kps, desc = run_cv(img, 1000)
    check(kps, desc, emulate(img, 1000, pat), "full frame")
    blob = records(kps, desc)

    cx, cy, cw, ch = CROP
    crop = np.ascontiguousarray(img[cy:cy + ch, cx:cx + cw])
    levels = orb_levels(crop, 8)
    blob += struct.pack("<IIIII", cx, cy, cw, ch, len(levels))
    fd = cv2.FastFeatureDetector_create(20, True)
    for level in levels:
        h, w = level.shape
        fast = sorted((int(k.pt[1]), int(k.pt[0]), int(k.response)) for k in fd.detect(level))
        xs, ys, sc = fast_nms(level)
        assert sorted(zip(ys.tolist(), xs.tolist(), sc.tolist())) == fast, f"FAST emulation {w}x{h}"
        m = BLUR_MARGIN
        blob += struct.pack("<II", w, h) + level.tobytes() + struct.pack("<I", len(fast))
        blob += b"".join(struct.pack("<HHB", x, y, s) for y, x, s in fast)
        blob += struct.pack("<Q", fnv1a(np.ascontiguousarray(blur(level)[m:h - m, m:w - m])))
    kps, desc = run_cv(crop, 500)
    check(kps, desc, emulate(crop, 500, pat), "crop")
    blob += records(kps, desc)

    (data / "orb_opencv.bin").write_bytes(blob)
    print(f"{len(blob)} bytes, cv2 {cv2.__version__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
