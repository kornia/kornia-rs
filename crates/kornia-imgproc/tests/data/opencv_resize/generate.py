"""Generate cv2.resize reference vectors for the kornia-rs OpenCV-compat mode.

Deterministic LCG inputs (matching kornia's test pattern generator) so the
fixtures are reproducible from this script alone. Output: flat little-endian
binaries + a JSON manifest.
"""
import numpy as np
import cv2

OUT = "crates/kornia-imgproc/tests/data/opencv_resize"

def lcg_u8(n):
    """Mirror kornia's pattern_u8: edge-case prefix then LCG bytes."""
    prefix = [0,255,255,0,0,0,255,255,255,1,254,128,128,128,64]
    v = list(prefix[:min(len(prefix), n)])
    state = 0x12345678
    while len(v) < n:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        v.append(state >> 24)
    return np.array(v, dtype=np.uint8)

def pattern_u8(h, w, c):
    return lcg_u8(h*w*c).reshape(h, w, c)

def pattern_f32(h, w, c):
    return (lcg_u8(h*w*c).astype(np.float32) / 255.0).reshape(h, w, c)

CASES = [
    # (src_h, src_w, dst_h, dst_w)
    (16, 16, 8, 8),        # dyadic down
    (8, 8, 16, 16),        # dyadic up
    (23, 37, 11, 17),      # odd down
    (11, 17, 23, 37),      # odd up
    (48, 64, 33, 47),      # non-integral ratio
    (5, 5, 9, 7),          # tiny asymmetric
    (32, 32, 32, 32),      # identity size
    (64, 48, 1, 1),        # collapse to 1x1
    (1, 1, 8, 8),          # expand from 1x1
]
INTERPS = {"linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}

count = 0
for (sh, sw, dh, dw) in CASES:
    for cn in (1, 3):
        for dtype in ("u8", "f32"):
            src = pattern_u8(sh, sw, cn) if dtype == "u8" else pattern_f32(sh, sw, cn)
            src_sq = src[..., 0] if cn == 1 else src  # cv2 wants HxW for 1ch
            for iname, iflag in INTERPS.items():
                dst = cv2.resize(src_sq, (dw, dh), interpolation=iflag)
                if cn == 1:
                    dst = dst.reshape(dh, dw, 1)
                # All parameters live in the filename; the Rust test parses it,
                # so no manifest file is needed.
                key = f"{dtype}_c{cn}_{sh}x{sw}_to_{dh}x{dw}_{iname}"
                src.tofile(f"{OUT}/{key}.src")
                np.ascontiguousarray(dst).tofile(f"{OUT}/{key}.dst")
                count += 1

print(f"{count} fixtures, cv2 {cv2.__version__}")
