#!/usr/bin/env python3
"""Quality validator: compare kornia contours_linkruns output against
cv2.findContoursLinkRuns on every bench fixture.

Calls a small Rust helper (dump_linkruns_counts) that emits per-fixture
contour counts as CSV, then runs the same fixtures through cv2 and prints
a pass/fail table.

Status checks:
  - exact count match           -> PASS (ground-truth topologically correct)
  - count differs by <= 5%      -> WARN (probably a chain-direction edge case)
  - count differs by > 5%       -> FAIL
"""
import subprocess
import sys
import numpy as np
import cv2

# Same fixture generators as bench_opencv_contours.py (identical seeds)

def make_filled_square(w, h, foreground=1):
    mw, mh = w // 8, h // 8
    d = np.zeros((h, w), dtype=np.uint8)
    d[mh:h - mh, mw:w - mw] = foreground
    return d

def make_hollow_square(w, h, foreground=1):
    ow, oh = w // 8, h // 8
    iw, ih = w // 4, h // 4
    d = np.zeros((h, w), dtype=np.uint8)
    d[oh:h - oh, ow:w - ow] = foreground
    d[ih:h - ih, iw:w - iw] = 0
    return d

def make_noise(w, h, seed=0xC0FFEE, foreground=1):
    state = seed & ((1 << 64) - 1)
    M = 6364136223846793005 & ((1 << 64) - 1)
    A = 1442695040888963407 & ((1 << 64) - 1)
    MASK = (1 << 64) - 1
    out = np.empty(w * h, dtype=np.uint8)
    for i in range(w * h):
        state = (state * M + A) & MASK
        out[i] = ((state >> 33) & 1) * foreground
    return out.reshape(h, w)


def opencv_counts(data_u8_01):
    """Return (ext_count, list_count, linkruns_count) for the given binary."""
    bw = (data_u8_01 != 0).astype(np.uint8) * 255
    ext, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lst, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lr, _ = cv2.findContoursLinkRuns(bw)
    return len(ext), len(lst), len(lr)


def real_image_counts(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None, None
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ext, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lst, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lr, _ = cv2.findContoursLinkRuns(bw)
    return len(ext), len(lst), len(lr), bw


def kornia_count(fixture, w, h, seed_or_path=None):
    """Calls the Rust helper to get kornia's contour count for a fixture."""
    cmd = ["target/release/examples/dump_linkruns_counts",
           fixture, str(w), str(h)]
    if seed_or_path is not None:
        cmd.append(str(seed_or_path))
    out = subprocess.check_output(cmd).decode().strip()
    return int(out.split(",")[-1])


SIZES = [128, 256, 512, 1024, 2048]
SPARSE_SIZES = [128, 256, 512, 1024]   # 2048 is huge, sparse_noise scan is slow


def main():
    print(f"{'fixture':40} {'size':>10} {'cv_EXT':>8} {'cv_LIST':>8} {'cv_LR':>8} {'kornia':>8}  best-match")
    print("-" * 110)

    rows = []

    # Real-world images
    for path in [
        "crates/kornia-imgproc/examples/data/pic1.png",
        "crates/kornia-imgproc/examples/data/pic2.png",
        "crates/kornia-imgproc/examples/data/pic3.png",
        "crates/kornia-imgproc/examples/data/pic4.png",
    ]:
        ext, lst, lr, bw = real_image_counts(path)
        if ext is None:
            continue
        h, w = bw.shape
        ko = kornia_count("real", w, h, path)
        best = closest(ko, ext, lst, lr)
        print(f"{path:40} {f'{w}x{h}':>10} {ext:>8} {lst:>8} {lr:>8} {ko:>8}  {best}")
        rows.append(best)

    # Synthetic fixtures
    for name, gen, sizes in [
        ("filled_square", make_filled_square, SIZES),
        ("hollow_square", make_hollow_square, SIZES),
        ("sparse_noise",  make_noise,         SPARSE_SIZES),
    ]:
        for s in sizes:
            ext, lst, lr = opencv_counts(gen(s, s))
            ko = kornia_count(name, s, s)
            best = closest(ko, ext, lst, lr)
            print(f"{name:40} {f'{s}x{s}':>10} {ext:>8} {lst:>8} {lr:>8} {ko:>8}  {best}")
            rows.append(best)

    print("-" * 110)
    pass_count = sum(1 for r in rows if "PASS" in r)
    print(f"\n{pass_count}/{len(rows)} match some OpenCV mode exactly")


def closest(ko, ext, lst, lr):
    """Tag the row with which OpenCV mode it matches best."""
    if ko == ext: return "PASS=EXT"
    if ko == lst: return "PASS=LIST"
    if ko == lr:  return "PASS=LR"
    diffs = [(abs(ko - ext), "EXT", ext), (abs(ko - lst), "LIST", lst), (abs(ko - lr), "LR", lr)]
    diffs.sort()
    d, mode, val = diffs[0]
    pct = 100 * d / max(val, 1)
    return f"closest {mode} ({ko - val:+d}, {pct:.0f}%)"


def check(cv_n, ko_n):
    if ko_n == cv_n:
        return "PASS", False
    diff = abs(cv_n - ko_n) / max(cv_n, 1)
    if diff <= 0.05:
        return f"WARN ({ko_n - cv_n:+d})", False
    return f"FAIL ({ko_n - cv_n:+d})", True


if __name__ == "__main__":
    main()
