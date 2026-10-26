"""Baselines for `erode_label_boundaries` (see `bench_label_boundaries.rs`).

Times the equivalent pipelines on the same synthetic label maps:

* OpenCV: seed mask `compare(dilate(L, k3), erode(L, k3), CMP_NE)` (3x3
  cross for 4-connectivity, 3x3 rectangle for 8), dilated by a diamond /
  square of radius `r - 1`, then `fill` written through the mask; timed
  on one thread and on all of OpenCV's threads.
* scikit-image: `find_boundaries(L, connectivity, mode="thick")`, then
  `binary_dilation` with a diamond / square of radius `r - 1`.

Both are checked against each other before timing. Run with the system
python3 (needs numpy, opencv-python; scikit-image optional):

    python3 bench_label_boundaries_baselines.py

`--write-fixture DIR` (needs scikit-image) instead writes
`label_boundaries_skimage.bin`, the parity fixture read by the
`erode_label_boundaries` tests, into DIR (the repository's `tests/data`).
"""

import pathlib
import struct
import sys
import time

import cv2
import numpy as np

SIZES = [(640, 480), (1280, 720)]
RADII = [1, 2, 5]
ROUNDS = 5
ITERS = 50


def labels_u16(w: int, h: int) -> np.ndarray:
    """Same Voronoi-like fixture as the Rust bench."""
    state = 0x9E3779B97F4A7C15
    mask64 = (1 << 64) - 1

    def nxt() -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & mask64
        return state >> 33

    n = max(w * h // 5000, 2)
    sites = []
    for _ in range(n):
        sx = nxt() % w
        sy = nxt() % h
        sites.append((sx, sy))
    ys, xs = np.mgrid[0:h, 0:w]
    best = np.full((h, w), np.iinfo(np.int64).max, dtype=np.int64)
    out = np.zeros((h, w), dtype=np.uint16)
    for k, (sx, sy) in enumerate(sites):
        d = (xs - sx).astype(np.int64) ** 2 + (ys - sy).astype(np.int64) ** 2
        closer = d < best
        best[closer] = d[closer]
        out[closer] = k + 1
    return out


def footprint(radius: int, eight: bool) -> np.ndarray:
    """Diamond (Manhattan) or square (Chebyshev) of the given radius."""
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    if eight:
        return np.ones_like(yy, dtype=np.uint8)
    return (np.abs(yy) + np.abs(xx) <= radius).astype(np.uint8)


def opencv_erode(labels: np.ndarray, radius: int, eight: bool, fill: int) -> np.ndarray:
    k3 = footprint(1, eight)
    seeds = cv2.compare(cv2.dilate(labels, k3), cv2.erode(labels, k3), cv2.CMP_NE)
    if radius > 1:
        seeds = cv2.dilate(seeds, footprint(radius - 1, eight))
    out = labels.copy()
    out[seeds.view(bool)] = fill
    return out


def skimage_erode(labels: np.ndarray, radius: int, eight: bool, fill: int) -> np.ndarray:
    from scipy import ndimage
    from skimage.segmentation import find_boundaries

    seeds = find_boundaries(labels, connectivity=2 if eight else 1, mode="thick")
    if radius > 1:
        seeds = ndimage.binary_dilation(seeds, structure=footprint(radius - 1, eight))
    return np.where(seeds, labels.dtype.type(fill), labels)


def best_ms(fn) -> float:
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        best = min(best, (time.perf_counter() - t0) * 1e3 / ITERS)
    return best


FIXTURE_RADII = (1, 3)


def write_fixture(out_dir: str) -> None:
    """A small random label map and scikit-image's masks for it.

    Little endian: `u64` width, `u64` height, the `u16` labels row-major,
    then one `u8` mask (1 = eroded) per (connectivity 1, 2) x
    `FIXTURE_RADII` pair, in that order.
    """
    rng = np.random.default_rng(7)
    w, h = 61, 37
    # Blocks of 4..9 pixels with random labels (some neighbours share one),
    # plus single-pixel speckle.
    ys = np.repeat(np.arange(h), rng.integers(4, 10, size=h))[:h]
    xs = np.repeat(np.arange(w), rng.integers(4, 10, size=w))[:w]
    block = rng.integers(0, 5, size=(h, w))
    labels = block[ys[:, None], xs[None, :]].astype(np.uint16)
    speckle = rng.random((h, w)) < 0.02
    labels[speckle] = rng.integers(0, 5, size=int(speckle.sum()))
    labels = labels * 1000 + 7
    blob = struct.pack("<QQ", w, h) + labels.astype("<u2").tobytes()
    for eight in (False, True):
        for radius in FIXTURE_RADII:
            out = skimage_erode(labels, radius, eight, 0xFFFF)
            blob += (out == 0xFFFF).astype(np.uint8).tobytes()
    pathlib.Path(out_dir, "label_boundaries_skimage.bin").write_bytes(blob)
    print(f"wrote {w}x{h}, {len(blob)} bytes")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--write-fixture":
        write_fixture(sys.argv[2])
        return
    try:
        import skimage  # noqa: F401

        have_skimage = True
    except ImportError:
        have_skimage = False
    threads = cv2.getNumThreads()
    print(f"opencv {cv2.__version__}, threads 1 / {threads}; scikit-image single-threaded")
    for w, h in SIZES:
        l16 = labels_u16(w, h)
        l8 = l16.astype(np.uint8)
        for eight in (False, True):
            tag = "c8" if eight else "c4"
            for radius in RADII:
                for name, lab in (("u8", l8), ("u16", l16)):
                    ref = opencv_erode(lab, radius, eight, 0)
                    line = f"{w}x{h} {name}_{tag}_r{radius}: "
                    for n in (1, threads):
                        cv2.setNumThreads(n)
                        ms = best_ms(lambda: opencv_erode(lab, radius, eight, 0))
                        line += f"opencv[{n}t] {ms:.3f} ms, "
                    if have_skimage:
                        assert np.array_equal(ref, skimage_erode(lab, radius, eight, 0))
                        ms = best_ms(lambda: skimage_erode(lab, radius, eight, 0))
                        line += f"skimage {ms:.3f} ms"
                    print(line)


if __name__ == "__main__":
    main()
