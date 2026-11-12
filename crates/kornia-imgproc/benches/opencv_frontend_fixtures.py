"""Write `tests/data/frontend_opencv.bin`: OpenCV reference outputs for the
fixed-point LK and pyrDown parity tests in kornia-imgproc.

Usage: python3 opencv_frontend_fixtures.py <kornia-rs root>

Inputs are `tests/data/mh01_frame1.png` and `mh01_frame2.png` (752x480).
Points: 150 Shi-Tomasi corners at least 10 px inside, 50 within 10 px of an
edge (where the window leaves the image). LK: 21x21 window, maxLevel 3,
COUNT|EPS 30 / 0.01, minEigThreshold 1e-4, one thread.

Layout, little endian:
  u32 n
  f32[n][2] prev points
  f32[n][2] OpenCV next points
  u8[n]     OpenCV status
  u32 w, u32 h      pyrDown crop size of the source (top-left 0, 0)
  u8[ceil(h/2)][ceil(w/2)] cv2.pyrDown(frame1[:h, :w])
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

CROP_W, CROP_H = 257, 161  # odd on purpose: exercises the reflect-101 right/bottom column


def main(root: Path) -> None:
    cv2.setNumThreads(1)
    data = root / "tests" / "data"
    f1 = cv2.imread(str(data / "mh01_frame1.png"), cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread(str(data / "mh01_frame2.png"), cv2.IMREAD_GRAYSCALE)
    assert f1.shape == (480, 752) and f2.shape == f1.shape
    h, w = f1.shape
    band = 10
    inner = np.zeros_like(f1)
    inner[band : h - band, band : w - band] = 255
    pi = cv2.goodFeaturesToTrack(f1, 150, 0.01, 7, mask=inner).reshape(-1, 2)
    pb = cv2.goodFeaturesToTrack(f1, 50, 0.001, 5, mask=255 - inner).reshape(-1, 2)
    pts = np.vstack([pi, pb]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        f1, f2, pts.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3, criteria=crit, minEigThreshold=1e-4
    )
    down = cv2.pyrDown(f1[:CROP_H, :CROP_W])
    blob = struct.pack("<I", len(pts))
    blob += pts.tobytes() + nxt.reshape(-1, 2).astype(np.float32).tobytes() + st.reshape(-1).astype(np.uint8).tobytes()
    blob += struct.pack("<II", CROP_W, CROP_H) + down.tobytes()
    (data / "frontend_opencv.bin").write_bytes(blob)
    print(f"{len(pts)} points, {int(st.sum())} tracked, pyrDown {down.shape}, {len(blob)} bytes, cv2 {cv2.__version__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
