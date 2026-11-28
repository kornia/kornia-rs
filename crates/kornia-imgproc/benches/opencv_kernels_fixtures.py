"""Write the OpenCV reference outputs of the kornia-imgproc parity tests for
`good_features_to_track`, `corner_sub_pix`, the window LK and `remap_u8_table`.

Usage:
  uv run --no-project --with opencv-python-headless==5.0.0.93 --with numpy \
      python3 crates/kornia-imgproc/benches/opencv_kernels_fixtures.py <kornia-rs root>

Pinned to opencv-python-headless 5.0.0.93 (cv2 5.0.0), which wrote the committed
fixtures; the script prints the cv2 version it ran. Inputs are `tests/data/mh01_frame1.png` and
`mh01_frame2.png` (752x480, EuRoC MH_01 cam0), read by the tests too, so the
fixtures do not embed them. One thread.

`tests/data/frontend_kernels_opencv.bin`, little endian:
  u32 n
  f32[n][2]  goodFeaturesToTrack(frame1, 200, 0.01, 23, mask = 9 px border, blockSize 3)
  f32[n][2]  cornerSubPix(frame1, those corners, winSize (5, 5), zeroZone (-1, -1),
             COUNT|EPS 30 / 0.01)
  f32[n][2]  calcOpticalFlowPyrLK(frame1, frame2, the refined corners, winSize (11, 11),
             maxLevel 0, COUNT|EPS 8 / 0.01, minEigThreshold 1e-4) next points
  u8[n]      its status

`tests/data/remap_q5_opencv.bin`, little endian:
  u32 w, u32 h  (376, 240)
  u8[h][w]      remap(frame1, convertMaps(map_x, map_y, CV_16SC2), INTER_LINEAR,
                BORDER_CONSTANT 0)
  with map_x = xs*1.96 + ys*0.031 + 3.3 and map_y = ys*1.93 - xs*0.017 + 5.7,
  every term np.float32 and evaluated left to right, as the Rust test does.
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_CORNERS, QUALITY, MIN_DISTANCE, BORDER = 200, 0.01, 23, 9
SUBPIX_HALF_WIN, SUBPIX_CRIT = 5, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
LK_WIN, LK_LEVELS, LK_CRIT = 11, 0, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 8, 0.01)
REMAP_W, REMAP_H = 376, 240


def kernels(f1: np.ndarray, f2: np.ndarray) -> bytes:
    h, w = f1.shape
    mask = np.zeros_like(f1)
    mask[BORDER : h - BORDER, BORDER : w - BORDER] = 255
    corners = cv2.goodFeaturesToTrack(f1, MAX_CORNERS, QUALITY, MIN_DISTANCE, mask=mask, blockSize=3)
    corners = corners.reshape(-1, 2).astype(np.float32)
    refined = corners.copy().reshape(-1, 1, 2)
    cv2.cornerSubPix(f1, refined, (SUBPIX_HALF_WIN, SUBPIX_HALF_WIN), (-1, -1), SUBPIX_CRIT)
    refined = refined.reshape(-1, 2).astype(np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        f1,
        f2,
        refined.reshape(-1, 1, 2),
        None,
        winSize=(LK_WIN, LK_WIN),
        maxLevel=LK_LEVELS,
        criteria=LK_CRIT,
        minEigThreshold=1e-4,
    )
    blob = struct.pack("<I", len(corners)) + corners.astype("<f4").tobytes() + refined.astype("<f4").tobytes()
    blob += nxt.reshape(-1, 2).astype("<f4").tobytes() + st.reshape(-1).astype(np.uint8).tobytes()
    print(f"{len(corners)} corners, {int(st.sum())} tracked")
    return blob


def remap_q5(f1: np.ndarray) -> bytes:
    ys, xs = np.mgrid[0:REMAP_H, 0:REMAP_W].astype(np.float32)
    f = np.float32
    map_x = xs * f(1.96) + ys * f(0.031) + f(3.3)
    map_y = ys * f(1.93) - xs * f(0.017) + f(5.7)
    assert map_x.dtype == np.float32 and map_y.dtype == np.float32
    fixed_xy, fixed_frac = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    out = cv2.remap(f1, fixed_xy, fixed_frac, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return struct.pack("<II", REMAP_W, REMAP_H) + out.astype(np.uint8).tobytes()


def main(root: Path) -> None:
    cv2.setNumThreads(1)
    data = root / "tests" / "data"
    f1 = cv2.imread(str(data / "mh01_frame1.png"), cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread(str(data / "mh01_frame2.png"), cv2.IMREAD_GRAYSCALE)
    assert f1 is not None and f2 is not None and f1.shape == (480, 752) and f2.shape == f1.shape
    for name, blob in [("frontend_kernels_opencv.bin", kernels(f1, f2)), ("remap_q5_opencv.bin", remap_q5(f1))]:
        (data / name).write_bytes(blob)
        print(f"{data / name}: {len(blob)} bytes")
    print(f"cv2 {cv2.__version__}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
