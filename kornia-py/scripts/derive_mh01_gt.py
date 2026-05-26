"""Derive the camera-frame ground-truth relative pose for the MH_01 frame pair.

The EuRoC MH_01_easy dataset ships:
  - mav0/state_groundtruth_estimate0/data.csv — body-frame pose in world, at
    200 Hz, timestamped in ns since UNIX epoch. Columns:
      #timestamp, p_RS_R_{x,y,z}, q_RS_{w,x,y,z}, v_RS_R_{x,y,z},
      b_w_RS_S_{x,y,z}, b_a_RS_S_{x,y,z}
  - mav0/cam0/sensor.yaml — cam0 intrinsics + T_BS (body-to-cam0 4x4 transform).

Our test pair lives at timestamps 1403636633263555584 and 1403636634263555584.
The relative camera-frame pose from view 1 to view 2 is:

    T_C1_C2 = T_BS^-1 · T_WB(t1)^-1 · T_WB(t2) · T_BS

This script computes that, prints rotation-angle + unit translation direction,
and diffs against the values currently hardcoded in the Rust test + Python
benchmark (2.698°, [0.2422, -0.2330, 0.9418]).

Usage:
    python kornia-py/scripts/derive_mh01_gt.py

Inputs expected at: tests/data/euroc_mh01_gt/{data.csv, sensor.yaml}
(both gitignored — see .gitignore entry).
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data" / "euroc_mh01_gt"
GT_CSV = DATA_DIR / "data.csv"
SENSOR_YAML = DATA_DIR / "sensor.yaml"

T1 = 1403636633263555584
T2 = 1403636634263555584

HARDCODED_ROT_DEG = 2.7021
HARDCODED_T_DIR = np.array([0.2422, -0.2330, 0.9418], dtype=np.float64)


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Hamilton convention, q = [w, x, y, z]."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_pose_at_timestamp(csv_path: Path, target_ts: int) -> np.ndarray:
    """Return 4x4 T_WB at the GT row closest to target_ts (ns)."""
    best_row = None
    best_diff = None
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header[0].startswith("#"):
            raise ValueError(f"unexpected header: {header[0]!r}")
        for row in reader:
            ts = int(row[0])
            diff = abs(ts - target_ts)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_row = row
    if best_row is None:
        raise RuntimeError("empty csv")
    px, py, pz = (float(best_row[1]), float(best_row[2]), float(best_row[3]))
    qw, qx, qy, qz = (
        float(best_row[4]),
        float(best_row[5]),
        float(best_row[6]),
        float(best_row[7]),
    )
    R = quat_wxyz_to_rot(np.array([qw, qx, qy, qz], dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]
    print(
        f"  ts={best_row[0]}  nearest Δ={best_diff} ns   "
        f"p=({px:+.4f},{py:+.4f},{pz:+.4f})  q=({qw:+.4f},{qx:+.4f},{qy:+.4f},{qz:+.4f})"
    )
    return T


def read_t_bs(yaml_path: Path) -> np.ndarray:
    """Parse `T_BS` 4x4 from sensor.yaml without needing pyyaml.

    The EuRoC sensor.yaml snippet looks like:

        T_BS:
          cols: 4
          rows: 4
          data: [0.0148655, -0.999881, 0.004140, ..., 0.0, 0.0, 0.0, 1.0]
    """
    text = yaml_path.read_text()
    # Grab the bracketed list after `T_BS:` ... `data:`.
    m = re.search(r"T_BS:\s*.*?data:\s*\[([^\]]+)\]", text, re.DOTALL)
    if not m:
        raise ValueError("T_BS data array not found in sensor.yaml")
    flat = [float(x) for x in re.split(r"[,\s]+", m.group(1).strip()) if x]
    if len(flat) != 16:
        raise ValueError(f"expected 16 elements in T_BS, got {len(flat)}")
    return np.array(flat, dtype=np.float64).reshape(4, 4)


def rotation_angle_deg(R: np.ndarray) -> float:
    cos_a = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    return math.degrees(math.acos(cos_a))


def t_dir_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    t_est = t_est / np.linalg.norm(t_est)
    t_gt = t_gt / np.linalg.norm(t_gt)
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(float(t_est @ t_gt))))))


def main() -> int:
    if not GT_CSV.exists() or not SENSOR_YAML.exists():
        print(f"missing GT files under {DATA_DIR}", file=sys.stderr)
        print(
            "fetch:\n"
            "  curl -o data.csv https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv\n"
            "  curl -o sensor.yaml https://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/mav0/cam0/sensor.yaml",
            file=sys.stderr,
        )
        return 1

    print(f"--- reading body poses at {T1} and {T2}")
    T_WB_1 = read_pose_at_timestamp(GT_CSV, T1)
    T_WB_2 = read_pose_at_timestamp(GT_CSV, T2)

    print(f"--- reading T_BS from {SENSOR_YAML.name}")
    T_BS = read_t_bs(SENSOR_YAML)
    print(f"T_BS =\n{T_BS}")

    # Camera-frame relative pose: T_C1_C2 = T_SB · T_BW1 · T_WB2 · T_BS
    # (T_SB = T_BS^-1 because EuRoC's T_BS maps body → sensor/camera)
    T_SB = np.linalg.inv(T_BS)
    T_BW_1 = np.linalg.inv(T_WB_1)
    T_C1_C2 = T_SB @ T_BW_1 @ T_WB_2 @ T_BS

    R = T_C1_C2[:3, :3]
    t = T_C1_C2[:3, 3]

    angle = rotation_angle_deg(R)
    t_mag = float(np.linalg.norm(t))
    t_dir = t / t_mag

    print("\n==== derived GT ====")
    print(f"rotation angle:         {angle:8.4f}°")
    print(f"translation magnitude:  {t_mag:8.4f} m")
    print(f"translation direction:  [{t_dir[0]:+.4f}, {t_dir[1]:+.4f}, {t_dir[2]:+.4f}]")

    print("\n==== hardcoded in repo ====")
    print(f"rotation angle:         {HARDCODED_ROT_DEG:8.4f}°")
    print(
        f"translation direction:  [{HARDCODED_T_DIR[0]:+.4f}, "
        f"{HARDCODED_T_DIR[1]:+.4f}, {HARDCODED_T_DIR[2]:+.4f}]"
    )

    rot_err = abs(angle - HARDCODED_ROT_DEG)
    t_err = t_dir_error_deg(t_dir, HARDCODED_T_DIR)
    print("\n==== diff ====")
    print(f"rotation |Δ|:           {rot_err:8.4f}°")
    print(f"translation dir angle:  {t_err:8.4f}°  (sign-folded)")

    # Pass if hardcoded is within 0.2° rotation and 2° t-direction — tight but
    # reasonable: the two are 50 Hz-sampled GT, our timestamps should align
    # exactly or within 5 ms (~0.01° of rotation at 2.7°/sec).
    if rot_err > 0.2:
        print("WARN: hardcoded rotation disagrees with derived GT by > 0.2°")
    if t_err > 2.0:
        print("WARN: hardcoded translation direction disagrees with derived GT by > 2°")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
