#!/usr/bin/env python3
"""Bit-exact correctness check: kornia find_contours output vs cv2.findContours.

Compares contour POINTS (not just count) — each contour must contain the same
ordered list of (x, y) coordinates.
"""
import json
import subprocess
import sys

import cv2
import numpy as np


def kornia_contours(image_path, mode="external", method="simple"):
    """Run kornia find_contours via the dump-contours example, return a list of contours."""
    out = subprocess.run(
        ["./target/release/examples/dump_contours", image_path, mode, method],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    return [np.array(c, dtype=np.int32).reshape(-1, 2) for c in data["contours"]]


def opencv_contours(image_path, mode_str, method_str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    mode = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST}[mode_str]
    method = {"simple": cv2.CHAIN_APPROX_SIMPLE, "none": cv2.CHAIN_APPROX_NONE}[method_str]
    contours, _ = cv2.findContours(bw, mode, method)
    return [c.reshape(-1, 2) for c in contours]


def normalize(contours):
    """Sort contours by their leftmost-topmost point so order doesn't matter."""
    keyed = []
    for c in contours:
        if len(c) == 0:
            continue
        # Sort key: lex-min point of the contour
        idx = np.lexsort((c[:, 1], c[:, 0]))
        keyed.append((tuple(c[idx[0]]), tuple(map(tuple, c.tolist()))))
    keyed.sort(key=lambda t: t[0])
    return keyed


def compare(image_path, mode, method):
    k = kornia_contours(image_path, mode, method)
    o = opencv_contours(image_path, mode, method)
    print(f"\n=== {image_path}  mode={mode}  method={method} ===")
    print(f"  kornia: {len(k)} contours, total points = {sum(len(c) for c in k)}")
    print(f"  opencv: {len(o)} contours, total points = {sum(len(c) for c in o)}")
    if len(k) != len(o):
        print(f"  ❌ COUNT MISMATCH ({len(k)} vs {len(o)})")
        return False
    kn, on = normalize(k), normalize(o)
    eq = sum(1 for ki, oi in zip(kn, on) if ki[1] == oi[1])
    print(f"  bit-exact contours: {eq}/{len(kn)}")
    if eq == len(kn):
        print("  ✅ BIT-EXACT MATCH")
        return True
    # Diff details
    for i, (ki, oi) in enumerate(zip(kn, on)):
        if ki[1] != oi[1]:
            print(f"    contour {i}: kornia start {ki[0]}, lens k={len(ki[1])} o={len(oi[1])}")
            if len(ki[1]) <= 8 and len(oi[1]) <= 8:
                print(f"      kornia points: {ki[1]}")
                print(f"      opencv points: {oi[1]}")
            if i >= 3:
                print(f"    ... ({len(kn) - i - 1} more diffs not shown)")
                break
    return False


if __name__ == "__main__":
    images = [
        "crates/kornia-imgproc/examples/data/pic1.png",
        "crates/kornia-imgproc/examples/data/pic3.png",
        "crates/kornia-imgproc/examples/data/pic4.png",
    ]
    all_ok = True
    for img in images:
        for mode in ("external", "list"):
            for method in ("simple", "none"):
                ok = compare(img, mode, method)
                all_ok = all_ok and ok
    print()
    print("✅ ALL MATCH" if all_ok else "❌ DIFFS FOUND")
    sys.exit(0 if all_ok else 1)
