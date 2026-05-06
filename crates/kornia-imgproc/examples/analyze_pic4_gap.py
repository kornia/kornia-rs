#!/usr/bin/env python3
"""Identify the 35 contours pic4 EXT misses vs cv2.

Strategy: find the unique start-points (lex-min of each contour) in cv2's
output that have no counterpart in kornia's output. Then read the binary
image and print 5x5 neighborhoods around each missing start to spot the
topological pattern.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def load(name):
    return json.loads(Path(f"crates/kornia-imgproc/tests/snapshots/{name}").read_text())["contours"]


def lex_min(contour):
    """Return (x_min, y_at_x_min) sorted lexicographically."""
    return min(tuple(p) for p in contour)


def main():
    k = load("kornia_pic4_external_simple.json")
    o = load("cv2_pic4_external_simple.json")
    print(f"kornia: {len(k)} contours, cv2: {len(o)} contours, gap: {len(o)-len(k)}")

    k_starts = {lex_min(c) for c in k}
    o_starts = {lex_min(c) for c in o}

    only_cv2 = sorted(o_starts - k_starts)
    only_kornia = sorted(k_starts - o_starts)
    print(f"\n  only in cv2: {len(only_cv2)} contour starts")
    print(f"  only in kornia: {len(only_kornia)} contour starts")

    img = cv2.imread("crates/kornia-imgproc/examples/data/pic4.png", cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    h, w = bw.shape
    print(f"\n  image: {w}x{h}")

    def topo_hist(starts, label):
        counts = {}
        for x, y in starts:
            sig = ""
            for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    sig += "1" if bw[yy, xx] else "0"
                else:
                    sig += "?"
            counts[sig] = counts.get(sig, 0) + 1
        print(f"\n  {label} histogram ({len(starts)} starts; sig=N NE E SE S SW W NW):")
        for sig, n in sorted(counts.items(), key=lambda t: -t[1])[:10]:
            print(f"    {sig}  ×{n}")
        return counts

    cv2_hist = topo_hist(only_cv2, "cv2-only (we miss)")
    k_hist = topo_hist(only_kornia, "kornia-only (we over-detect)")

    # Detail: show full row context for first 5 cv2-only starts.
    # This reveals what the previous-row + current-row pixels look like.
    print("\n  detailed context for first 5 cv2-only missing starts (3 rows × 11 cols):")
    for x, y in only_cv2[:5]:
        print(f"\n  ({x}, {y}):")
        for dy in range(-1, 2):
            yy = y + dy
            if 0 <= yy < h:
                row_str = ""
                for dx in range(-5, 6):
                    xx = x + dx
                    if 0 <= xx < w:
                        v = bw[yy, xx]
                        marker = "*" if (dy == 0 and dx == 0) else " "
                        row_str += marker + ("1" if v else ".")
                    else:
                        row_str += " ?"
                print(f"    y={yy:3d}: {row_str}")

    overlap = set(cv2_hist) & set(k_hist)
    print(f"\n  topology overlap: {len(overlap)} signatures appear in BOTH miss + over-detect lists")
    print(f"  (suggests near-misses where same topology is mislabeled — start point off by 1 px)")
    return  # skip the rest of the original code

    print("\n  topologies (5x5 around each missing start; 1=fg, .=bg):")
    print("  point         pattern (5x5 centered, with start *)")
    for x, y in only_cv2[:20]:
        rows = []
        for dy in range(-2, 3):
            row = ""
            for dx in range(-2, 3):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    v = bw[yy, xx]
                    if dy == 0 and dx == 0:
                        row += "*" if v else "x"
                    else:
                        row += "1" if v else "."
                else:
                    row += "?"
            rows.append(row)
        print(f"  ({x:3d},{y:3d})  | {' | '.join(rows)}")
    if len(only_cv2) > 20:
        print(f"  ... ({len(only_cv2)-20} more not shown)")

    # Compact: count topology classes by the 8-conn signature of each start.
    print("\n  topology histogram (8-conn neighborhood signature N NE E SE S SW W NW):")
    counts = {}
    for x, y in only_cv2:
        sig = ""
        for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                sig += "1" if bw[yy, xx] else "0"
            else:
                sig += "?"
        counts[sig] = counts.get(sig, 0) + 1
    for sig, n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"    {sig}  ×{n}")


if __name__ == "__main__":
    main()
