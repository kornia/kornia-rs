#!/usr/bin/env python3
"""Disk-based snapshot diff: compare kornia_*.json vs cv2_*.json.

Faster than `check_correctness.py` because it doesn't re-run cv2 — it reads the
frozen cv2 baseline from `tests/snapshots/`. Use during the cv2-parity work to
iterate on `find_contours` and quickly see whether the gap is shrinking.

Workflow:
  1. one-time: `bash crates/kornia-imgproc/examples/dump_snapshots.sh`
  2. after each kornia change: regen kornia snapshots only:
       for c in pic1 pic2 pic3 pic4; do
         for m in external list; do for a in simple none; do
           ./target/release/examples/dump_contours \
             "crates/kornia-imgproc/examples/data/$c.png" "$m" "$a" \
             > "crates/kornia-imgproc/tests/snapshots/kornia_${c}_${m}_${a}.json" \
             2>/dev/null || true
         done; done
       done
  3. `python3 crates/kornia-imgproc/examples/diff_snapshots.py`
"""
import json
import sys
from pathlib import Path

SNAP = Path("crates/kornia-imgproc/tests/snapshots")

# pic2 LIST snapshots aren't generated (5722 contours, too large to be useful).
COMBOS = []
for fixture in ("pic1", "pic3", "pic4"):
    for mode in ("external", "list"):
        for method in ("simple", "none"):
            COMBOS.append((fixture, mode, method))
for method in ("simple", "none"):
    COMBOS.append(("pic2", "external", method))


def load(name):
    p = SNAP / name
    if not p.exists():
        return None
    return json.loads(p.read_text())["contours"]


def normalize(contours):
    """Sort by lex-min point so contour ordering doesn't affect equality."""
    keyed = []
    for c in contours:
        if not c:
            continue
        # lex-sort points to find leftmost-topmost
        idx = min(range(len(c)), key=lambda i: (c[i][0], c[i][1]))
        keyed.append((tuple(c[idx]), tuple(map(tuple, c))))
    keyed.sort(key=lambda t: t[0])
    return keyed


def diff(fixture, mode, method):
    k = load(f"kornia_{fixture}_{mode}_{method}.json")
    o = load(f"cv2_{fixture}_{mode}_{method}.json")
    label = f"{fixture} {mode:8s} {method:6s}"
    if k is None or o is None:
        return f"  {label}  SKIP (snapshot missing — regenerate via dump_snapshots.sh)"
    if len(k) != len(o):
        return f"  {label}  COUNT  kornia={len(k):4d} cv2={len(o):4d} delta={len(k)-len(o):+d}"
    kn, on = normalize(k), normalize(o)
    eq = sum(1 for a, b in zip(kn, on) if a[1] == b[1])
    if eq == len(kn):
        return f"  {label}  BIT-EXACT ({eq}/{len(kn)})"
    return f"  {label}  CONTENT {eq}/{len(kn)} contours match (count ok, points differ)"


def main():
    print(f"snapshot diff (vs cv2 baseline at {SNAP}):")
    print()
    bit_exact = 0
    for combo in COMBOS:
        line = diff(*combo)
        print(line)
        if "BIT-EXACT" in line:
            bit_exact += 1
    print()
    print(f"  bit-exact: {bit_exact}/{len(COMBOS)}")
    sys.exit(0 if bit_exact == len(COMBOS) else 1)


if __name__ == "__main__":
    main()
