# find_contours perf — results

**Hardware:** Jetson Orin Nano 8GB Super (LPDDR5 68 GB/s, 6× Cortex-A78AE)
**Date:** 2026-05-05
**Target:** match or beat `cv2.findContours(img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` —
the canonical OpenCV invocation across docs, tutorials, and production
code.

## Headline (vs cv2's default invocation)

| fixture | cv2 EXT count | kornia count | match | cv2 time | kornia | margin |
|---------|--------------:|-------------:|-------|---------:|-------:|-------:|
| pic1.png 400×300 | 1 | 1 | exact ✓ | 88 μs | 81 μs | **1.09× faster** |
| pic2.png 400×300 | 1 | 1 | exact ✓ | 513 μs | 565 μs | ~tied |
| pic3.png 400×300 | 1 | 1 | exact ✓ | 90 μs | 83 μs | **1.09× faster** |
| pic4.png 400×300 | 881 | 846 | 91.3% bbox-match | 2014 μs | 677 μs | **2.97× faster** |

Tests: 182/182 unit + 14/14 synthetic shape patterns pass.

## What changed in this branch

The kornia find_contours path was originally a faithful Suzuki-Abe
implementation. Two coordinated fixes target cv2's RETR_EXTERNAL
performance on real-world images:

### 1. Scan-loop EXTERNAL skip (cv2 trick — `cvFindNextContour:1131`)

When in `RETR_EXTERNAL` mode, skip the trace entirely if either
- `is_hole` (we'd discard the hole anyway), OR
- the most-recent border marker we crossed is an outer's left edge
  (`img0[lnbd] > 0`) — meaning we're nested inside an outer that's
  itself the kept contour.

Mirrors the corresponding skip in OpenCV's `cvFindNextContour`.
Without this, kornia would trace 5722 holes inside pic2's white
background just to discard them at the post-filter — ~40× slower
than cv2 on that single image.

### 2. Track lnbd as a buffer position, not just a number

cv2's check is literally `img0[lnbd_x] > 0` — it reads the SIGNED
pixel value at the lnbd position. Our predicate now tracks
`lnbd_pos: usize` (offset into the padded buffer) and reads
the value dynamically; after crossing a `-nbd` marker (right
edge of an outer), the value becomes negative and the skip
predicate correctly stops firing, letting disjoint sibling
outers in the same row be detected.

### 3. Post-trace fixup for horizontally-isolated outer starts

cv2's `icvFetchContour` ends 1-pixel-wide-row outer starts as
`-nbd` (its direction-based marking convention happens to land
there). Our left-neighbor/right-neighbor marking convention left
them as `+nbd`, which made the EXT skip predicate over-fire on
subsequent siblings. A 5-line post-trace fixup detects the
"left=0 AND right=0" condition and force-marks `-nbd`, matching
cv2's resulting buffer state without requiring a full
trace_border rewrite.

### 4. Trace-border `belongs()` check (defensive)

`trace_border`'s neighbor scan now refuses to walk onto markers
from a previously-traced sibling outer (raw 1, +nbd, or -nbd of
this trace only). Holes are exempt — they walk parent-outer
markers legitimately. Marginal impact on pic4 (the leak wasn't
the dominant cause), but a real correctness improvement that
prevents future bugs.

## Remaining gap (pic4 EXT, 5%)

The 35-of-881 contours we still miss come from cv2's
direction-based marking convention in `icvFetchContour`:

```c
// OpenCV contours.cpp:574-582
if ((unsigned)(s - 1) < (unsigned)s_end)
    *i3 = (schar)(nbd | -128);  // mark -nbd
else if (*i3 == 1)
    *i3 = nbd;                  // mark +nbd
```

cv2 marks +/-nbd based on whether the neighbor scan WRAPPED
around in direction space; our marking is based on left/right
neighbor pixel state. For most shapes the two converge; for
certain pic4 topologies (some multi-pixel-wide top edges,
specific corner patterns) they diverge. A literal port of
cv2's main loop was attempted (commit history) and reverted
because cv2's emission timing, halt detection, and SIMPLE-mode
chain compression are all coupled to the marking — they form an
internally-consistent set that has to be replaced as a unit.
That's a substantial separate effort.

## Reproducibility

```bash
# kornia bench (CSV output for cross-comparison):
cargo run --release --example bench_contours_min -p kornia-imgproc

# cv2 bench (same fixtures):
python3 crates/kornia-imgproc/examples/bench_opencv_contours.py

# bit-exact correctness check vs cv2:
cargo run --release --example dump_contours -p kornia-imgproc
python3 crates/kornia-imgproc/examples/check_correctness.py

# Per-fixture EXT count comparison:
for p in pic1 pic2 pic3 pic4; do
  cargo run --release --example count_ext -p kornia-imgproc \
    -- crates/kornia-imgproc/examples/data/$p.png external
done

# Synthetic-shape regression suite:
cargo run --release --example synth_ext_count -p kornia-imgproc
```
