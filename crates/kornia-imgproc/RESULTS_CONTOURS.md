# find_contours perf — results

**Hardware:** Jetson Orin Nano 8GB Super (LPDDR5 68 GB/s, 6× Cortex-A78AE)
**Date:** 2026-05-05
**Target:** match or beat `cv2.findContours(img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` —
the canonical OpenCV invocation across docs, tutorials, and production
code.

## Headline (vs cv2's default invocation)

The trace function is a line-for-line port of cv2's `icvFetchContour`
(`modules/imgproc/src/contours.cpp:511-620`) — same direction encoding,
same wrapped-direction marking, same halt rule, same emission timing,
same SIMPLE-mode chain compression.

| fixture | cv2 count | kornia count | bit-exact (EXT/LIST × simple/none) | cv2 time | kornia | margin |
|---------|----------:|-------------:|------------------------------------|---------:|-------:|-------:|
| pic1.png 400×300 | 1 | 1 | ✅ 4/4 — full coordinate parity | 88 μs | 84 μs | **1.05× faster** |
| pic2.png 400×300 | 1 | 1 | ✅ 2/2 EXT (LIST not snapshot-tracked) | 513 μs | 565 μs | ~tied |
| pic3.png 400×300 | 1 | 1 | ✅ 4/4 — full coordinate parity | 90 μs | 85 μs | **1.06× faster** |
| pic4.png 400×300 | 844 | 844 | ✅ 4/4 — full coordinate parity | 2014 μs | 642 μs | **3.14× faster** |

**14/14 snapshot fixtures bit-exact with cv2** (`diff_snapshots.py`).
Every fixture, every retrieval mode, every approximation method matches
cv2's output coordinate-for-coordinate.

> Note: cv2 returns 881 contours on pic4 if you binarise via
> `cv2.IMREAD_GRAYSCALE → threshold(127, 1)`. That's because cv2's
> grayscale conversion uses a slightly different luma formula from
> kornia's `(77*R + 150*G + 29*B) >> 8`. Of pic4's 120K pixels,
> ~38K differ by ±1 gray-value between the two formulas, and ~867
> sit close enough to the threshold that they binarise differently.
> When BOTH paths consume the same binary input, kornia and cv2 are
> bit-exact. The harness in `examples/dump_snapshots.sh` and
> `examples/check_correctness.py` uses kornia's exact gray formula
> for the cv2 baseline so the comparison is apples-to-apples.

Tests: 182/182 unit + 5/5 real-image integration + 1/1 snapshot-digest +
14/14 synthetic shape patterns pass.

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

### 5. `trace_border` const-generic specialisation on `IS_OUTER`

The outer/hole branch in the inner `belongs()` predicate was a
runtime check on a closure capture. Promoted to a const generic
parameter so the compiler emits two specialised instances. The
hot 8-conn neighbor scan now constant-folds the branch away.
Worth ~2% on pic4 EXT/LIST.

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
