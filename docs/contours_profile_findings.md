# find_contours profiling findings — what scan_other actually does

Granular per-branch counters (build with `--features profile_contours`) reveal
exactly where the dispatch loop spends its time per fixture.

## Methodology

Counters added to `execute_scan`:
- `iters` — total scalar dispatch-loop iterations (one per pixel processed)
- `zero` — entered the `pixel == 0` path (zero-skip via SWAR + NEON)
- `one` — entered the `pixel == 1` path (all-1 SWAR + NEON whole-chunk skip)
- `labeled` — entered the labeled-pixel branch (pixel ≠ 0, 1) — updates lnbd
- `starts` — actual contour starts (outer or hole)

Run: `./target/release/examples/bench_contours_min 2>&1 | grep ^PROFILE`

## Per-fixture breakdown (warm runs, Jetson Orin Nano)

### pic1 (400×300, External, 20 contours found)
```
iters=4954  zero=698 (14%)  one=2265 (46%)  labeled=1971 (40%)  starts=20 (0.4%)
scan_other ≈ 65 μs   trace_border ≈ 31 μs   bin ≈ 14 μs
TOTAL ~110 μs   (OpenCV: 86 μs)
```

### pic4 (400×300, List, 1081 contours)
```
iters=34089  zero=7493 (22%)  one=11680 (34%)  labeled=13835 (41%)  starts=1081 (3%)
scan_other ≈ 380 μs   trace_border ≈ 480 μs   bin ≈ 14 μs
TOTAL ~880 μs   (OpenCV: 2070 μs — we win 2.7×)
```

### filled_square 1024² (1 contour, big rectangle)
```
iters=4864  zero=1792 (37%)  one=1536 (32%)  labeled=1535 (32%)  starts=1
scan_other ≈ 213 μs   trace_border ≈ 25 μs   bin ≈ 153 μs
TOTAL ~390 μs   (OpenCV: 389 μs — parity)
```

## Headline finding

**Across every fixture, the labeled-pixel branch is 32-41% of dispatch
iterations.** These are pixels along contour perimeters that have already
been marked by `trace_border` and now just need an `lnbd` (last-newly-
detected-border) state update.

Per-iteration cost is small (~10-15 ns), so the absolute time spent here
is moderate (e.g., ~25 μs for pic1's 1971 labeled hits). But it's the
single largest contribution to scan_other after the existing SWAR
optimizations.

## Why NEON-skip of labeled runs DOESN'T work

We tried 4 different NEON optimization approaches in this branch. All
regressed performance. Documented in commits + this paragraph as a
warning to future contributors:

1. **NEON 8-pixel pre-check** (commit `58337f8`) — checks "any contour
   start in next 8?" before falling to scalar. Cost: 3 loads + 2 compares
   + horizontal max per chunk. Per-call NEON overhead exceeded savings.
2. **Bitmap precompute** (uncommitted, reverted) — produces bitmap of
   start candidates during binarize. The bitmap construction itself
   costs as much as the dispatch loop it's trying to skip.
3. **Labeled-skip with abs-max** (uncommitted, reverted) — checks
   "are 8 lanes all labeled?" and updates lnbd to max(abs). 5-14% slower.
4. **Labeled-skip minimal** (uncommitted, reverted) — same as #3 but
   reads only last lane for lnbd (no max). Also 5-14% slower.

**Why all 4 failed**: the existing scalar SWAR is at the speed-of-memory
ceiling (~0.23 cycles/pixel on filled_square 1024²). Per-chunk NEON
overhead (3-15 cycles depending on the operation) only wins if it
replaces 16+ cycles of scalar work — which doesn't happen because
scalar SWAR is already fast.

The labeled-pixel runs in real images are SHORT (1-3 pixels along
contour borders). NEON's 8-lane chunks rarely contain 8 consecutive
labeled pixels, so the "all 8 labeled" predicate rarely fires. When it
fires, the savings are small. When it doesn't, we paid pure overhead.

## Where the 10× actually lives (revisited)

Per the literature (LSL paper, Lacassagne 2009; SIMD RLE paper,
Lemaitre/Lacassagne 2020), the scan-loop bottleneck is fundamentally
addressed by switching from per-pixel processing to **run-based
processing**. For pic4-style images with 41% labeled pixels in
contour-perimeter runs, a run-encoded representation processes
EACH RUN (the entire contour border) in O(1) iterations instead of
O(perimeter) iterations.

Predicted speedup: 5-10× on the dispatch phase.

Full plan: `docs/contours_10x_plan.md` — 5-day project, ARM-tuned
LSL implementation with bit-exact output validation against this
branch's correctness checker.

## Bottom line for the GSoC student

The bench harness + correctness validator + per-branch profiler in this
branch make optimization claims *automatically verifiable*. The 4 negative
NEON results documented above mean future contributors can SKIP per-pixel
NEON layering attempts and go straight to the LSL algorithmic restructuring
described in the 10× plan.
