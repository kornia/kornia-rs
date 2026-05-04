# find_contours 10× plan — what the literature says works

This document captures the research from **published 2009-2024 literature** on
state-of-the-art Connected Component Labeling (CCL) + boundary tracing
algorithms, and proposes a concrete implementation path for kornia-rs to
beat OpenCV by **5-10× on real-world images**.

## Why our session-scope optimizations capped at ~25%

Suzuki/Abe (1985) is a per-pixel scan algorithm. Its inner loop reads each
pixel + its left/right neighbors and branches. Even with SWAR (current kornia
impl runs at ~0.23 cycles/pixel, near memory ceiling on Jetson), it's
fundamentally bound by the number of pixels processed.

NEON layering on top regresses perf (commit `58337f8`) because the per-call
overhead of NEON intrinsics exceeds the work being saved on ARM.
**The bottleneck is not vectorization; it's the per-pixel iteration count.**

## The literature's answer: switch to a run-based algorithm

All modern CCL implementations process **runs** (consecutive same-value pixel
sequences in a row), not individual pixels. For real images this gives a
massive iteration-count reduction:

| image                | pixels | runs/row | total iters | reduction |
|----------------------|-------:|---------:|------------:|----------:|
| filled_square 1024²  |     1M |      2-3 |        3000 | **333×**  |
| pic1 (400×300)       |   120k |    10-20 |        5000 | **24×**   |
| pic4 (400×300)       |   120k |   50-100 |       25000 | **5×**    |
| sparse_noise 256²    |    65k |     ~120 |       30000 | 2×        |

Even the worst case (sparse noise) gets 2× iteration reduction. Realistic
images get 5-300× fewer iterations.

## Recommended algorithm: LSL or DRAG (NOT Spaghetti)

Per Lemaitre/Lacassagne (HAL 2020), **Spaghetti was Intel-tuned and
underperforms on ARM**. On ARM the winners are:

1. **LSL (Light Speed Labeling)** — Lacassagne 2009 + parallelized 2018 +
   SIMD U1 variant 2020. Run-based, RISC-friendly, 3-pass.
2. **DRAG** — block-based but with ARM-friendly memory access patterns.
3. **Chang/Chen (2004)** — unified CCL + contour tracing in one pass
   (useful since findContours needs both).

## Implementation plan (5-day estimate for one engineer)

### Day 1: Run-length encoding pass
- Add a `RunLengthRow { starts: Vec<u32>, ends: Vec<u32> }` struct
- `binarize_to_runs(src: &[u8], rows: &mut Vec<RunLengthRow>)` — single
  pass over the image, NEON-vectorized to find run boundaries via
  `vceqq_u8 + vmvnq_u8 + bit-extract`
- The i16 padded image is no longer needed
- **Validate**: round-trip RLE → reconstruct binary, byte-exact

### Day 2: Connected component labeling via LSL
- For each row, compute "line-relative labels" (sequential per-row IDs)
- Merge equivalences across rows using union-find with run intervals
- Output: array of component labels per run
- **Validate**: cv2.connectedComponents byte-exact match

### Day 3: Boundary tracing per component
- For each labeled component, find leftmost run, trace boundary using
  Chang/Chen approach
- This pass is O(total perimeter) — should match Suzuki/Abe perimeter cost
  but no longer pays the per-pixel scan cost
- **Validate**: bit-exact contour points vs cv2.findContours

### Day 4: NEON optimization pass
- NEON-vectorize the run-extraction (Day 1) — process 16 bytes at a time
- NEON-vectorize the row-merge (Day 2) — process pairs of overlapping runs
  using vector compare + select
- Bench against the OpenCV baseline already in this branch

### Day 5: Validation, polish, PR
- Run `check_correctness.py` end-to-end (must match cv2 byte-exact for all
  External-mode fixtures; List-mode hole-detection rule alignment is bonus)
- Run full `bench_contours_min` + `bench_opencv_contours.py`
- Update `RESULTS_CONTOURS.md` with 5-10× numbers
- Open PR

## Expected results (per the LSL/Spaghetti benchmark literature)

| fixture              | current | OpenCV | LSL-based predicted | vs OpenCV |
|----------------------|--------:|-------:|--------------------:|----------:|
| pic1                 | 108 μs  |  86 μs |             ~25 μs  | **3.4× FASTER** |
| pic4                 | 944 μs  | 2070 μs|            ~150 μs  | **13.8× FASTER** |
| filled_square 1024²  | 506 μs  | 389 μs |             ~50 μs  | **7.8× FASTER** |
| sparse_noise 256²    | 3939 μs | 762 μs |           ~1500 μs  | parity (sparse always hard) |

Predictions based on: 5-10× iteration reduction × 0.5 (overhead) + LSL paper's
reported 1.7-1.9× speedup over Spaghetti on x86 (which is itself ~3× faster
than OpenCV's findContours).

## Why this is the right next work item

- **Self-contained**: doesn't touch other kornia modules
- **Well-defined success criteria**: bit-exact match + measurable speedup
- **Standing on published shoulders**: LSL paper has full pseudocode
- **Reproducible bench harness already exists** (this prototype)
- **Real impact**: makes kornia-rs the fastest find_contours on Jetson

## References

- Suzuki & Abe (1985) — original boundary tracing algorithm (current impl)
- Chang, Chen, Lu (2004) — "A linear-time component-labeling algorithm using contour tracing technique"
- Lacassagne & Zavidovique (2009) — LSL original paper:
  https://lip6.fr/Lionel.Lacassagne/Publications/ICIP09_LSL.pdf
- Cabaret & Lacassagne (2018) — "Parallel Light Speed Labeling":
  https://largo.lip6.fr/~lacas/Publications/JRTIP18_LSL.pdf
- Bolelli et al. (2019) — Spaghetti algorithm (state-of-the-art on x86)
- Lemaitre, Lacassagne (2020) — "How to speed Connected Component Labeling up with SIMD RLE algorithms"
- YACCLAB — https://github.com/prittt/YACCLAB (benchmark suite, integrates with our bench harness)
