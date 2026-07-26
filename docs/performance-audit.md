# kornia-rs per-op performance audit (Jetson AGX Orin, 2026-07-19)

Whole-library audit of every python-exposed op against the per-op 10x
goal: **every GPU op ≥ 10x vs both OpenCV CPU and VPI-CUDA**, with
physics exceptions documented rather than hidden. Numbers from
`scripts/bench_audit.py` — 1080p, locked clocks (GPU pinned 1020 MHz),
owned CUDA stream, stream-synchronized rounds, min-of-rounds. Labels,
bytes, and edges are exact vs the cv2 references everywhere a parity
suite exists (no approximate fast modes anywhere in the library).

Caveats: cv2 CPU baselines drift ±80% run-to-run (kornia GPU is stable
±3%) — ratios are one-sitting comparisons. CPU rows were taken at
residual load ~2 (of 6 cores); multithreaded CPU paths are the most
load-sensitive and read conservative.

## Per-op table (ms; ratios = baseline / kornia-GPU)

| op | k-cpu | k-gpu | cv2 | VPI | ×cv2 (GPU) | ×VPI | ×cv2 (CPU) |
|---|---|---|---|---|---|---|---|
| resize-nearest | 0.285 | 0.085 | 0.670 | 0.615 | 7.9 | 7.3 | 2.4 |
| resize-bilinear | 2.058 | 0.198 | 2.126 | 0.691 | **10.8** | 3.5 | 1.0 |
| resize-bicubic | 4.928 | 0.572 | 5.568 | 0.924 | 9.7 | 1.6 | 1.1 |
| resize-lanczos | 7.658 | 0.923 | 9.394 | — | **10.2** | — | 1.2 |
| warp-affine | 7.019 | 0.842 | 10.303 | 1.422 | **12.2** | 1.7 | 1.5 |
| warp-perspective | 8.201 | 1.154 | 12.165 | 1.131 | **10.5** | 1.0 | 1.5 |
| dilate 3×3 C1 | 25.960 | 0.164 | 0.604 | 1.095 | 3.7 | 6.7 | 0.02 |
| erode 3×3 C1 | 28.645 | 0.164 | 0.589 | 1.171 | 3.6 | 7.2 | 0.02 |
| gaussian 5×5 C3 | 2.187 | 0.667 | 6.221 | — | 9.3 | — | 2.8 |
| gaussian 5×5 C1 | — | 0.480 | 2.038 | 0.694 | 4.2 | 1.5 | — |
| box 5×5 C3 | 2.256 | 0.666 | 6.222 | — | 9.3 | — | 2.8 |
| box 5×5 C1 | — | 0.480 | 2.009 | 0.677 | 4.2 | 1.4 | — |
| sobel 3×3 f32 | 62.283 | 1.326 | 2.686 | — | 2.0 | — | 0.04 |
| median 3×3 C1 | 0.302 | 0.306 | 0.247 | 0.859 | 0.8 | 2.8 | 0.8 |
| median 5×5 C1 | 1.744 | 1.183 | 1.798 | 1.416 | 1.5 | 1.2 | 1.0 |
| bilateral d5 C1 | 8.527 | 3.240 | 10.270 | 0.853 | 3.2 | 0.3 | 1.2 |
| histogram C1 | 0.987 | 0.194 | 1.715 | 0.556 | 8.9 | 2.9 | 1.7 |
| equalize-hist C1 | 1.476 | 0.325 | 0.889 | 0.859 | 2.7 | 2.7 | 0.6 |
| clahe C1 | 1.609 | 0.534 | 3.345 | — | 6.3 | — | 2.1 |
| canny | 2.529 | 1.242 | 1.751 | 2.168 | 1.4 | 1.7 | 0.7 |
| connected-components | 2.253 | 1.836 | 1.759 | — | 1.0 | — | 0.8 |
| gray_from_rgb u8 | 0.593 | 0.094 | 0.367 | 0.812 | 3.9 | 8.6 | 0.6 |
| hsv_from_rgb f32 | 2.921 | 0.528 | 1.440 | — | 2.7 | — | 0.5 |
| lab_from_rgb f32 | 12.809 | 0.538 | 16.490 | — | **30.6** | — | 1.3 |
| ycbcr_from_rgb f32 | 1.392 | 0.538 | 1.304 | — | 2.4 | — | 0.9 |

VPI has no op for: lanczos, CLAHE, connected components, sobel,
histogram-equalization pairs beyond eqhist, or any color conversion
except format converts (gray). VPI bilateral and Canny use different
algorithms than cv2/kornia (outputs not comparable — kornia matches cv2
byte-for-byte instead).

## 10x scorecard

**PASS (≥10x vs cv2, GPU)**: resize-bilinear 10.8, resize-lanczos 10.2,
warp-affine 12.2, warp-perspective 10.5, lab_from_rgb 30.6. Bicubic
(9.7), gaussian/box C3 (9.3), histogram (8.9) sit at the boundary and
cross it run-to-run.

**Envelope-capped (vs VPI)**: 10x vs VPI is DRAM-physics-infeasible for
bandwidth-bound ops — both stacks saturate the same ~80 GB/s u8 / ~55
GB/s f32-C3 measured envelopes (204 GB/s spec is SoC-shared,
unreachable by one stream). warp-perspective at 0.98–1.0x VPI is the
cleanest demonstration of the shared ceiling. kornia is
fastest-or-tied vs VPI on every comparable op except bilateral (0.3x —
different algorithm: VPI's is approximate, kornia matches cv2
byte-for-byte).

**Overhead-capped (vs cv2)**: ops whose cv2 CPU baseline is already
sub-millisecond (dilate/erode 0.6 ms, median3 0.25 ms, equalize 0.9 ms,
gray 0.37 ms) put 10x inside fixed launch+sync cost (~50-150 µs/op).
The GPU path still wins whenever the data is already device-resident;
the fused pipeline (73x vs the cv2 chain, `scripts/bench_10x.py`) is
where the 10x goal is met end-to-end by eliminating those per-op costs.

**Behind (GPU)**: median-3×3 0.8x vs cv2's exceptional NEON (GPU floor
0.31 ms vs cv2 0.25 ms); connected-components ~1.0x on this content
(beats cv2 1.0-1.8x on the dedicated content sweep in PR #1033); canny
1.4x (hysteresis is latency-bound, persistent-kernel variant measured
2.4x WORSE and documented in-source).

## CPU gaps found by this audit

1. **dilate/erode CPU: 26-29 ms** (cv2 0.6 ms, 43x behind) — the CPU
   morphology path is the generic `Ord`-based engine; it never got the
   NEON/word-trick treatment because the roadmap optimized GPU only.
   Worst gap in the library.
2. **sobel f32 CPU: 62 ms** (cv2 2.7 ms, 23x behind) — generic
   separable-filter path, no SIMD specialization.
3. equalize (0.6x), canny (0.7x), gray u8 (0.6x), hsv f32 (0.5x) CPU
   trail cv2 — measured under residual load; re-check, then optimize or
   document.

## API gaps (python)

- Pyramids (`pyrdown`/`pyrup`/`build_pyramid`) are rust-only — no
  python binding.
- `gaussian_blur`/`box_blur` host path is C3-only (C1 raises); GPU
  accepts both.
- `hsv/lab/ycbcr_from_rgb` host arms are f32-only; u8 is GPU-only —
  and the u8-host error message is a confusing pyo3 downcast error
  rather than a dtype message.
- `compute_histogram` host path takes numpy arrays but not host
  `Image`s.

## Follow-ups

- CPU morphology + sobel SIMD passes (close the two 20-40x CPU gaps).
- Consolidate the four bench scripts' duplicated harnesses into
  `scripts/bench_common.py`; unify `kornia-apriltag/src/rle_cc.rs` with
  the imgproc run-based CCL core.
- Fusion generalization (#39) — the pipeline-level answer to
  overhead-capped ops.
