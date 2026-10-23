A passing review reports all of:

1. major or blocking, §0b — the inline serial 256-bin histogram loop re-implements `histogram::compute_histogram` (row-parallel, CUDA-dispatching) in the same crate. Fix: call it.
2. major, §0b.4 — no reference is named for the algorithm (Otsu 1979, OpenCV `THRESH_OTSU` or scikit-image `threshold_otsu`), and the only test is a range check (`t >= 12 && t < 200`). Fix: cite the reference in the rustdoc and pin a threshold generated with OpenCV or scikit-image on a stated input.
3. minor or major, §3 — a new op with no bench in `crates/kornia-imgproc/benches/` (there is no `bench_threshold`).
4. §0b — a `Reuse searches` block with a pasted search that finds `compute_histogram`; finding 1 is tagged measured.

Items 1, 2 and 4 are the pass bar.

2026-10-22 (Opus 5.5 and Sonnet 5.5, one run each per skill version): items 1-3 found in every run, with or without section 0b. Item 4: 0/2 without 0b, 2/2 with the `Reuse searches` block.
