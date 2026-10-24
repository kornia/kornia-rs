A passing review reports all of:

1. major or blocking, §0b — the hand-written `jacobi_symmetric` eigen-solver re-implements an existing decomposition. Preferred fix (`kornia-arch`, linear algebra): build the covariance as `Mat3F64` and call `kornia_algebra::linalg::svd::svd3_f64` (used in `kornia-3d/src/pose/planar.rs`, `fundamental.rs`, `homography.rs`); the normal is the singular vector of the smallest singular value. Pointing at nalgebra `symmetric_eigen()` (already used in kornia-3d) also passes.
2. minor or major, §0b — `dot3`, `normalize3`, `centroid` and the `[f64; 3]` types duplicate `kornia_3d::linalg::dot_product3` and `Vec3F64` operations; use kornia-algebra types.
3. §0b — a `Reuse searches` block with pasted searches that find an existing eigen/SVD routine and the vector helpers; finding 1 is tagged measured.

Items 1 and 3 are the pass bar. Bonus: the test uses an axis-aligned square whose covariance is already diagonal, so the Jacobi loop never rotates (the test does not reach the algorithm); degenerate input (collinear, coincident, non-finite) returns `Ok`.

2026-10-22 (Opus 5.5 and Sonnet 5.5, one run each per skill version): items 1-2 found in every run, with or without section 0b; reviewers split between `svd3_f64` and nalgebra before `kornia-arch` named a backend. Item 3: 0/2 without 0b, 2/2 with the `Reuse searches` block.

2026-10-24 (skill at the kornia-workflow step 6 / kornia-curation rule 3 edit): Opus 5.5 3/3 + bonus (no-op axis-aligned test, degenerate input) and Sonnet 5.5 3/3 + bonus; both reached `svd3_f64` and measured the degenerate outputs with a Python transcription.
