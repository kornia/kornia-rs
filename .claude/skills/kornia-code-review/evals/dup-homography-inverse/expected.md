A passing review reports all of:

1. blocking or major, §0b — `warp/points.rs` `mat3_inv` duplicates `invert_homography` / `inverse_perspective_matrix` in `warp/perspective.rs` (same crate, `pub(crate)` / `pub(super)`). Fix: call the existing routine.
2. blocking, §0b.2 — the copies disagree: `mat3_inv` uses a raw `det.abs() < 1e-8`, the original is scale-invariant (`det / |h[8]|³`), so an unnormalised homography that `warp_perspective` accepts (e.g. 0.001·H, det ≈ 1e-9) is rejected here. A concrete input is given, or the existing test `invert_homography_small_scale_accepted` is cited.
3. major or minor, §0b — `apply_h` duplicates `transform_point` in `perspective.rs`.
4. §0b — a `Reuse searches` block with pasted searches that find `invert_homography` and `transform_point`; findings 1 and 3 are tagged measured.

Items 1 and 4 are the pass bar; items 1-4 are full credit.

2026-10-22 (Opus 5.5 and Sonnet 5.5, one run each per skill version): items 1-3 found in every run, with or without section 0b. Item 4: 0/2 without 0b, 2/2 with the `Reuse searches` block.

2026-10-24 (same skill edit): Opus 5.5 4/4, Sonnet 5.5 4/4; both gave `1e-3·I` as the disagreeing input. Both put items 1 and 2 on one line.
