Verdict: request changes

- PR body — blocking — Author walkthrough is generic ("Per file: see description", "covered by the unit tests"). It names no real risk and ties nothing to the diff. The human must write it in their own words. Question for the author: what happens when `w` is ≈0 for a point, and why is the singularity threshold `1e-8` on the raw determinant? — fix: write a real walkthrough — [inferred]
- PR body — blocking — The Self-review is a placeholder: `Verdict: approve` and `Skill gaps: none` only. It has no per-section results and no `Reuse searches` block. Run the skill and paste its output — [inferred]
- crates/kornia-imgproc/src/warp/points.rs:4 — blocking — `mat3_inv` duplicates `invert_homography` (perspective.rs:41), and the two disagree on singularity. This one rejects when raw `|det| < 1e-8`. The existing one normalises by `|h8|³` and uses `1e-10`. Input `H = 0.001·[1,0,2; 0,1,3; 0,0,1]` has det = 1e-9 and is valid. `invert_homography_small_scale_accepted` (perspective.rs:410) already pins it as accepted. `unwarp_points_perspective` returns `CannotComputeDeterminant` for it, while `warp_perspective` accepts the same matrix. Fix: delete `mat3_inv` and call `inverse_perspective_matrix` or `invert_homography`. The latter is `pub(crate)`, and `inverse_perspective_matrix` is `pub(super)`, so both are reachable from `warp::points`. — [measured: see Reuse searches]
- crates/kornia-imgproc/src/warp/points.rs:23 — major — `apply_h` duplicates `transform_point` (perspective.rs:67), the same projective divide. Reuse it, or make it `pub(super)` and share it. — [measured]
- crates/kornia-imgproc/src/warp/points.rs:48 — major — The length mismatch is reported as `ImageError::InvalidChannelShape(dst_len, src_len)`. Its message is "Data length ({0}) does not match the image size ({1})", which is misleading for point slices. These are not images. Use a more fitting error or add a variant. Doing so also avoids bending the `ImageError` semantics. — [inferred]
- crates/kornia-imgproc/src/warp/points.rs:44 — major — Nothing handles `w ≈ 0`. A point mapping to infinity yields inf/NaN written silently, with no documentation of it. Document the behaviour or reject it. — [inferred]
- crates/kornia-imgproc/src/warp/points.rs:62 — major — The tests only use identity and translation (`h6 = h7 = 0`, `w = 1`), so the projective divide is never exercised. Neither test would catch a wrong inverse in the perspective terms. The singular test only uses the all-zero matrix. Add a non-affine H with expected values from OpenCV `cv2.perspectiveTransform` with `cv2.getPerspectiveTransform`, saying how they were generated. Add a round trip against forward `transform_point`, and a case for the length-mismatch `Err`. — [inferred]
- crates/kornia-imgproc/src/warp/points.rs:29 — minor — The rustdoc has no `# Examples` and no reference such as OpenCV `perspectiveTransform`. It does not say that `dst_points` are the warp output and `src_points` the result, or what `m` is relative to the `src -> dst` convention (perspective.rs:78). — [inferred]
- crates/kornia-imgproc/src/warp/mod.rs:17 — minor — The PR does not say whether the function is exposed in `kornia-py` or the C++ bindings. Either update them or state that it is not exposed. — [inferred]
- PR body — minor — There is no `changelog.d/<PR>.added.md` entry (the directory holds only a README), and the PR does not say it is a no-changelog change. — [measured: `ls changelog.d` shows only `README.md`]
- PR body — nit — "Tests: cargo test … passes" has no quoted output. The tests are in `points.rs` and should be named as run, with counts. I did not run them, since cargo is off-limits. — [inferred]
- PR body — nit — The PR says "keep the module independent of the warp kernels". That is not a goal that justifies duplicating the kernels' helpers. — [inferred]

Scope: the change is one concern, about 90 lines, and maps to #9998. I found no out-of-scope hunks. `ISSUE.md` was readable.

Section 1 (copies/allocations) and section 2 (latency): nothing found. The output goes into a caller-provided slice, with no allocation and no parallelism. Section 3 (benchmarks): this is a tiny O(n) loop and not a kernel hot path, so a bench is not required. If speed is claimed, `bench_warp.rs` exists to extend.

Reuse searches:
- `mat3_inv` (invert a 3x3 homography)
  `$ grep -rnE "fn \w*(invert|inv|inverse)\w*" crates/`
  - `crates/kornia-imgproc/src/warp/perspective.rs:41: pub(crate) fn invert_homography(m: &[f32; 9]) -> Option<[f32; 9]>`
  - `crates/kornia-imgproc/src/warp/perspective.rs:62: pub(super) fn inverse_perspective_matrix(m: &[f32; 9]) -> Result<[f32; 9], ImageError>`
  - `crates/kornia-imgproc/src/warp/perspective.rs:376-410: invert_homography_known, invert_homography_round_trip, invert_homography_singular_returns_none, invert_homography_small_scale_accepted`
  - `crates/kornia-algebra/src/mat.rs:67: pub fn inverse(self) -> Self`
  - The existing helper has a scale-invariant check (`det / |h8|³`, `1e-10`), while the new copy uses raw `|det| < 1e-8`.
- `apply_h` (apply a homography to a point)
  `$ grep -rnE "fn \w*(transform_point|apply_h|perspective_transform|project_point)\w*" crates/`
  - `crates/kornia-imgproc/src/warp/perspective.rs:67: fn transform_point(x: f32, y: f32, m: &[f32; 9]) -> (f32, f32)`
  - `crates/kornia-3d/src/linalg.rs:23: pub fn transform_points3d(`
  - The `kornia-3d` hit works on 3D points and is not a fit. The perspective.rs hit is the same math.
- Error mapping
  `$ grep -rnE "CannotComputeDeterminant" crates/`
  - `crates/kornia-imgproc/src/warp/perspective.rs:63` already maps `invert_homography(m).ok_or(ImageError::CannotComputeDeterminant)`.
  - `$ grep -n InvalidChannelShape -B2 crates/kornia-image/src/error.rs` shows the message "Data length ({0}) does not match the image size ({1})".

Skill gaps:
- The self-review output was a placeholder, so the author's self-review missed the blocking duplicate (`mat3_inv`) and the disagreeing singularity threshold. Rule for `kornia-coding-style` or `kornia-code-review` 0b: a new private helper with the same name-agnostic job as an existing `pub(crate)` or `pub(super)` helper must delegate to it. Compare thresholds on a scaled-down matrix, such as `H·0.001`.
- Section 4 could be sharper: a test of a projective transform must include a non-affine H (`h6` or `h7` ≠ 0), or the divide is never exercised.
- Reusing an `ImageError` variant for non-image data (point slices) has no rule. Propose one in `kornia-api-flavor`: don't reuse an error variant whose message names an image for non-image arguments.
- "Tests pass" with no output is accepted by the PR template. Propose requiring the quoted test names and counts.
