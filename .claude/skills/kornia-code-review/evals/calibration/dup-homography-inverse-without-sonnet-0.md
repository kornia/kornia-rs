**Review of PR: feat(imgproc): add `unwarp_points_perspective`**

**Verdict: request changes.** The math is right, but the PR duplicates existing code and has a real numerical bug.

**Must fix**

1. **It duplicates existing inversion code, and the copy is worse.** `warp/perspective.rs` already has `invert_homography` (line 41), which returns `Option`. It also has `inverse_perspective_matrix`, which returns `Result<_, ImageError::CannotComputeDeterminant>`, along with `determinant3x3`, `adjugate3x3` and `transform_point`.
   - `mat3_inv` and `apply_h` in `points.rs` re-implement all of these.
   - The PR says the helpers "keep the module independent of the warp kernels". That isn't a good reason to fork a matrix inverse.
   - `inverse_perspective_matrix` is `pub(super)`, so `points.rs` can call it directly. That removes about 40 lines and gives the same error type.
   - `transform_point` is private to `perspective.rs`. Either widen it to `pub(super)` or put the new function in `perspective.rs`.

2. **The singularity check is not scale-invariant.** `mat3_inv` uses `det.abs() < 1e-8` on the raw determinant. The existing `invert_homography` normalises by `|h[8]|³` for this exact reason, and its doc comment explains why: `det(kH) = k³·det(H)`.
   - A valid homography scaled by 0.001 has `det ≈ 1e-9` and would be rejected as singular.
   - A near-singular matrix at a large scale would be accepted.
   - Reusing the existing function fixes this. Homographies are only defined up to scale, so this matters.

3. **Points at infinity are not handled.** `apply_h` divides by `w` with no check. If the inverse maps a point onto the line at infinity (`w ≈ 0`), the output is `inf` or `NaN`, and the function still returns `Ok(())`. Keypoints near the horizon of a strong perspective are realistic inputs.
   - Return an error for such points, or document that non-finite values can occur.
   - At minimum, add a test for it.

4. **The wrong error variant is used for a length mismatch.** `InvalidChannelShape(usize, usize)` prints "Data length (…) does not match the image size (…)". That message is misleading for two point slices. Check whether a more appropriate variant exists, or add one. Alternatively, make the API infallible on length by writing into `dst_points.len().min(..)` and documenting that, or by using a `&mut [[f32; 2]]` of the same length via an iterator over a fixed-size array.

**Tests are too weak**

- The PR says "numerical edge cases, covered by the unit tests", but the only cases are identity, a pure translation, and an all-zero matrix.
- None of them has a non-trivial projective component (`h[6]` or `h[7]` ≠ 0), which is the only place the `/w` division and the inverse's third row matter.
- Add a round-trip test: apply the forward `H` to points, call `unwarp_points_perspective`, and compare with the originals. The existing `invert_homography_round_trip` matrix is a good choice.
- Add a test that the result agrees with what `warp_perspective` actually samples, to pin down the `src → dst` convention.
- Add tests for a length mismatch (`unwrap_err` on the variant), an empty slice, a scaled-matrix case (see item 2), and `w ≈ 0`.
- `unwarp_singular` only asserts `is_err()`. Assert the variant with `matches!`.
- Identity and translation use `assert_eq!` on floats in one test and tolerances in the other. That's fine here, but keep it consistent.

**Process and style**

- **No changelog fragment.** `CONTRIBUTING.md:228` requires `changelog.d/<PR>.added.md` for user-visible changes. This is a new public API.
- **No doc example.** Sibling public functions such as `warp_perspective` have a runnable `# Example` doctest. Add one.
- **Missing license header or module doc.** Check whether the sibling files have a header and match it.
- **The PR description's "Self-review: approve / Skill gaps: none" is not credible.** The review missed the duplicate helpers, which are in the same directory, and the test gaps. The "Riskiest part" line should name the real risks: the singularity threshold, `w ≈ 0`, and the convention.
- The PR claims `cargo test -p kornia-imgproc warp passes`. I could not run it here.

**Minor**

- The `#[rustfmt::skip]` on `mat3_inv` goes away once the function is removed.
- Consider also offering an in-place variant, or a `Vec`-returning convenience, to match the repo's API style. The no-allocation signature is fine as the primary API.