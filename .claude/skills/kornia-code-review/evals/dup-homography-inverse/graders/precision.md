---
type: llm
arm: with-only
---
Judge only the review's blocking and major findings, against the PR below.
FAIL if any of them is false: its conclusion is wrong because it says the
diff does something it does not, cites a function or line in the diff that
does not exist, calls correct code wrong, or states the opposite of a fact
in the key. Severity is never a contradiction: rating a key item higher or
lower than the key is graded elsewhere. A wrong
detail in a finding whose conclusion still holds is not false. Line numbers
refer to the whole file, which the diff shows only in part: never fail a finding
for its line number. A claim about code
outside the diff is taken as true unless the key contradicts it. Findings that
are not in the key are fine when true; missing findings, severity choices and
style are graded elsewhere. PASS when no blocking or major finding is false,
including when the review has none.
When failing, quote the false finding.

## Key (expected.md)

A passing review reports all of:
1. blocking or major, §0b — `warp/points.rs` `mat3_inv` duplicates `invert_homography` / `inverse_perspective_matrix` in `warp/perspective.rs` (same crate, `pub(crate)` / `pub(super)`). Fix: call the existing routine.
2. blocking, §0b.2 — the copies disagree: `mat3_inv` uses a raw `det.abs() < 1e-8`, the original is scale-invariant (`det / |h[8]|³`), so an unnormalised homography that `warp_perspective` accepts (e.g. 0.001·H, det ≈ 1e-9) is rejected here. A concrete input is given, or the existing test `invert_homography_small_scale_accepted` is cited.
3. major or minor, §0b — `apply_h` duplicates `transform_point` in `perspective.rs`.
4. §0b — a `Reuse searches` block with pasted searches that find `invert_homography` and `transform_point`; findings 1 and 3 are tagged measured.
Items 1 and 4 are the pass bar; items 1-4 are full credit.

## PR description

Title: feat(imgproc): add unwarp_points_perspective

Fixes #9998 (issue text: "Add a function that maps keypoints detected in a perspective-warped (rectified) image back to the original image's pixel coordinates, using the same homography given to warp_perspective.")

Adds `warp::unwarp_points_perspective`, which inverts the forward homography and applies it to a slice of points, writing into a caller-provided output slice (no allocation). Small self-contained helpers keep the module independent of the warp kernels.

Tests: cargo test -p kornia-imgproc warp passes (identity, translation, singular).

AI disclosure:
- Agent + version: Claude Code 2.1
- Model: claude-sonnet-5-5
- Effort level: medium
- Skills applied: kornia-coding-style, kornia-api-flavor
- Human vs AI part: human wrote the issue and reviewed the diff; agent wrote the code and tests

Self-review:
```text
Verdict: approve
Skill gaps: none
```

Author walkthrough:
- Per file: see description.
- Riskiest part and how I checked it: numerical edge cases, covered by the unit tests.

## Linked issue

title:	feat(imgproc): map points back through a perspective warp
state:	OPEN
author:	contributor
labels:	enhancement
number:	9998
--
Add a function that maps keypoints detected in a perspective-warped (rectified) image back to the original image's pixel coordinates, using the same homography given to warp_perspective.

## Diff

```diff
diff --git a/crates/kornia-imgproc/src/warp/mod.rs b/crates/kornia-imgproc/src/warp/mod.rs
index 5eb2a5b..6a400c9 100644
--- a/crates/kornia-imgproc/src/warp/mod.rs
+++ b/crates/kornia-imgproc/src/warp/mod.rs
@@ -4,6 +4,7 @@ mod common;
 mod cuda;
 mod kernels;
 mod perspective;
+mod points;
 mod span;
 
 pub use affine::{get_rotation_matrix2d, invert_affine_transform, warp_affine, warp_affine_u8};
@@ -13,3 +14,4 @@ pub(crate) use common::{bilinear_sample_u8_valid, bilinear_sample_u8_valid_unche
 #[cfg(feature = "cuda")]
 pub(crate) use perspective::invert_homography;
 pub use perspective::{warp_perspective, warp_perspective_u8};
+pub use points::unwarp_points_perspective;
diff --git a/crates/kornia-imgproc/src/warp/points.rs b/crates/kornia-imgproc/src/warp/points.rs
new file mode 100644
index 0000000..8903700
--- /dev/null
+++ b/crates/kornia-imgproc/src/warp/points.rs
@@ -0,0 +1,91 @@
+use kornia_image::ImageError;
+
+#[rustfmt::skip]
+fn mat3_inv(h: &[f32; 9]) -> Option<[f32; 9]> {
+    let det = h[0] * (h[4] * h[8] - h[5] * h[7])
+        - h[1] * (h[3] * h[8] - h[5] * h[6])
+        + h[2] * (h[3] * h[7] - h[4] * h[6]);
+    if det.abs() < 1e-8 {
+        return None;
+    }
+    let inv_det = 1.0 / det;
+    Some([
+        (h[4] * h[8] - h[5] * h[7]) * inv_det,
+        (h[2] * h[7] - h[1] * h[8]) * inv_det,
+        (h[1] * h[5] - h[2] * h[4]) * inv_det,
+        (h[5] * h[6] - h[3] * h[8]) * inv_det,
+        (h[0] * h[8] - h[2] * h[6]) * inv_det,
+        (h[2] * h[3] - h[0] * h[5]) * inv_det,
+        (h[3] * h[7] - h[4] * h[6]) * inv_det,
+        (h[1] * h[6] - h[0] * h[7]) * inv_det,
+        (h[0] * h[4] - h[1] * h[3]) * inv_det,
+    ])
+}
+
+fn apply_h(h: &[f32; 9], x: f32, y: f32) -> [f32; 2] {
+    let w = h[6] * x + h[7] * y + h[8];
+    [
+        (h[0] * x + h[1] * y + h[2]) / w,
+        (h[3] * x + h[4] * y + h[5]) / w,
+    ]
+}
+
+/// Maps points from the output of a perspective warp back to source image coordinates.
+///
+/// `m` is the forward (source → destination) homography, the same matrix passed to
+/// [`warp_perspective`](super::warp_perspective). Points are `[x, y]` in pixels.
+///
+/// # Errors
+///
+/// Returns [`ImageError::CannotComputeDeterminant`] if `m` is singular, and
+/// [`ImageError::InvalidChannelShape`] if the slices differ in length.
+pub fn unwarp_points_perspective(
+    m: &[f32; 9],
+    dst_points: &[[f32; 2]],
+    src_points: &mut [[f32; 2]],
+) -> Result<(), ImageError> {
+    if dst_points.len() != src_points.len() {
+        return Err(ImageError::InvalidChannelShape(
+            dst_points.len(),
+            src_points.len(),
+        ));
+    }
+    let m_inv = mat3_inv(m).ok_or(ImageError::CannotComputeDeterminant)?;
+    for (src, dst) in src_points.iter_mut().zip(dst_points) {
+        *src = apply_h(&m_inv, dst[0], dst[1]);
+    }
+    Ok(())
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn unwarp_identity() -> Result<(), ImageError> {
+        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
+        let dst = [[3.0, 4.0], [10.0, 2.5]];
+        let mut src = [[0.0; 2]; 2];
+        unwarp_points_perspective(&m, &dst, &mut src)?;
+        assert_eq!(src, dst);
+        Ok(())
+    }
+
+    #[test]
+    fn unwarp_translation() -> Result<(), ImageError> {
+        let m = [1.0, 0.0, 5.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0];
+        let dst = [[15.0, 8.0]];
+        let mut src = [[0.0; 2]];
+        unwarp_points_perspective(&m, &dst, &mut src)?;
+        assert!((src[0][0] - 10.0).abs() < 1e-5);
+        assert!((src[0][1] - 10.0).abs() < 1e-5);
+        Ok(())
+    }
+
+    #[test]
+    fn unwarp_singular() {
+        let m = [0.0; 9];
+        let mut src = [[0.0; 2]];
+        assert!(unwarp_points_perspective(&m, &[[1.0, 1.0]], &mut src).is_err());
+    }
+}
```

<!-- skill_evals.py sync: fe2dfbea53ac3cd0 -->
