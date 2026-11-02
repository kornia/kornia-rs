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
1. blocking, §0 — `crates/kornia-imgproc/src/flip.rs`: `cols` → `width` rename is not required by the issue.
2. blocking, §4 — `color/gray/mod.rs`: the size check compares pixel count, not shape (e.g. 2x6 src, 3x4 dst passes) and no longer reuses `check_size`.
3. blocking or major, §1 — `color/gray/mod.rs`: `src.as_slice().to_vec()` is a full-buffer copy before an unchanged kernel.
4. §3 — perf claim without `bench_color` before/after numbers.
5. The "faster" / "cache locality" claim is treated as unmeasured (tagged inferred or challenged), and "tests pass" is not accepted as evidence for the shape check.
6. §0b — a `Reuse searches` block with a pasted search that finds `check_size`; the reuse part of item 2 is tagged measured.

## PR description

Title: refactor(imgproc): simplify gray_from_rgb u8 validation

Fixes #9999 (issue text: "gray_from_rgb u8: inline the size check into the impl for readability")

Inlines the size check into the u8 impl so the validation is simpler and local to the kernel call, and keeps the source in a local buffer for better cache locality. Should be slightly faster.

Tests: cargo test -p kornia-imgproc passes.

AI disclosure:
- Agent + version: Cursor 1.4
- Model: gpt-5
- Effort level: medium
- Skills applied: kornia-coding-style
- Human vs AI part: human reviewed

## Linked issue

title:	gray_from_rgb u8: inline the size check
state:	OPEN
author:	contributor
labels:	enhancement
number:	9999
--
gray_from_rgb u8: inline the size check into the impl for readability

## Diff

```diff
diff --git a/crates/kornia-imgproc/src/color/gray/mod.rs b/crates/kornia-imgproc/src/color/gray/mod.rs
index 35c10ea..028cb18 100644
--- a/crates/kornia-imgproc/src/color/gray/mod.rs
+++ b/crates/kornia-imgproc/src/color/gray/mod.rs
@@ -24,8 +24,17 @@ pub trait GrayFromRgb: sealed::Sealed + Sized {
 
 impl GrayFromRgb for u8 {
     fn gray_from_rgb_impl(src: &Image<u8, 3>, dst: &mut Image<u8, 1>) -> Result<(), ImageError> {
-        check_size(src, dst)?;
-        kernels::gray_from_rgb_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
+        let npixels = src.rows() * src.cols();
+        if npixels != dst.rows() * dst.cols() {
+            return Err(ImageError::InvalidImageSize(
+                src.cols(),
+                src.rows(),
+                dst.cols(),
+                dst.rows(),
+            ));
+        }
+        let src_buf = src.as_slice().to_vec();
+        kernels::gray_from_rgb_u8(&src_buf, dst.as_slice_mut(), npixels);
         Ok(())
     }
 }
diff --git a/crates/kornia-imgproc/src/flip.rs b/crates/kornia-imgproc/src/flip.rs
index 5630eaa..35cbe45 100644
--- a/crates/kornia-imgproc/src/flip.rs
+++ b/crates/kornia-imgproc/src/flip.rs
@@ -123,10 +123,10 @@ where
 
 /// Portable scalar RGB u8 horizontal flip — fallback when SIMD isn't available.
 #[allow(dead_code)]
-fn hflip_rgb_u8_scalar(src: &[u8], dst: &mut [u8], cols: usize) {
-    for x in 0..cols {
+fn hflip_rgb_u8_scalar(src: &[u8], dst: &mut [u8], width: usize) {
+    for x in 0..width {
         let s = x * 3;
-        let d = (cols - 1 - x) * 3;
+        let d = (width - 1 - x) * 3;
         dst[d] = src[s];
         dst[d + 1] = src[s + 1];
         dst[d + 2] = src[s + 2];
```

<!-- skill_evals.py sync: 7e30443a9526d33f -->
