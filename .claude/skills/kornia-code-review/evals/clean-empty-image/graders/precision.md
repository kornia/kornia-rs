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

A passing review approves (negative control: measures false positives):
1. No blocking or major finding, except optionally a request for before/after bench numbers on the changed rayon splits.
2. No blocking or major finding is false (`graders/precision.md`, reported, not scored).

## PR description

Title: fix(imgproc): no panic on zero-size images in distance_transform and compute_histogram

Fixes #1163

User-visible fix: `changelog.d/1166.fixed.md`.

## Changes Made
- `distance_transform.rs`: the four rayon splits (`grid` and `src` rows, `scratch` columns, the `transpose_map` destination) go through the crate's `parallel::par_rows_exact_mut` / `par_row_chunks` helpers, which yield no rows for a zero-size image instead of panicking on a zero chunk size. The Returns section now specifies the output shape for empty input.
- `histogram.rs`: `par_chunks(ROWS_PER_TASK * width)` becomes `par_row_chunks(src_slice, width, ROWS_PER_TASK)`, the same split for non-empty input.

#1163 reproduces the panic in these two functions. It lists the other row-splitting sites as possibilities the reporter had not checked.

## How Was This Tested?
- Unit tests `distance_transform_empty_image` and `test_compute_histogram_empty_image` cover 0x5, 5x0 and 0x0. They fail on main (tests applied to main's code, source hunks left out) and pass here:
```text
$ cargo test -p kornia-imgproc --lib empty_image   # main + the two new tests
test distance_transform::tests::distance_transform_empty_image ... FAILED
test histogram::tests::test_compute_histogram_empty_image ... FAILED
thread 'distance_transform::tests::distance_transform_empty_image' panicked at crates/kornia-imgproc/src/distance_transform.rs:75:14:
chunk_size must not be zero
thread 'histogram::tests::test_compute_histogram_empty_image' panicked at crates/kornia-imgproc/src/histogram.rs:74:10:
chunk_size must not be zero
test result: FAILED. 6 passed; 2 failed; 0 ignored; 0 measured; 395 filtered out; finished in 0.01s

$ cargo test -p kornia-imgproc --lib empty_image   # this PR
test histogram::tests::test_compute_histogram_empty_image ... ok
test distance_transform::tests::distance_transform_empty_image ... ok
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 395 filtered out; finished in 0.00s
```
- Non-empty images: `par_row_chunks(d, w, n)` is `d.par_chunks(w * n)` and `par_rows_exact_mut(d, w)` is `d.par_chunks_exact_mut(w)` once `w > 0` (`parallel.rs:26,36,46`). The immutable source uses `par_row_chunks(src_slice, width, 1)` because the exact-row helper takes a mutable slice. The source and grid both contain exactly `width * height` elements, so the two iterators yield the same rows. No perf change is claimed and no bench was run.

## Self-review
```text
Verdict: approve
§0 — no finding — #1163 names both panic sites. The zero-size tests and the empty-output rustdoc cover those sites. — [inferred]
§0b — no finding — the row helpers in the search below handle zero-length buffers. — [measured]
§1–3 — no finding — for non-empty buffers the helpers use the same chunk sizes and borrow the same slices. No speedup claimed. — [inferred]
§4 — no finding — the empty_image test output above goes from two chunk-size panics to “8 passed; 0 failed”. — [measured]
§5 — no finding — the fix uses safe slice helpers and documents the empty output. — [inferred]
§6 — no finding — the changelog fragment describes the zero-size panic. — [inferred]
Reuse searches:
- zero-size-safe row split (split a buffer into rows without panicking on empty images)
  $ rg -n "fn par_row|fn par_rows" crates/
  crates/kornia-imgproc/src/parallel.rs:21:pub(crate) fn par_row_chunks_mut<T: Send>(
  crates/kornia-imgproc/src/parallel.rs:31:pub(crate) fn par_row_chunks<T: Sync>(
  crates/kornia-imgproc/src/parallel.rs:42:pub(crate) fn par_rows_exact_mut<T: Send>(
  -> reused; no new helper.
Skill gaps: none
```

## Author walkthrough
- **Per file:** `distance_transform.rs` splits buffers with rayon at four sites (`grid` and `src` rows, `scratch` columns, and the `transpose_map` destination). With width or height 0 the chunk size is 0 and rayon panics, which is the bug in #1163. The crate already has row helpers that skip empty buffers, so I switched all four to them rather than adding an early return. `histogram.rs` had the same split once.
- **Riskiest part and how I checked it:** changing the split in a hot loop. For non-empty input the helpers reduce to the same `par_chunks` / `par_chunks_exact_mut` calls (they only clamp a zero chunk size to 1), and `chunks_exact` matches `chunks` because every buffer is exactly `width * height`. The existing distance-transform and histogram tests still pass.

## AI Usage Disclosure
- **Agent + version:** Claude Code 2.1
- **Model:** claude-sonnet-5-5
- **Effort level:** medium
- **Skills applied:** kornia-workflow, kornia-coding-style, kornia-code-review
- **Human vs AI part:** human diagnosed the panic and chose the helpers; agent wrote the tests and rustdoc

## Linked issue

title:	[Bug]: distance_transform and compute_histogram panic on zero-width images
state:	OPEN
author:	Aatman09
labels:	bug, help wanted
assignees:	Aatman09
projects:
milestone:
number:	1163
--
### 🐛 Describe the bug

`DistanceTransformExecutor::execute` and `compute_histogram` in `kornia-imgproc` panic when the input image has `width == 0`.

`Image::new` accepts a zero-width image without error, so the panic can be reached through the public API. Both functions split the pixel buffer with rayon's `par_chunks` using a chunk size derived from the width, and rayon panics when the chunk size is 0:

- `crates/kornia-imgproc/src/distance_transform.rs:75`: `par_chunks_mut(width)`
- `crates/kornia-imgproc/src/histogram.rs:74`: `par_chunks(ROWS_PER_TASK * width)`


### 🔄 Steps to Reproduce

```bash
1. Create an Image<f32, 1> or Image<u8, 1> with ImageSize { width: 0, height: 5 } and an empty data vector. Image::new returns Ok.
2. Pass it to DistanceTransformExecutor::new().execute(&img) or to compute_histogram(&img, &mut hist, 4).
3. Both calls panic with `chunk_size must not be zero` instead of returning.
```

### 💻 Minimal Code Example

```rust
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{distance_transform::DistanceTransformExecutor, histogram::compute_histogram};

#[test]
fn dt_zero_width() {
    let img = Image::<f32, 1>::new(ImageSize { width: 0, height: 5 }, vec![]).unwrap();
    let _ = DistanceTransformExecutor::new().execute(&img); // panics
}

#[test]
fn hist_zero_width() {
    let img = Image::<u8, 1>::new(ImageSize { width: 0, height: 5 }, vec![]).unwrap();
    let mut hist = vec![0usize; 4];
    let _ = compute_histogram(&img, &mut hist, 4); // panics
}
```

### ✅ Expected behavior

No panic. Either:
- return early: `compute_histogram` leaves `hist` unchanged, and `execute` returns an empty image of the same size; or
- return an `ImageError` for empty input.

Happy to go with whichever the maintainers prefer. Returning early seems more natural, since `Image::new` treats an empty image as valid.


### ❌ Actual behavior

thread dt_zero_width panicked at crates/kornia-imgproc/src/distance_transform.rs:75:14:
chunk_size must not be zero

thread 'hist_zero_width' panicked at crates/kornia-imgproc/src/histogram.rs:74:10:
chunk_size must not be zero


### 🔧 Environment

```shell
- kornia-rs version: 0.2.0 (main @ be0ac0cb)
- Rust version (`rustc -V`): rustc 1.96.0 (ac68faa20 2026-05-25)
- Cargo version (`cargo -V`): cargo 1.96.0 (30a34c682 2026-05-25)
- OS (e.g., Linux, macOS, Windows): Linux
- Target architecture (if cross-compiling): x86_64 (not cross-compiling)
- Python version (if using Python bindings): N/A
```

### 📝 Additional context

- Other functions split rows by width the same way and may have the same issue: `features/fast.rs`, `features/responses.rs`, `filter/ops.rs`, `color/yuv/kernels.rs`. I haven't checked whether they already guard against `width == 0`; I can check them as part of the fix.
- Related but separate: for an image with no foreground pixels, `DistanceTransformExecutor::execute` returns `1e10`, while `distance_transform_vanilla` returns `f32::MAX`. Not part of this fix; mentioning it in case it should be tracked.


### 🤝 Contribution Intent

- [x] I plan to submit a PR to fix this bug
- [ ] I'm reporting this bug but not planning to fix it

## Diff

```diff
diff --git a/crates/kornia-imgproc/src/distance_transform.rs b/crates/kornia-imgproc/src/distance_transform.rs
index b80dc2bb..2a7101d5 100644
--- a/crates/kornia-imgproc/src/distance_transform.rs
+++ b/crates/kornia-imgproc/src/distance_transform.rs
@@ -1,3 +1,4 @@
+use crate::parallel::{par_row_chunks, par_rows_exact_mut};
 use kornia_image::{Image, ImageError, ImageSize};
 use rayon::prelude::*;
 
@@ -65,7 +66,8 @@ impl DistanceTransformExecutor {
     /// # Returns
     ///
     /// An image of the same size holding the distance from each pixel to the
-    /// nearest foreground pixel.
+    /// nearest foreground pixel. An image with zero width or height returns an
+    /// empty image of the same size.
     ///
     /// If the image has no foreground pixels, every output value is `1e10`
     /// (not `f32::MAX` as in [`distance_transform_vanilla`]). Distances are
@@ -89,9 +91,8 @@ impl DistanceTransformExecutor {
         let src_slice = image.as_slice();
 
         // Parallelize over rows and fuse binary-to-INF conversion with the transform.
-        self.grid
-            .par_chunks_mut(width)
-            .zip(src_slice.par_chunks(width))
+        par_rows_exact_mut(&mut self.grid, width)
+            .zip(par_row_chunks(src_slice, width, 1))
             .for_each_init(
                 || (vec![0.0; width], vec![0usize; width], vec![0.0; width + 1]),
                 |(f, v, z), (grid_row, src_row)| {
@@ -104,7 +105,7 @@ impl DistanceTransformExecutor {
 
         transpose_map(&self.grid, &mut self.scratch, width, height, |x| x);
 
-        self.scratch.par_chunks_mut(height).for_each_init(
+        par_rows_exact_mut(&mut self.scratch, height).for_each_init(
             || {
                 (
                     vec![0.0; height],
@@ -176,7 +177,7 @@ fn transpose_map<F>(src: &[f32], dst: &mut [f32], width: usize, height: usize, o
 where
     F: Fn(f32) -> f32 + Sync + Send,
 {
-    dst.par_chunks_mut(height)
+    par_rows_exact_mut(dst, height)
         .enumerate()
         .for_each(|(x, dst_row)| {
             for y in 0..height {
@@ -233,4 +234,25 @@ mod tests {
 
         Ok(())
     }
+
+    #[test]
+    fn distance_transform_empty_image() -> Result<(), ImageError> {
+        let mut executor = DistanceTransformExecutor::new();
+
+        for (width, height) in [(0, 5), (5, 0), (0, 0)] {
+            let image = Image::<f32, 1>::new(ImageSize { width, height }, vec![])?;
+            let result = executor.execute(&image);
+            assert!(
+                result.is_ok(),
+                "case {width}x{height}: got {:?}",
+                result.as_ref().err()
+            );
+            let output = result?;
+            assert_eq!(output.width(), width, "case {width}x{height}");
+            assert_eq!(output.height(), height, "case {width}x{height}");
+            assert!(output.as_slice().is_empty(), "case {width}x{height}");
+        }
+
+        Ok(())
+    }
 }
diff --git a/crates/kornia-imgproc/src/histogram.rs b/crates/kornia-imgproc/src/histogram.rs
index 57ffe357..0a7c19fc 100644
--- a/crates/kornia-imgproc/src/histogram.rs
+++ b/crates/kornia-imgproc/src/histogram.rs
@@ -1,6 +1,8 @@
 use kornia_image::{Image, ImageError};
 use rayon::prelude::*;
 
+use crate::parallel::par_row_chunks;
+
 /// Compute the pixel intensity histogram of an image.
 ///
 /// NOTE: this is limited to 8-bit 1-channel images.
@@ -70,8 +72,7 @@ pub fn compute_histogram(
     // O(rows). Each task accumulates its rows into one local histogram before
     // the reduce step merges them.
     const ROWS_PER_TASK: usize = 16;
-    let partial_hist = src_slice
-        .par_chunks(ROWS_PER_TASK * width)
+    let partial_hist = par_row_chunks(src_slice, width, ROWS_PER_TASK)
         .map(|chunk| {
             let mut local_hist = vec![0_usize; num_bins];
             for &pixel in chunk {
@@ -118,6 +119,20 @@ mod tests {
 
         Ok(())
     }
+
+    #[test]
+    fn test_compute_histogram_empty_image() -> Result<(), ImageError> {
+        for (width, height) in [(0, 5), (5, 0), (0, 0)] {
+            let image = Image::<u8, 1>::new(ImageSize { width, height }, vec![])?;
+            let mut histogram = vec![7; 4];
+
+            let result = super::compute_histogram(&image, &mut histogram, 4);
+            assert!(result.is_ok(), "case {width}x{height}: got {result:?}");
+            assert_eq!(histogram, vec![7; 4], "case {width}x{height}");
+        }
+
+        Ok(())
+    }
 }
 
 /// OpenCV-exact histogram-equalization LUT from a 256-bin histogram.
diff --git a/changelog.d/1166.fixed.md b/changelog.d/1166.fixed.md
new file mode 100644
--- /dev/null
+++ b/changelog.d/1166.fixed.md
@@ -0,0 +1 @@
+For images with zero width or height, `DistanceTransformExecutor::execute` returns an empty image of the same size and `compute_histogram` leaves the histogram unchanged.
```

<!-- skill_evals.py sync: 010eda1da517b9ee -->
