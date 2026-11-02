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
1. major or blocking, §0b — the inline serial 256-bin histogram loop re-implements `histogram::compute_histogram` (row-parallel, CUDA-dispatching) in the same crate. Fix: call it.
2. major, §0b.4 — no reference is named for the algorithm (Otsu 1979, OpenCV `THRESH_OTSU` or scikit-image `threshold_otsu`), and the only test is a range check (`t >= 12 && t < 200`). Fix: cite the reference in the rustdoc and pin a threshold generated with OpenCV or scikit-image on a stated input.
3. minor or major, §3 — a new op with no bench in `crates/kornia-imgproc/benches/` (there is no `bench_threshold`).
4. §0b — a `Reuse searches` block with a pasted search that finds `compute_histogram`; finding 1 is tagged measured.
Items 1, 2 and 4 are the pass bar.

## PR description

Title: feat(imgproc): add Otsu threshold

Fixes #9996 (issue text: "Implement Otsu's automatic threshold for u8 grayscale images (see TODO in threshold.rs).")

Adds `threshold_otsu`, which computes the histogram, picks the threshold that maximises between-class variance, and applies `threshold_binary`. Returns the chosen threshold.

Tests: cargo test -p kornia-imgproc threshold passes.

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

title:	feat(imgproc): Otsu threshold
state:	OPEN
author:	contributor
labels:	enhancement
number:	9996
--
Implement Otsu's automatic threshold for u8 grayscale images (see TODO in threshold.rs).

## Diff

```diff
diff --git a/crates/kornia-imgproc/src/threshold.rs b/crates/kornia-imgproc/src/threshold.rs
index be81b91..f189833 100644
--- a/crates/kornia-imgproc/src/threshold.rs
+++ b/crates/kornia-imgproc/src/threshold.rs
@@ -365,7 +365,66 @@ where
     Ok(())
 }
 
-// TODO: otsu, triangle
+/// Thresholds a grayscale image with Otsu's method and returns the chosen threshold.
+///
+/// The threshold maximises the between-class variance of the intensity histogram.
+/// Pixels above the threshold are set to `max_value`, the rest to 0.
+///
+/// # Arguments
+///
+/// * `src` - The input grayscale image.
+/// * `dst` - The output image, same size as `src`.
+/// * `max_value` - The value written for pixels above the threshold.
+pub fn threshold_otsu(
+    src: &Image<u8, 1>,
+    dst: &mut Image<u8, 1>,
+    max_value: u8,
+) -> Result<u8, ImageError> {
+    if src.size() != dst.size() {
+        return Err(ImageError::InvalidImageSize(
+            src.cols(),
+            src.rows(),
+            dst.cols(),
+            dst.rows(),
+        ));
+    }
+
+    let mut hist = [0usize; 256];
+    for &v in src.as_slice() {
+        hist[v as usize] += 1;
+    }
+
+    let total = src.as_slice().len() as f64;
+    let sum_all: f64 = hist.iter().enumerate().map(|(i, &h)| i as f64 * h as f64).sum();
+
+    let mut sum_bg = 0.0;
+    let mut weight_bg = 0.0;
+    let mut best_var = 0.0;
+    let mut best_t = 0u8;
+    for (t, &h) in hist.iter().enumerate() {
+        weight_bg += h as f64;
+        if weight_bg == 0.0 {
+            continue;
+        }
+        let weight_fg = total - weight_bg;
+        if weight_fg == 0.0 {
+            break;
+        }
+        sum_bg += t as f64 * h as f64;
+        let mean_bg = sum_bg / weight_bg;
+        let mean_fg = (sum_all - sum_bg) / weight_fg;
+        let var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
+        if var > best_var {
+            best_var = var;
+            best_t = t as u8;
+        }
+    }
+
+    threshold_binary(src, dst, best_t, max_value)?;
+    Ok(best_t)
+}
+
+// TODO: triangle
 
 #[cfg(test)]
 mod tests {
@@ -546,4 +605,15 @@ mod tests {
 
         Ok(())
     }
+
+    #[test]
+    fn threshold_otsu_bimodal() -> Result<(), ImageError> {
+        let data = vec![10u8, 12, 11, 200, 210, 205];
+        let image = Image::<_, 1>::new(ImageSize { width: 3, height: 2 }, data)?;
+        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;
+        let t = super::threshold_otsu(&image, &mut thresholded, 255)?;
+        assert!(t >= 12 && t < 200);
+        assert_eq!(thresholded.as_slice(), &[0, 0, 0, 255, 255, 255]);
+        Ok(())
+    }
 }
```

<!-- skill_evals.py sync: c926efd48f7ef4d9 -->
