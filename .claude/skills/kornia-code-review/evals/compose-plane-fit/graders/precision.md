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
1. major or blocking, §0b — the hand-written `jacobi_symmetric` eigen-solver re-implements an existing decomposition. Preferred fix (`kornia-arch`, linear algebra): build the covariance as `Mat3F64` and call `kornia_algebra::linalg::svd::svd3_f64` (used in `kornia-3d/src/pose/planar.rs`, `fundamental.rs`, `homography.rs`); the normal is the singular vector of the smallest singular value. Pointing at nalgebra `symmetric_eigen()` (already used in kornia-3d) also passes.
2. minor or major, §0b — `dot3`, `normalize3`, `centroid` and the `[f64; 3]` types duplicate `kornia_3d::linalg::dot_product3` and `Vec3F64` operations; use kornia-algebra types.
3. §0b — a `Reuse searches` block with pasted searches that find an existing eigen/SVD routine and the vector helpers; finding 1 is tagged measured.
Items 1 and 3 are the pass bar. Bonus: the test uses an axis-aligned square whose covariance is already diagonal, so the Jacobi loop never rotates (the test does not reach the algorithm); degenerate input (collinear, coincident, non-finite) returns `Ok`.

## PR description

Title: feat(3d): add least-squares plane fitting

Fixes #9997 (issue text: "Add a plane fit for 3D point sets (total least squares), needed for ground-plane estimation.")

Adds `kornia_3d::plane::fit_plane`: centroid, covariance, and a small cyclic Jacobi eigen-solver; the normal is the eigenvector of the smallest eigenvalue. No new dependencies.

Tests: cargo test -p kornia-3d plane passes.

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

title:	feat(3d): least-squares plane fit for point sets
state:	OPEN
author:	contributor
labels:	enhancement
number:	9997
--
Add a plane fit for 3D point sets (total least squares), needed for ground-plane estimation.

## Diff

```diff
diff --git a/crates/kornia-3d/src/lib.rs b/crates/kornia-3d/src/lib.rs
index 12e79b8..2f3bde5 100644
--- a/crates/kornia-3d/src/lib.rs
+++ b/crates/kornia-3d/src/lib.rs
@@ -19,6 +19,9 @@ pub mod linalg;
 /// Operations on 3D data processing.
 pub mod ops;
 
+/// Plane fitting.
+pub mod plane;
+
 /// Point cloud traits.
 pub mod pointcloud;
 
diff --git a/crates/kornia-3d/src/plane.rs b/crates/kornia-3d/src/plane.rs
new file mode 100644
index 0000000..093042b
--- /dev/null
+++ b/crates/kornia-3d/src/plane.rs
@@ -0,0 +1,133 @@
+/// Error type for plane fitting.
+#[derive(Debug, thiserror::Error, PartialEq)]
+pub enum PlaneFitError {
+    /// Fewer than three points were given.
+    #[error("need at least 3 points, got {0}")]
+    TooFewPoints(usize),
+}
+
+/// A plane `n · p + d = 0` with unit normal `n`.
+#[derive(Debug, Clone, Copy, PartialEq)]
+pub struct Plane {
+    /// Unit normal.
+    pub normal: [f64; 3],
+    /// Signed offset from the origin.
+    pub d: f64,
+}
+
+fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
+    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
+}
+
+fn normalize3(a: &[f64; 3]) -> [f64; 3] {
+    let n = dot3(a, a).sqrt();
+    [a[0] / n, a[1] / n, a[2] / n]
+}
+
+fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
+    let mut c = [0.0; 3];
+    for p in points {
+        c[0] += p[0];
+        c[1] += p[1];
+        c[2] += p[2];
+    }
+    let n = points.len() as f64;
+    [c[0] / n, c[1] / n, c[2] / n]
+}
+
+/// Cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix.
+///
+/// Returns `(eigenvectors_as_columns, eigenvalues)`.
+#[allow(clippy::needless_range_loop)]
+fn jacobi_symmetric(a: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
+    let mut m = *a;
+    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
+    for _ in 0..50 {
+        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
+        if off < 1e-15 {
+            break;
+        }
+        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
+            if m[p][q].abs() < 1e-300 {
+                continue;
+            }
+            let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
+            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
+            let c = 1.0 / (t * t + 1.0).sqrt();
+            let s = t * c;
+            for k in 0..3 {
+                let mkp = m[k][p];
+                let mkq = m[k][q];
+                m[k][p] = c * mkp - s * mkq;
+                m[k][q] = s * mkp + c * mkq;
+            }
+            for k in 0..3 {
+                let mpk = m[p][k];
+                let mqk = m[q][k];
+                m[p][k] = c * mpk - s * mqk;
+                m[q][k] = s * mpk + c * mqk;
+            }
+            for k in 0..3 {
+                let vkp = v[k][p];
+                let vkq = v[k][q];
+                v[k][p] = c * vkp - s * vkq;
+                v[k][q] = s * vkp + c * vkq;
+            }
+        }
+    }
+    (v, [m[0][0], m[1][1], m[2][2]])
+}
+
+/// Fits a plane to 3D points by least squares (total least squares on the centroid-centred
+/// covariance). The normal is the eigenvector of the smallest eigenvalue.
+///
+/// Points and the returned plane are in the same (caller's) frame and units.
+pub fn fit_plane(points: &[[f64; 3]]) -> Result<Plane, PlaneFitError> {
+    if points.len() < 3 {
+        return Err(PlaneFitError::TooFewPoints(points.len()));
+    }
+    let c = centroid(points);
+    let mut cov = [[0.0; 3]; 3];
+    for p in points {
+        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
+        for i in 0..3 {
+            for j in 0..3 {
+                cov[i][j] += d[i] * d[j];
+            }
+        }
+    }
+    let (vecs, vals) = jacobi_symmetric(&cov);
+    let mut k = 0;
+    for i in 1..3 {
+        if vals[i] < vals[k] {
+            k = i;
+        }
+    }
+    let normal = normalize3(&[vecs[0][k], vecs[1][k], vecs[2][k]]);
+    Ok(Plane {
+        normal,
+        d: -dot3(&normal, &c),
+    })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn fit_plane_z0() -> Result<(), PlaneFitError> {
+        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
+        let plane = fit_plane(&pts)?;
+        assert!((plane.normal[2].abs() - 1.0).abs() < 1e-9);
+        assert!(plane.d.abs() < 1e-9);
+        Ok(())
+    }
+
+    #[test]
+    fn fit_plane_too_few() {
+        assert_eq!(
+            fit_plane(&[[0.0; 3], [1.0; 3]]),
+            Err(PlaneFitError::TooFewPoints(2))
+        );
+    }
+}
```

<!-- skill_evals.py sync: 584376f14df03421 -->
