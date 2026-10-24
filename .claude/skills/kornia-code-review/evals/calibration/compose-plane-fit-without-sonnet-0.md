**Review: feat(3d): add least-squares plane fitting (#9997)**

**Verdict: request changes.** The algorithm is the right one and the Jacobi rotation algebra looks correct. But there is a scale-dependent correctness bug, degenerate inputs are not handled, and the new code duplicates solvers the crate already depends on. All of this is from reading the diff; I did not build or run anything.

**Blocking**

1. **Wrong normal for small-scale data (inferred from reading the code).** `jacobi_symmetric` stops when the sum of the off-diagonals is below `1e-15`, an absolute threshold. The covariance is an unnormalized sum of squares, so it scales with the square of the coordinate size.
   - Take a tilted plane with extent around 1e-8 (for example, points in meters at sub-micron scale, or a normalized/tiny cloud). Its off-diagonals are about 1e-16, so the loop breaks before any rotation.
   - `v` is then still the identity, and the "normal" is whichever axis has the smallest diagonal entry. That is silently wrong, not an error.
   - The fix is to make the test relative (`off <= eps * (m00 + m11 + m22)`), or to normalize the covariance first.
   - The inner skip `m[p][q].abs() < 1e-300` has the same problem in spirit, though it is much less likely to matter.

2. **No handling of degenerate input.** `fit_plane` only checks `len() < 3`.
   - Collinear points, coincident points, and inputs containing NaN or inf all return `Ok(Plane)` with an arbitrary normal. For identical points the normal is just `e0`. The doc doesn't say so.
   - The ground-plane use case in the issue will hit this. Add an error variant such as `Degenerate`, for example when the middle eigenvalue is ≲ eps times the largest, or the sum is non-finite. Also validate that the inputs are finite.

3. **Tests don't exercise the solver.** The only fit test uses the z=0 square.
   - There the covariance is already diagonal, so the Jacobi rotation loop never runs. The code that matters has zero coverage, and it would have caught item 1.
   - Add at least: a tilted plane with a known normal (compare up to sign); noisy points with the normal checked against a reference; a tiny-scale and a large-scale (translated far from the origin) cloud; collinear and duplicate points; and a comparison against `nalgebra`'s `SymmetricEigen` or SVD.
   - The PR claims "numerical edge cases, covered by the unit tests". They aren't covered.

4. **Reinvents existing utilities (AI_POLICY Law 2).** `kornia-3d` already depends on `nalgebra` and `faer`, and uses `.svd()` in `pose/triangulation.rs` and `pose/essential.rs`.
   - A hand-written Jacobi solver, `dot3`, `normalize3` and `centroid` are all new helpers that nalgebra already provides (`SymmetricEigen` on a `Matrix3`, or the SVD of the centered points).
   - "No new dependencies" is not a reason to avoid a dependency the crate already has. Either use it, or justify the custom solver with a benchmark or accuracy argument.
   - Law 1 also requires a library reference in the description (OpenCV, PCL, nalgebra, ...). None is given.

**Should fix**

5. **Sign ambiguity.** The normal's sign is arbitrary, because it is whatever sign the eigenvector comes out with. Callers doing ground-plane work usually need a convention, such as pointing toward a given direction. Document this, or offer an orientation option.

6. **Missing API and docs conventions (kornia-api-flavor #10, #6).**
   - There are no Arguments/Returns/Errors/Example sections in the rustdoc.
   - There is no `changelog.d/` fragment.
   - There is no crate README entry.
   - There is no statement of the units of `d`.

7. **Type choice.** The API is `f64` and `[f64; 3]` only. Other `kornia-3d` APIs use `f32`, `Vec3` from `kornia-algebra` and the pointcloud types.
   - Consider taking those types, or being generic, so it composes with `PointCloud`. At minimum, say why raw arrays are used.

8. **Minor.**
   - `#[allow(clippy::needless_range_loop)]` is blanket-applied to the whole function.
   - The three rotation loops can be simplified.
   - `normalize3` divides by `n`, which is unguarded; it is only safe because `n` can't be 0 for a unit eigenvector.
   - The test array formatting looks like it wasn't run through `rustfmt`.

**PR process**

- The description says "`cargo test -p kornia-3d plane` passes" but pastes no log, which AI_POLICY Law 1 requires.
- The Author walkthrough is a placeholder ("see description"). The "riskiest part" claim, numerical edge cases covered by tests, doesn't hold (see item 3).
- The self-review says "Verdict: approve / Skill gaps: none", but the findings above show otherwise.
