Verdict: request changes

Findings, blocking first:

- PR body — blocking — Self-review is a placeholder. It contains only `Verdict: approve` and `Skill gaps: none`, with no per-section results and no `Reuse searches` block (§6). Re-run the skill and paste the full output. — [inferred]
- PR body — blocking — Author walkthrough is generic ("Per file: see description", "covered by the unit tests"). It explains no hunk and names no real risk. The real risks are the Jacobi convergence threshold and degenerate inputs (§6). The author should write it themselves. A question for them: what does `fit_plane` return for collinear or coincident points, and why is `1e-15` an absolute threshold? — [inferred]
- PR body — blocking — Issue #9997 is readable and the diff maps to it, so scope itself is fine. The body also lists "Skills applied: kornia-coding-style, kornia-api-flavor" but not `kornia-code-review`, so no review step was recorded. — [inferred]
- crates/kornia-3d/src/plane.rs:44 — blocking — The convergence test `off < 1e-15` is absolute, not relative to the matrix scale. For small-scale clouds (spread around 1e-6, so covariance entries around 1e-12) it exits before any rotation. `v` stays the identity and the "smallest eigenvalue" normal is an arbitrary axis, so the plane is silently wrong. For large-scale clouds (covariance around 1e6) it never reaches 1e-15 and wastes all 50 sweeps. Fix: scale the threshold by the Frobenius norm or the diagonal sum, and add a test at 1e-6 and 1e3 scales. — [inferred]
- crates/kornia-3d/src/plane.rs:98 — major — Degenerate input is not detected. Collinear or coincident points (3 or more identical points, or a rank-1 covariance) have two or three equal smallest eigenvalues. The function returns `Ok` with an arbitrary normal, and NaN input propagates silently. The `# Errors` section only covers `TooFewPoints`. Fix: add a `Degenerate` variant when the two smallest eigenvalues are not separated, and document it. — [inferred]
- crates/kornia-3d/src/plane.rs:19 — major — Reuse. `crates/kornia-algebra/src/linalg/svd.rs:242` already has a `jacobi_eigenanalysis`, and `svd.rs:817` has `pub fn svd3(mat: &Mat3F64)`. `kornia-algebra` is already a dependency of `kornia-3d` (Cargo.toml:35). `dot3` and `normalize3` duplicate `Vec3F64::dot` and `normalize` (`kornia-algebra/src/vec.rs:53,61`). The PR adds a third hand-written 3x3 eigen-solver (a 6x6 one already exists at `kornia-calib/src/covariance.rs:25`) with no bench or measurement showing why `svd3` on the covariance is insufficient. Fix: build the covariance as a `Mat3F64`, call `svd3`, and take the last singular vector. Otherwise justify the new solver with a measurement. — [measured: `Grep` hits for `fn \w*(eigen|jacobi|svd)\w*` and `fn (dot|normalize)` over `crates/`]
- crates/kornia-3d/src/plane.rs:117 — major — The tests give no evidence for the solver. `fit_plane_z0` uses axis-aligned points whose covariance is already diagonal, so every `m[p][q]` is 0 and the rotation branch never runs (§4 no-op input). Nothing pins values from a reference (numpy `eigh` or SVD on a tilted or noisy plane, with the generation method stated). There is no test for `d`, for a tilted plane, for the sign convention, or for the degenerate and scale cases. Fix: add a tilted noisy plane with numpy-generated expected values, plus the scale tests above. — [inferred]
- crates/kornia-3d/src/plane.rs:50 — minor — The rustdoc names no reference (§0b.4). The normal sign is arbitrary and undocumented, so `d` is sign-ambiguous across calls. Fix: document the sign, or orient the normal deterministically. — [inferred]
- crates/kornia-3d/src/plane.rs:1 — minor — Public API choices don't match the rest of the crate. The `f64` `[[f64; 3]]` input and the `Plane` type are used instead of `Vec3F64`, which the rest of the crate uses. It follows the `kornia-api-flavor` point-set convention only if that skill says so, which I didn't check. The `#[allow(clippy::needless_range_loop)]` at line 32 is a smell. — [inferred]
- PR body — minor — There is no `changelog.d/<PR>.added.md`. `changelog.d/` holds only a README, and this is a user-visible new public API (§6). — [inferred]
- PR body — minor — "cargo test -p kornia-3d plane passes" is unquoted. I did not run it, since cargo is forbidden here. There is no output, test count or MSRV check. — [inferred]
- crates/kornia-3d/src/plane.rs:49 — nit — The Jacobi routine is correct in form: the rotation formulas match J^T·A·J, and `signum(0.0)` is `+1`, so `theta == 0` is fine. It takes only one `jacobi_symmetric` call per fit, so there is no allocation or copy concern. Bindings are not touched, but the PR does not say the API is not exposed. — [inferred]

Reuse searches:
- plane/eigen/jacobi/covariance/centroid/svd (3x3 symmetric eigen-solve)
  `Grep "fn \w*(plane|eigen|jacobi|covariance|centroid|svd)\w*" crates/`
  - crates/kornia-algebra/src/linalg/svd.rs:242: `fn jacobi_eigenanalysis(mut s: Symmetric3x3) -> Mat3F32`
  - crates/kornia-algebra/src/linalg/svd.rs:817: `pub fn svd3(mat: &Mat3F64) -> SVD3Set`
  - crates/kornia-calib/src/covariance.rs:25: `fn symmetric_eigen(mut a: [[f64; 6]; 6])`
  - No existing `fit_plane`.
- dot3 / normalize3 (vector dot and normalize)
  `Grep "fn (dot|normalize|cross|mean)\b" crates/`
  - crates/kornia-algebra/src/vec.rs:53: `pub fn dot(self, rhs: Self)`
  - crates/kornia-algebra/src/vec.rs:61: `pub fn normalize(self)`
  - Local `dot` copies also exist at kornia-3d/src/pnp/ap3p.rs:755 and kornia-3d/src/pnp/epnp.rs:494.
- centroid: no existing point-set centroid helper found (`fn \w*centroid\w*` returned no hits in `crates/`).
- Issue-number grep for inline notes referencing #9997: no hits in `crates/`, `kornia-py` or `kornia-cpp`.

Skill gaps:
- The self-review missed two blocking findings (the placeholder Self-review and the generic walkthrough) and the major reuse finding. The `kornia-code-review` skill is not listed among the skills applied. Suggested rule for `kornia-workflow`: the PR template must list `kornia-code-review` as applied, and a bare "approve / none" is rejected.
- Proposed rule for `kornia-coding-style`: iterative numeric solvers use relative convergence tolerances, and tests cover scale extremes and degenerate rank. A planted eval case for a Jacobi solver with an absolute epsilon would catch the threshold bug.
- Ambiguity in the skill: §0b says "composed from primitives", but it doesn't say whether a 3x3 covariance eigen-solve counts as covered by `svd3`. It would help to point to the `kornia-algebra` linalg entry in the `kornia-arch` table.