# SQPnP review of PR #1183

Reviewed against PR head `34f5519aca12efafa0d2dfecde67ae6ae0139b17` using the architecture,
coding-style, API and workflow guidance from [PR #1206](https://github.com/kornia/kornia-rs/pull/1206).
The draft guidance was read as a reference; its unrelated CI changes are not included here.

## Review sources and applicability

The review also read [`best_practices.md`](../best_practices.md), the repository's `AGENTS.md`,
all PR #1183 review comments, and the discussions and final changes in three recently merged PRs:

| Reference | Practice applied here |
| --- | --- |
| [#1195: weighted Umeyama](https://github.com/kornia/kornia-rs/pull/1195) | Use Kornia types at the algebra boundary, define validation precedence, handle extreme magnitudes, and test geometric invariants independently. |
| [#1169: RANSAC engine](https://github.com/kornia/kornia-rs/pull/1169) | Preserve established behavior outside the feature; document deliberate exceptions and compare identical workloads. Shared reprojection helpers retain EPnP behavior; finite-result rejection stays in SQPnP. |
| [#1179: VideoWriter](https://github.com/kornia/kornia-rs/pull/1179) | Bound potentially unending work, report failures explicitly, and test the resulting data as well as completion. Scene sampling is bounded and fixture errors propagate. |

PR #1206 is still a draft. Its reviews caution that structural checks alone do not establish
correctness and that some proposed guidance is stale. Executable repository configuration and
the current contributor instructions take precedence over that draft. The screenshot is
review context, not a separate instruction source.

The applicable `best_practices.md` patterns are named numerical parameters, fallible operations,
factual documentation, domain result types, shared fixtures, minimal API changes and avoidance
of redundant hot-path work. Image-layout, color conversion and SIMD dispatch rules do not apply
to this pose solver. Parameter references follow the existing `PnPSolver` trait rather than
introducing a second calling convention for one implementation.

## Correctness findings

The original objective used the orthogonal projector `I - vvᵀ / (vᵀv)`. This minimizes squared
perpendicular distance to image rays. SQPnP's algebraic reprojection objective instead uses
`A = [[1, 0, -u], [0, 1, -v]]` and `Q = AᵀA`, giving residuals `X - uZ` and `Y - vZ` in
normalized image coordinates. These objectives agree on exact data but weight noisy data
differently. The corrected implementation accumulates `Q` and eliminates translation before
optimizing rotation. An independent test evaluates the two residuals directly, without using
the implementation's normal matrix.

A fixed noisy scene with a wide field of view also checks the pose against OpenCV 5.0.0's
`SOLVEPNP_SQPNP` result. The corrected solver passes; the original PR implementation fails.
The direct residual test also fails on the original implementation (`rᵀΩr = 3.186354475382372`
versus `3.200031840734217`). The baseline implementation was checked against the PR head;
only the shared benchmark harness and test oracles were changed in the temporary baseline.

References: [Terzakis and Lourakis, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460.pdf),
[OpenCV `computeOmega`](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/sqpnp.cpp).

Invalid SQPnP parameters previously changed branching and convergence behavior silently.
Tolerances now have documented finite ranges and the iteration cap must be nonzero. Unsupported
camera matrices are rejected before inversion: the solver requires positive focal lengths, zero
skew and homogeneous bottom row `[0, 0, 1]`, matching its reprojection model.

A failed KKT solve previously broke out of the loop and projected the preceding iterate into a
candidate. It now discards the candidate. Search also rejects non-finite objective costs and
centroid depths. Regression tests cover these cases.

## Review requests and changes

| Concern | Current implementation |
| --- | --- |
| Hand-written casts and matrix ordering | Existing `From` conversions remain the precision boundary. Constraint evaluation borrows column-major matrix views instead of copying through arrays. |
| Unclear copying | A borrowed `From<&Mat3F64>` nalgebra view shares the original storage; a pointer-identity test verifies this. Owned conversions and SVD workspace still copy values as required. Converting f32 to f64 cannot share the same storage. |
| nalgebra exposed throughout rotation fitting | `Mat3F64::try_svd` returns the existing Kornia SVD result type. Projection and decomposition share one bounded, scaled backend; projection neither normalizes twice nor restores unused singular values. Reflections are corrected in the smallest-singular-vector direction. The 9×9 objective and 15×15 KKT systems remain local nalgebra implementation details. |
| Candidate projection failures | Candidates are discarded on failed numerical solves or projection. Every returned SQPnP rotation passes through SO(3) projection. |
| Tuples of unrelated vectors | Search uses `Candidate` / `SqpSolution`; benchmarks use the same `Scene` and `SceneConfig` as tests. |
| Relative source imports | The benchmark imports shared fixtures through the crate's development-only `test-utils` feature. There are no source-path includes. |
| Duplicated benchmark fixtures | Removed the separate fixed-pose generator, camera constants and outlier sampling from the benchmark. Timing and accuracy now share seeded scenes. |
| Unchecked fixture inputs and unbounded rejection sampling | Scene generation and outlier insertion return exhaustive `SceneError` values. Validation precedes sampling or correspondence edits, storage reservation is fallible, and rejection sampling has a checked finite budget. Tests cover invalid ranges, non-finite observations, sampling exhaustion and preservation of correspondence data on failure. |
| Inconsistent camera models in SQPnP RANSAC | Both SQPnP variants reject distortion inside RANSAC, whose scoring is pinhole-only. Direct SQPnP retains distortion support. A regression also preserves EPnP's existing acceptance. |
| Naming and numeric parameters | Scene generation has explicit names and configuration. Solver tolerances and decomposition limits are named. |
| Documentation | Added required sections to the new conversion/projection/SVD APIs, documented parameter ranges and the benchmark feature, and corrected the objective description. |
| Benchmark and SLAM claims | Old tables are explicitly superseded because they measured a different objective and scene distribution. The changelog does not claim a SLAM accuracy improvement or a change to AP3P's EPnP refit. |
| Registration edge cases | Empty or mismatched correspondence arrays return `None`, without a library assertion or writes to output buffers. |

The bounded SVD backend is deliberate: the existing custom `svd3_f64` leaves `RᵀM` asymmetric
by approximately `2e-5` on the checked regression matrix. A nearest-rotation result should satisfy
this polar-decomposition stationarity condition; the new backend passes the `1e-12` check.
This does not replace the custom SVD in unrelated algorithms. Projection is also tested on an
all-`f64::MAX` rank-one matrix: its singular values overflow, but its nearest rotation remains
well-defined because only the singular vectors are needed.

## Scope and limitations

The public PnP API remains f32, as do EPnP, AP3P, LM refinement and RANSAC. Native f64 inputs and
results would need a coherent precision API across that pipeline; accepting f64 only to round
it through f32 LM would be misleading. The solver performs f64 optimization without allocating
converted correspondence arrays.

The pre-existing EPnP/Umeyama issue [#1181](https://github.com/kornia/kornia-rs/issues/1181) and
planar-pose reflection handling are not changed by this revision. RANSAC still scores pinhole
residuals; SQPnP now requires pre-undistorted observations there. EPnP's existing acceptance of
distortion remains for compatibility, with the model mismatch documented.
AP3P still refits with EPnP. These limitations must be considered in any downstream SLAM test.

The existing bundle-adjustment digest test
`ba_schur::tests::defaults_are_bit_identical_to_the_frozen_solver` fails on both the original
PR head and this revision on this machine: computed `0xe1fa3d679808a412`, frozen
`0x968c5cb0c993bf05`. Its expected digest has not been changed to hide the failure.

Synthetic tests and local review do not establish an improvement on a real camera or SLAM
sequence. The contributor still needs to review the code and verify their own use case before
requesting maintainer review.

This revision was prepared by Codex. The exact deployed model version and reasoning setting
are not exposed in this session. No human review, personal camera use case or downstream
SLAM improvement is claimed.

## Local validation

| Check | Result |
| --- | --- |
| SQPnP unit tests | 26 passed, including the OpenCV reference fixture and validation-precedence regression. |
| `cargo test -p kornia-algebra --lib` | 322 passed. |
| `cargo test -p kornia-3d --lib --features test-utils -- --skip defaults_are_bit_identical_to_the_frozen_solver` | 361 passed, 2 ignored, 1 explicitly filtered. |
| Doctests for both crates, with `kornia-3d/test-utils` | 34 passed, 2 ignored. |
| `cargo fmt --all -- --check`, `git diff --check` | Passed. |
| `python3 scripts/check_version_pins.py` | Passed; dependency version pins unchanged. |
| CI-feature test command | Fails on the existing bundle-adjustment digest assertion described above. |
| `cargo +1.96 clippy -p kornia-algebra -p kornia-3d --no-deps --all-targets --features kornia-3d/test-utils -- -D warnings` | Passed on Rust 1.96.1. The repository's CUDA lint job explicitly selects the 1.96 toolchain. This is an affected-crate check, not a full workspace/CUDA check. |
| Strict Clippy for both affected crates and all targets on default Rust 1.99.0 | Blocked by existing lints in unchanged COLMAP and fundamental-matrix code. |
| Isolated Clippy with only `chunks_exact_to_as_chunks` and `map_or_identity` suppressed on the command line | Passed with `-D warnings`; no lint suppressions added to source. |

The full CI checks and default-toolchain strict Clippy are **not green**. On this macOS installation, the initial Pixi CI build
also needed `CC=/usr/bin/clang`, `CXX=/usr/bin/clang++` and
`DYLD_FALLBACK_LIBRARY_PATH` pointing at `.pixi/envs/default/lib` to get past the CMake compiler
check and GStreamer library loading. These are local environment overrides, not repository
configuration changes. The release-script check also fails with `kornia: unbound variable`
in the host's Bash; no release/version edits are part of this change.

Before/after timing measurements and their limitations are recorded in
[`crates/kornia-3d/benchmarks.md`](../crates/kornia-3d/benchmarks.md). The first objective correction
was slower for planar 100- and 500-point scenes. The follow-up comparison ran under extreme
competing load and cannot establish a speedup; its 100-point samples were slightly slower.
