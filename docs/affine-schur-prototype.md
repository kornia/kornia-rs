# Affine Schur BA prototype

This branch tests an end-to-end float64 affine bundle adjuster before splitting
upstream contributions. Production `oblique-ug` remains unchanged. The adapter in
`scripts/benchmark_affine_ba.py` can replace its `affine.bundle_adjust` for experiments.

## Model and solver

`camera::AffineCamera` stores an unconstrained world-to-image 2×4 matrix, including
extrinsics. Projection is `M X + t`; its analytical Jacobians are `M` for the point
and two disjoint copies of `[X, Y, Z, 1]` for the eight camera coefficients. This is
an affine camera, not an SE(3) pose with pinhole intrinsics. Back-projection is not
part of this prototype's API.

`ba_affine::bundle_adjust_affine` assembles camera, point and cross blocks in f64,
damps both camera and point diagonals, eliminates the point increments, solves the
dense camera system, and back-substitutes point increments. nalgebra supplies the
Cholesky decompositions; no new linear-algebra dependency or custom inversion was
introduced. Cauchy weighting applies to each **2D vector norm**, with cost
`0.5 * scale² * sum(log1p(||r||² / scale²))`. LM accepts only finite cost-decreasing
trials and updates damping using the gain ratio. Iteration counts are linearizations,
not individual damping attempts. The result reports iteration counts and convergence.

World points are centered and isotropically scaled internally. Four noncoplanar
fixed points remove the 12-dimensional affine gauge. Additional fixed points impose
additional constraints. Anchor selection belongs to the caller; the experiment uses
pivoted QR on normalized homogeneous points. The solver rejects anchor matrices
with smallest singular value below `1e-6` times the largest. It also rejects empty
or disconnected graphs, duplicate camera/point observations, invalid indices,
nonfinite input, free points seen in fewer than two views, and cameras with fewer
than four observations. These checks do not prove global observability: remaining
local degeneracies are damped, not silently dropped.

The Python function accepts aligned C-contiguous float64 cameras, points and pixels,
int64 observation indices, and fixed-point indices. It copies inputs into owned Rust
buffers before releasing the GIL, returns a result dictionary, and does not mutate
NumPy inputs. Fixed points retain their exact original coordinates. C++ is not exposed
in this prototype.

## Deliberate prototype boundaries

- All cameras are adjustable, all residuals have the same scale; no mixed camera
  models, shared intrinsics, pose priors, or automatic component/gauge splitting.
- A dedicated affine solver leaves the existing pinhole solver untouched. The
  eight-dimensional f64 assembly is **not yet a shared generic Schur backend**.
  Extracting reusable blocks is a follow-up design decision for the PR split.
- Coordinates need not match the alternating solver: compare after affine gauge
  alignment and inspect reprojection/metric-frame quality. Outlier filtering can
  retain a different set of observations.
- Results below cover one real aerial capture and synthetic tests, not a diverse
  dataset or downstream house-model IoU. No claim of universal speedup is made.
- No `pixi run pr-check`, workspace-wide CI-feature tests, C++ tests, or MSRV test
  was run. This is an experimental branch, not a merge-ready combined PR.

## Build and reproduce

The measurements use Rust 1.99.1, Python 3.10.15, NumPy 2.2.6, and a release CPU wheel
on an Intel Xeon at 2.20 GHz. No CUDA is used. The wheel was installed in an isolated
`/tmp/kornia-affine-python` directory, leaving the normal Python installation alone.

```bash
maturin build --release --no-default-features -m kornia-py/Cargo.toml -o /tmp/kornia-affine-wheels
python -m pip install --no-deps --target /tmp/kornia-affine-python /tmp/kornia-affine-wheels/*.whl
OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/tmp/kornia-affine-python python scripts/benchmark_affine_ba.py \
  --oblique-root /path/to/oblique-ug --tracks /path/to/aerial_ba_raw_tracks.npz \
  --reconstruct weak-perspective --directions east north south west none \
  --output /tmp/affine-results.json
```

The local capture is order 19934329, originally five views. The fifth is unregistered;
after selecting finite points and registered views, both solvers receive 4 cameras,
9,288 points and 21,031 observations. Captures are not committed. The NPZ contains
`cam_idx`, `pt_idx`, `xy`, and scalar `n_views` before outlier filtering.

Timings are medians of three repetitions with one BLAS/Rayon/OMP thread and a warmup,
including wrapper copies and result conversion but excluding imports and initialization.
The first Rust call is reported separately. The JSON records the tested Rust source
revision and extension path. Its dirty flag reflects benchmark/docs files being prepared;
the compiled library code is committed at that revision.

## Measured results

The final repeat ran with no concurrent build. Raw repetitions and provenance are in
[the results JSON](benchmarks/affine-schur-aerial.json).

| First BA call | Iterations | Median time | Cauchy cost |
|---|---:|---:|---:|
| NumPy alternation | 300 | 4.436 s | 20641.452 |
| Rust Schur, 20 iterations | 20 | 0.200 s | 20537.318 |
| Rust Schur, converged | 64 | 0.614 s | 20530.180 |

At 20 iterations the Rust cost is already 0.50% lower
than the capped baseline, in 22.2× less time. At its stopping
criterion it takes 7.2× less time. Neither comparison claims
the baseline reached the same optimum. The first one-iteration Rust call, without
JIT compilation, took 14.8 ms; this
excludes Python imports. The converged point difference after affine alignment has
median 0.0314 and 95th percentile
0.0759 in the initial affine frame (scene RMS
radius 10.18); this is not ground-truth accuracy.

The complete aerial reconstruction function (initialization, three BA/filter rounds,
and metric-frame upgrade, excluding matching and house modeling) improves from
**6.43 s to 1.18 s (5.46×)**.
Median reprojection error changes from 0.765 to 0.762 px;
95th percentile from 2.296 to 2.281 px. The final outputs
contain 9,177 versus 9,241 points and 19,485 versus
19,587 observations. These residual distributions use different retained
sets, so they complement rather than replace the common-observation cost comparison.

## Validation

- `cargo test -p kornia-3d`: **334 passed, 1 ignored**; doctests **23 passed, 2 ignored**.
- Selected Python tests (`test_ba_affine.py`, `test_robotics.py`, `test_package.py`):
  **16 passed**, including independent finite-difference SciPy least squares on the
  same vector-norm Cauchy objective, input validation, coordinate scaling/translation,
  exact fixed-point preservation, camera-only refinement, and near-planar gauge rejection.
- Existing `oblique-ug/tests/test_affine.py` with `adjust_oblique` substituted:
  **16 passed**. This exercises the metric frame and registration pipeline.
- Rust Jacobians match central finite differences; a Schur step matches a full damped
  normal-system solve; synthetic perturbed geometry and outliers are covered.
- `cargo fmt --all -- --check`, skills structure check, version-pin check, and
  `bash scripts/release_rust.sh --check` pass.
- Unmodified strict Clippy on Rust 1.99 fails on existing `deprecated`,
  `chunks_exact_to_as_chunks`, `map_or_identity`, and generated `clone_on_copy`
  findings outside this change. The affected crates pass:

```bash
cargo clippy -p kornia-3d -p kornia-py --no-deps --no-default-features --all-targets -- \
  -D warnings -A deprecated -A clippy::chunks_exact_to_as_chunks \
  -A clippy::map_or_identity -A clippy::clone_on_copy
```

A standalone synthetic Criterion benchmark is included:
`cargo bench -p kornia-3d --bench bench_affine_ba -- --sample-size 10`.
It measured **436.78–441.07 ms** for 4 cameras, 10,000 points, 40,000 observations,
and a cap of 20 linearizations. This is a different workload from the aerial capture,
not a before/after comparison with an existing Rust affine solver.

## Suggested contribution split

1. Affine camera model and analytical Jacobians.
2. Float64 affine Schur LM, tests and benchmark; decide which assembly/LM pieces
   should be shared with the pinhole implementation before stabilizing its API.
3. Python bindings, reference validation and downstream integration.

The commits follow these boundaries. The final tooling/docs commit carries the real
benchmark and provisional changelog fragment. Issue links, separate PR descriptions,
final API review, fragment numbering and full CI gates belong to the later PR split.

## Prototype self-review

Verdict: comment — suitable for downstream experiments; split and complete PR gates
before proposing a merge. This is the deliberately combined branch requested by the
maintainer, not a proposed combined PR.

- Scope: camera, solver, bindings, validation and aerial adapter match the prototype
  request; no production default or existing pinhole path was changed. [inferred]
- Correctness: Jacobian finite differences, full-system/Schur step agreement, SciPy
  objective comparison, coordinate normalization and explicit anchor preservation
  were exercised by the passing tests listed above. [measured]
- Performance: owned buffers cross the GIL boundary intentionally. Normal-equation
  arrays are allocated once per solve and reused; the small camera matrix is copied
  for each Cholesky trial. The reported end-to-end timings include these costs.
  The solver is single-threaded and performs no I/O in its hot loops. [inferred]
- Upstream design follow-up: the existing pinhole solver's fixed six-parameter f32
  helpers cannot directly serve this eight-parameter f64 model. Extract reusable
  assembly and LM policy during the PR split rather than permanently maintaining
  two general solver implementations. [inferred]
- Release follow-up: MSRV, full CI-feature checks, diverse real scenes and downstream
  house-model quality remain unverified; the provisional changelog name must be
  replaced with actual PR numbers. [measured]

Reuse searches (ripgrep was unavailable on the host):

```text
grep -nE 'fn (ata_6x6|atb_6x3|invert_3x3)|pub fn bundle_adjust_schur' crates/kornia-3d/src/ba_schur.rs
969:fn ata_6x6_into(acc: &mut [f32; 36], j: &[f32; 12]) {
992:fn atb_6x3_into(acc: &mut [f32; 18], jp: &[f32; 12], jx: &[f32; 6]) {
1021:fn invert_3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
1335:pub fn bundle_adjust_schur(

grep -n 'pub(crate) fn c_slice' kornia-py/src/pyutils.rs
98:pub(crate) fn c_slice<'a, T: Element, D: Dimension>(

grep -nE 'pub struct BaObservation|pub pixel:|pub fn cholesky_3x3' crates/kornia-3d/src/ba.rs crates/kornia-algebra/src/linalg/cholesky.rs
crates/kornia-3d/src/ba.rs:42:pub struct BaObservation {
crates/kornia-3d/src/ba.rs:48:    pub pixel: [f32; 2],
crates/kornia-algebra/src/linalg/cholesky.rs:14:pub fn cholesky_3x3(mat: &Mat3F32) -> Option<Mat3F32> {
```

The implementation reuses the existing NumPy layout validator and algebra vector
operations, and uses nalgebra's existing f64 Cholesky rather than extending the
f32-only helper or writing another inverse.

Skill gaps: the small-PR/linked-issue rules apply at the later contribution stage;
this combined exploratory branch is explicitly requested. No policy files were
changed to make the prototype pass those future gates.
