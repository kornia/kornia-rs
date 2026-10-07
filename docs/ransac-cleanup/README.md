# RANSAC architecture cleanup

This follow-up is based on PR #1167 at
`7e77446560d9b58c416e653b58f93a01afda64a6`. The use case is maintaining and
benchmarking robust fundamental/essential estimation for two-view vision
without having to understand pose recovery and SIMD implementation at once.

## Reference designs

[PoseLib's LO-RANSAC driver](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/ransac_impl.h)
is the primary structural reference. Its driver orchestrates candidate
selection, stopping and refinement through a small estimator interface.
[Fundamental-specific methods](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/estimators/relative_pose.cc#L468-L496)
keep geometry outside the driver. The useful lesson is responsibility and
lifecycle separation; we do not adopt PoseLib's numerical or acceptance policy.

[GC-RANSAC's scoring interface](https://github.com/danini/graph-cut-ransac/blob/799a645dbb58e5a122dfc01b493b8a8104572729/src/pygcransac/include/scoring_function.h)
and [settings](https://github.com/danini/graph-cut-ransac/blob/799a645dbb58e5a122dfc01b493b8a8104572729/src/pygcransac/include/settings.h)
show useful separation between evaluation, preemption and local optimization.
Its main driver is still a large header, so it is a policy reference rather
than a code-layout template. Neither project is vendored or copied here.

## This change

The former 4,131-line `pose/twoview.rs` had public pose APIs, model-estimation
loops, local optimization, degeneracy recovery and architecture-specific
scoring interleaved. The private `pose/twoview/` directory now separates:

| File | Responsibility |
| --- | --- |
| `mod.rs` | Public API, solver/refiner adapters, model selection, cheirality and triangulation pipeline |
| `robust.rs` | Dedicated F7/F8/E5/H4 estimation and historical stopping policies |
| `local_optimization.rs` | Fundamental LO+ schedule and dominant-plane recovery |
| `scoring.rs` | SoA scoring interface, mask lifecycle, F7 bounded/hybrid policy, SIMD/scalar kernels |
| `tests.rs` | Existing end-to-end and numerical tests plus scorer contract regression coverage |

The `ScoringPoints` view groups four validated coordinate slices without
allocation. Full scoring overwrites the complete mask. Candidate scoring
returns only strict improvements and ensures every returned winner has a
complete mask. Rejected candidates may leave partial scratch, which is never
consumed. Callers no longer decide when to clear masks, switch the F7 hybrid
strategy or rematerialize a winning root.

The generic fundamental estimator likewise has a small `mod.rs` for fitting
and trait adaptation, `kernels.rs` for AoS SIMD/scalar residual evaluation,
and `tests.rs`. Matrix arithmetic and backend dispatch are preserved.

The generic serial driver's normal, valid-SPRT and invalid-SPRT branches now
produce one evaluation outcome and share incumbent acceptance. Minimal and
LO candidates use the same strict acceptance/cap update. Final refit remains
a separate stage because it historically accepts tied scores. Regression
tests cover invalid-SPRT fallback and unordered (NaN) consensus scores.

## Compatibility contracts

These distinctions are existing behavior, not differences introduced here:

| Policy | Generic API | Dedicated two-view API |
| --- | --- | --- |
| Input representation | AoS `Match2d2d` | Pixel points, cached SoA coordinates |
| Threshold argument | Residual units (squared pixels for F) | Pixels, squared at entry |
| Inlier predicate | Strict `<` | Inclusive `<=` |
| Winner ordering | Maximum consensus score; first wins ties | Maximum count, then minimum inlier-error sum; first wins exact ties |
| Final inlier refit | Always considered when support exceeds minimal size; ties accepted | Optional `refit`; F uses LO+ then DEGENSAC |
| SPRT | Optional, seedable evaluation order | Absent |
| Homography stopping | Generic confidence cap | Historical confidence default plus 200-draw stagnation limit |
| Result | Optional model, mask, score, draw count | Mandatory model or error, mask, count, residual-sum score |

The public Rust and Python names, defaults, solver arithmetic, random draws,
root order, masks, stopping logic and threshold semantics remain unchanged.
The two SIMD data layouts remain distinct. This first change separates the
seams; it does **not** claim to have unified the two RANSAC drivers.

## Next migration: one internal driver

The desired structure is thin public wrappers around one internal lifecycle:

```text
public options + geometry preprocessing
                  |
  sample -> generate models -> evaluate -> accept -> update stop budget
                  |                            |
            estimator fit              optional local refinement
                  |
          final refinement -> materialize winner mask -> public result
```

The shared driver should own sample/candidate traversal, incumbent state and
termination, with explicit hooks for evaluation and refinement. Estimators
should own minimal/nonminimal fitting and model validity. Scoring policies
must express strict/inclusive threshold comparison and count-only versus
count/error ordering directly. LO+, DEGENSAC and SPRT should configure hooks
around the common loop rather than adding independent copies of it.

Before migrating a dedicated wrapper, compare it to a frozen binary of the
existing path for matrices, masks, counts, scores, failure messages and RNG
state/draw counts where observable. Include single-draw multi-root cases,
full-support roots, empty/invalid inputs, threshold equality, degenerate
geometry, local refinement, SPRT and custom consensus. Sample-level stopping
must finish the remaining roots; geometry wrappers retain their current
postprocessing and error behavior.

Unifying evaluation does not require discarding AoS/SoA SIMD specialization.
A shared scoring interface can dispatch to layout-specific loads while keeping
one definition of the arithmetic and score/mask contracts. Replacing numerical
kernels must be a separately measured step, since SIMD operation order can
alter threshold decisions even when formulas look equivalent.

This migration can preserve historical behavior through private policies.
Changing public defaults or harmonizing output semantics is a separate API
migration, requiring explicit accuracy evaluation and release notes.

## Validation

Local validation uses Rust 1.93.0 on Apple M1:

- Focused Clippy passes with warnings denied and all targets in `kornia-3d`
  and `kornia-calib`.
- 284 3D library tests pass, one is ignored, and the pre-existing
  `defaults_are_bit_identical_to_the_frozen_solver` digest mismatch is filtered.
  That exact failure was reproduced on unchanged PR #1167 with this compiler;
  PR #1167's locked-compiler CI passes on ARM64 and x86.
- 81 focused generic RANSAC tests pass after the final driver signature cleanup.
- 21 doctests pass, two are ignored. Nine fundamental Python tests pass against
  the built cleanup extension.
- Independent review verified the moved kernels, public reexports and driver
  policies against the frozen base, with no actionable findings.

The standalone [`audit.rs`](audit.rs) emits matrix/pose/point floating-point
bits, masks, counts, residual sums and errors. Compile the same source against
frozen before/after revisions, preserving both executables under `/tmp`, then
compare their stdout. It covers 288 configurations across F7/F8/E5/H4 (1,152
model results), plus 16 full pose-pipeline results. Inputs include invalid
lengths, minimal samples, odd SIMD tails, planar/nonplanar scenes, outliers,
three seeds, single-draw versus 128-draw caps, optional LO+/DEGENSAC, and both
fundamental and essential pose adapters. All 1,168 results match bit for bit; hashes are
recorded in [`equivalence.json`](equivalence.json).

To compile the audit, create a disposable Cargo package under `/tmp` with
`kornia-3d` and `kornia-algebra` path dependencies on the revision being checked.
Use the same lockfile/compiler and release profile (`lto = "thin"`,
`codegen-units = 1`) for both builds. If a target directory is reused between
checkouts, clean the changed path packages before rebuilding; preserve the
before binaries first. Never compare two copies of an accidentally cached
binary.

The real-data parity and timing audit uses the existing
`kornia-py/benchmarks/bench_fundamental_scoring.py` harness, with this cleanup's
base as `--before`. The native audit uses `bench_fundamental`, three alternating
rounds, 20 samples, 0.5 s warmup and 1 s measurement. Results are recorded
in [`public.json`](public.json), [`public-raw.json.gz`](public-raw.json.gz),
[`native.json`](native.json) and the six Criterion logs here. Benchmarking is
sequential, without concurrent builds. All 2,040 real cases return exactly the
same matrices, masks and failures; exposed generic draw counts also match.
The dedicated Python API does not expose draw counts.


## Timing results

Speedup is before time divided by after time; values below one are slower.
These are host-specific measurements, not claims of a speed improvement.
The real F7 corpus is within 0.5% of the frozen base. The native dedicated F7
cases are 1.9–3.7% slower, so the refactor is not universally performance-neutral.
Generic F7 is roughly 1% faster natively, while F8 native controls range from
0.8% to 3.1% slower. All measurements, including slower cases, are retained.

| Real F7 draw cap | Generic speedup | Dedicated speedup |
| --- | ---: | ---: |
| 64 | 1.001× | 0.995× |
| 256 | 1.001× | 0.998× |
| 1000 | 1.000× | 0.999× |
| 4096 | 1.000× | 0.999× |
| 16384 | 1.000× | 1.000× |

| Native benchmark | Speedup |
| --- | ---: |
| `fundamental_minimal/7point` | 1.001× |
| `fundamental_minimal/8point` | 0.999× |
| `fundamental_ransac_fixed_budget/7point/0.2` | 1.007× |
| `fundamental_ransac_fixed_budget/7point/0.5` | 1.009× |
| `fundamental_ransac_fixed_budget/7point/0.8` | 1.009× |
| `fundamental_ransac_fixed_budget/8point/0.2` | 0.986× |
| `fundamental_ransac_fixed_budget/8point/0.5` | 0.988× |
| `fundamental_ransac_fixed_budget/8point/0.8` | 0.988× |
| `fundamental_twoview_fixed_budget/7point/0.2` | 0.981× |
| `fundamental_twoview_fixed_budget/7point/0.5` | 0.969× |
| `fundamental_twoview_fixed_budget/7point/0.8` | 0.963× |
| `fundamental_twoview_fixed_budget/8point/0.2` | 0.992× |
| `fundamental_twoview_fixed_budget/8point/0.5` | 0.991× |
| `fundamental_twoview_fixed_budget/8point/0.8` | 0.969× |
