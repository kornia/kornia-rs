# Scoring optimization, October 2026

This compares the scoring changes against PR #1167 at
`35eb3c36fa517a3444cbd04dc58314069593b048`. The minimal seven-point solver,
sampling, stopping probability, score predicates and model ordering are unchanged.

## Implementation

The dedicated two-view F7 scorer prepares SIMD constants once per candidate
and initially computes only count and residual-error sum. Losing roots avoid
clearing and writing an inlier mask. Strict winners materialize their mask.
When the incumbent supports at least 75% of matches, the existing masked
bounded scorer avoids the extra winner pass. This is a performance decision;
both paths use identical score order, pruning bounds and tie-breaking.

The reusable RANSAC scorer also prepares its SIMD constants once per root.
Const-generic kernels fuse strict threshold counting with residual computation,
retain the original 128-point pruning boundaries, and write masks only for
completed candidates. Residual-only calls compile without counting/pruning work.

## Measurements

Final tables and raw observations are recorded alongside this file. Measurements
use an Apple M1 and Rust 1.93.0, with the repository's release profile (ThinLTO,
one codegen unit). These are host-specific measurements.

The public API comparison uses 34 St Peter's Square pairs, seeds 0/1/2, draw caps
64/256/1000/4096/16384, confidence 0.999999, and a shared **0.5 px** cutoff.
That is `threshold=0.25` in the generic API and `ransac_threshold=0.5` in the
dedicated API. Each case has five alternating before/after timing rounds;
tables report the mean of per-case medians, including Python/FFI overhead.
The APIs retain their existing refinement settings: generic final inlier refit,
no iterative LO; dedicated `refit=false`.

All 2,040 cases compare returned matrices and masks exactly, including matching
failure messages. Generic draw counts are compared exactly too. The dedicated
Python API does not expose draw counts. There is no threshold tuning between
binaries and no changed geometric output in this audit.

The native Criterion comparison uses the checked-in `bench_fundamental` scenes
with 500 matches and known inlier fractions 20%, 50% and 80%. It reports medians
of three alternating before/after runs (20 samples, 0.5 s warm-up and 1 s
measurement per benchmark). Adaptive stopping remains enabled under the 1,000
draw cap, identically before and after. F8 and the unchanged minimal solvers
are controls.

## Reproduction

Build and preserve Python extension binaries from the baseline and optimized
revisions with the same compiler and release profile, then run:

```sh
python kornia-py/benchmarks/bench_fundamental_scoring.py \
  --before /tmp/pr1167-before.so --after /tmp/pr1167-after.so \
  --data-root /path/to/pydegensac/benchmarks/data \
  --json /tmp/pr1167-scoring.json
```

For native measurements, compile `bench_fundamental` on both revisions (copy
this PR's expanded benchmark source onto the baseline) and preserve both
binaries under `/tmp`. Alternate their invocations:

```sh
/tmp/pr1167-before-bench --bench \
  'fundamental_(minimal|ransac_fixed_budget|twoview_fixed_budget)' \
  --sample-size 20 --warm-up-time 0.5 --measurement-time 1
```

Use the same command for the after binary and reverse execution order in the
next round. JSON metadata records input and binary hashes.

## Validation

- Pre-commit and focused Clippy with warnings denied pass, including all targets
  in `kornia-3d` and `kornia-calib`.
- 281 3D library tests pass with one ignored, excluding the existing local
  `defaults_are_bit_identical_to_the_frozen_solver` failure. That exact digest
  failure was reproduced on the unchanged PR sources with Rust 1.93.0; the
  GitHub ARM64 suite passes with its locked compiler.
- 19 doctests pass, with two ignored.
- Nine fundamental Python tests pass against the built optimized extension.
- Direct NEON equivalence, SIMD lane/tail boundaries, strict threshold equality,
  NaN/infinity handling, pruning and masked/count-only score equivalence pass.
  AVX2 correctness is covered by runtime-gated tests on x86 CI; these local
  timing measurements exercise NEON.

## Public API results

| API / solver | Draw cap | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|---:|
| Reusable RANSAC / F7 | 64 | 0.1017 | 0.1011 | 1.006× |
| Reusable RANSAC / F7 | 256 | 0.3627 | 0.3594 | 1.009× |
| Reusable RANSAC / F7 | 1000 | 1.1188 | 1.1079 | 1.010× |
| Reusable RANSAC / F7 | 4096 | 2.6662 | 2.6328 | 1.013× |
| Reusable RANSAC / F7 | 16384 | 6.7346 | 6.6459 | 1.013× |
| Dedicated two-view / F7 | 64 | 0.1634 | 0.1047 | 1.560× |
| Dedicated two-view / F7 | 256 | 0.6655 | 0.3803 | 1.750× |
| Dedicated two-view / F7 | 1000 | 2.0019 | 1.1693 | 1.712× |
| Dedicated two-view / F7 | 4096 | 4.3281 | 2.7678 | 1.564× |
| Dedicated two-view / F7 | 16384 | 10.1243 | 6.9869 | 1.449× |

Dedicated F7 improves by **1.45–1.75×** on these inputs. Reusable F7 improves
by roughly 1%; its native synthetic gain is about 2%. F8 controls and every
raw timing are retained in `public.json` and `public-raw.json.gz`.

## Native results

| Benchmark | Before | After | Speedup |
|---|---:|---:|---:|
| fundamental_minimal/7point | 0.486 µs | 0.487 µs | 0.997× |
| fundamental_minimal/8point | 0.977 µs | 0.979 µs | 0.998× |
| fundamental_ransac_fixed_budget/7point/0.2 | 2322.900 µs | 2268.800 µs | 1.024× |
| fundamental_ransac_fixed_budget/7point/0.5 | 1677.700 µs | 1636.700 µs | 1.025× |
| fundamental_ransac_fixed_budget/7point/0.8 | 92.269 µs | 90.531 µs | 1.019× |
| fundamental_ransac_fixed_budget/8point/0.2 | 1982.400 µs | 1979.500 µs | 1.001× |
| fundamental_ransac_fixed_budget/8point/0.5 | 2010.900 µs | 2010.800 µs | 1.000× |
| fundamental_ransac_fixed_budget/8point/0.8 | 170.330 µs | 169.830 µs | 1.003× |
| fundamental_twoview_fixed_budget/7point/0.2 | 2867.500 µs | 2618.900 µs | 1.095× |
| fundamental_twoview_fixed_budget/7point/0.5 | 2058.100 µs | 1794.300 µs | 1.147× |
| fundamental_twoview_fixed_budget/7point/0.8 | 82.440 µs | 84.148 µs | 0.980× |
| fundamental_twoview_fixed_budget/8point/0.2 | 1837.600 µs | 1839.000 µs | 0.999× |
| fundamental_twoview_fixed_budget/8point/0.5 | 1911.100 µs | 1906.000 µs | 1.003× |
| fundamental_twoview_fixed_budget/8point/0.8 | 152.470 µs | 149.900 µs | 1.017× |

The 80%-inlier dedicated case is about 2% slower in these native runs; the
unchanged controls span roughly 0.3% slower to 1.7% faster. The earlier
count-only strategy was about 10% slower there, motivating the measured hybrid.
