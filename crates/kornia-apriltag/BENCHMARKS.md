# kornia-apriltag Baseline Benchmarks

## Hardware

| Property | Value |
|---|---|
| Device | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| CPU | ARM Cortex-A78AE (CPU part 0xd42, implementer 0x41) |
| Cores | 6 |
| Max clock | 1728 MHz |
| Architecture | AArch64 (ARMv8) |

## Software

| Property | Value |
|---|---|
| Date | 2026-06-27 |
| Rust toolchain | rustc 1.93.0 (254b59607 2026-01-19) |
| OS | Linux 5.15.148-tegra (Ubuntu 22.04) |
| apriltag C lib | v3.2.0 (statically compiled from source via `apriltag-sys` crate, `APRILTAG_SYS_METHOD=raw,static`) |

## Commands

```bash
# Detection benchmark (main perf metric)
cd /home/nvidia/kornia-rs-apriltag-pose
APRILTAG_SYS_METHOD=raw,static \
APRILTAG_SRC=~/.cargo/registry/src/*/apriltag-sys-*/apriltag-src \
  cargo bench -p kornia-apriltag --bench bench_decoding

# Tag family construction benchmark
APRILTAG_SYS_METHOD=raw,static \
APRILTAG_SRC=~/.cargo/registry/src/*/apriltag-sys-*/apriltag-src \
  cargo bench -p kornia-apriltag --bench bench_tagfamily
```

Note: the system `libapriltag3` package (Ubuntu 22.04) is missing `image_u8_*` utility
symbols, so static compilation from the bundled apriltag source is required.

## bench_decoding — Full detection pipeline (799×533 image, Tag36H11)

| Benchmark | Median time | vs C ratio |
|---|---|---|
| kornia-apriltag | 13.341 ms | **1.69× slower** than C |
| apriltag-c | 7.915 ms | baseline |
| aprilgrid-rs | 5.847 ms | 1.35× faster than C |
| **Target** | **< 0.792 ms** | **≥10× faster than C** |

### Gap to target

- Current kornia: **13.341 ms**
- Target: **< 0.792 ms** (C median / 10)
- Required speedup: **16.9×** relative to today's Rust implementation

## bench_tagfamily — Detector initialization time (one-shot, not per-frame)

| Family | kornia median | C median | ratio |
|---|---|---|---|
| Tag16H5 | 89.3 µs | 46.7 µs | kornia 1.91× **slower** |
| Tag36H11 | 120.6 µs | 31.574 ms | kornia **262× faster** |
| Tag25H9 | 93.0 µs | 173.6 µs | kornia 1.87× faster |
| TagCircle21H7 | 91.6 µs | 129.7 µs | kornia 1.42× faster |
| TagCircle49H12 | 3.108 ms | > 3 s (timed out) | kornia **~1000× faster** (est.) |
| TagCustom48H12 | — | — | not collected (killed with above) |
| TagStandard41H12 | — | — | not collected |
| TagStandard52H13 | — | — | not collected |

**Note on TagCircle49H12/apriltag-c:** Criterion's 3-second warmup did not complete
after 35+ minutes of wall-clock time, indicating the C library regenerates the full
49 000-codeword hash table on every construction call. kornia uses a precomputed
static table and initializes in 3.1 ms. TagCustom48H12 and TagStandard families were
not reached before the benchmark was killed.

**Note on Tag36H11:** the C library also regenerates at runtime (31.6 ms per call);
kornia's precomputed table is 262× faster. This advantage is one-time at startup,
not per-frame.

## Post-NEON bench_decoding (commit 4ad18e2, 2026-06-27)

| Benchmark | Median time | vs C ratio | vs baseline |
|---|---|---|---|
| kornia-apriltag | 13.330 ms | **1.68× slower** than C | −11µs (−0.08%) |
| apriltag-c | 7.918 ms | baseline | — |
| aprilgrid-rs | 5.876 ms | 1.35× faster than C | — |

### Per-stage profile (steady-state, timing probes in release build)

| Stage | Time | % | NEON status |
|---|---|---|---|
| resize (decimation) | 150 µs | 1.1% | not attempted |
| adaptive_threshold pass 1 (tile min/max) | ~70 µs | 0.5% | **NEON (batch-of-4)** |
| adaptive_threshold pass 2 (classify pixels) | ~320 µs | 2.4% | scalar |
| find_connected_components | 2.0 ms | 15.0% | serial (union-find) |
| find_gradient_clusters | 7.1 ms | 53.4% | serial (HashMap) |
| fit_quads | 2.0 ms | 15.0% | not attempted |
| decode_tags | 1.8 ms | 13.5% | not attempted |

### Why the 10× target was not met

The NEON tile min/max optimization is correctly implemented and structurally
sound (gated behind `#[cfg(target_arch = "aarch64")]`, scalar fallback retained,
all 48 unit tests pass), but the stage it targets accounts for < 1% of pipeline
time. The benchmark result is within measurement noise of the baseline.

The dominant bottleneck is `find_gradient_clusters` (53% = 7.1 ms):

- Inner loop calls `uf.get_representative()` (union-find tree walk, serial)
- Then `clusters.entry(key).or_default()` (HashMap probe + possible realloc, serial)
- Neither is NEON-able; both have data-dependent control flow

`find_connected_components` (15%) is similarly union-find bound.

**To reach ≥10× vs C (< 0.79 ms) from 13.3 ms requires ≥16.9× speedup.
NEON realistically saves at most ~600 µs (threshold + partial quads/decode)
— still 12.7 ms short of target. Reaching the goal requires algorithmic changes:**

1. Replace `HashMap<(usize,usize), Vec<GradientInfo>>` with a pre-allocated
   slab/arena keyed on component-ID pairs → eliminate hashing overhead
2. Parallel union-find (split-merge) for connected components → 4–6× on cc
3. SIMD pixel classification in threshold pass 2 (saves ≈ 300 µs)
4. Possibly GPU offload for decimation + thresholding

## Summary

| Metric | Status |
|---|---|
| Detection speed vs C | 1.68× **slower** (after NEON pass-1) |
| Detection speed vs target (≥10× faster than C) | **Not met** — 16.9× speedup needed |
| Family construction (Tag36H11) | 262× faster than C |
| Family construction (TagCircle49H12) | Estimated >1000× faster than C |

NEON optimized the tile min/max (pass 1 of adaptive_threshold). The pipeline
bottleneck is find_gradient_clusters (HashMap-bound, 53% of time) which
requires algorithmic rather than SIMD optimization to close the 10× gap.
