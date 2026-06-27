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

## Summary

| Metric | Status |
|---|---|
| Detection speed vs C | 1.69× **slower** (baseline, no NEON) |
| Detection speed vs target (≥10× faster than C) | **Not met** — 16.9× speedup needed |
| Family construction (Tag36H11) | 262× faster than C |
| Family construction (TagCircle49H12) | Estimated >1000× faster than C |

The detection pipeline is the bottleneck. NEON optimization of the critical path
(gradient/thresholding/quad detection) is required to reach the ≥10× target.
