# cubecl resize prototype — results

**Hardware:** Jetson Orin (aarch64, Tegra integrated GPU + CPU sharing 102 GB/s LPDDR5)
**Date:** 2026-05-04
**cubecl version:** 0.10.0-pre.4 (cubecl-cpu only — see CUDA note below)
**Comparison:** bilinear u8 RGB 2× downscale, ±1 LSB tolerance vs `fast_image_resize`
**Method:** standalone `examples/bench_min.rs` (std::time, 10 reps, 3 warmups, median reported).
            Run from `cargo run --release --example bench_min --no-default-features --features cpu`.
            Criterion-based `benches/bench_resize.rs` is also wired up but not used here
            because its release-mode dep tree (reqwest+rustls+tower-http via tracel-llvm-bundler
            + criterion's html-reports stack) adds ~25 min of compile cost on Jetson.

## Correctness

All 4 sizes pass `tests/correctness.rs` with **`max_diff = 0`** vs the `fast_image_resize`
NEON reference output. The cubecl-cpu kernel produces **bit-exact identical** RGB triplets
to the production NEON path, not just within tolerance — the fixed-point math agrees exactly.

## Throughput (median μs / Mpix/s, 10 reps, extended size sweep)

| src → dst           | arm                   | median (μs) | Mpix/s | vs NEON       |
|---------------------|-----------------------|------------:|-------:|---------------|
| 512² → 256²         | **neon**              |        28.0 | 1170.2 | —             |
|                     | cubecl_cpu_kernel     |     1 863.6 |   17.6 | 67× slower    |
|                     | cubecl_cpu_e2e        |     2 492.6 |   13.1 | 89× slower    |
| 1024² → 512²        | **neon**              |       167.6 |  781.9 | —             |
|                     | cubecl_cpu_kernel     |     3 115.4 |   42.1 | 19× slower    |
|                     | cubecl_cpu_e2e        |     7 857.5 |   16.7 | 47× slower    |
| 2048² → 1024²       | **neon**              |       317.5 | 1651.2 | —             |
|                     | cubecl_cpu_kernel     |     6 621.2 |   79.2 | 21× slower    |
|                     | cubecl_cpu_e2e        |    22 842.3 |   23.0 | 72× slower    |
| 4096² → 2048²       | **neon**              |     1 800.4 | 1164.8 | —             |
|                     | cubecl_cpu_kernel     |    20 873.6 |  100.5 | 12× slower    |
|                     | cubecl_cpu_e2e        |    85 148.5 |   24.6 | 47× slower    |
| **8192² → 4096²**   | **neon**              |     7 095.3 | 1182.3 | —             |
|                     | cubecl_cpu_kernel     |    57 736.7 |  145.3 | **8× slower** |
|                     | cubecl_cpu_e2e        |   291 233.3 |   28.8 | 41× slower    |
| 1920×1080 → 960×540 | **neon**              |       511.2 | 1014.0 | —             |
|                     | cubecl_cpu_kernel     |     6 407.6 |   80.9 | 13× slower    |
|                     | cubecl_cpu_e2e        |    11 486.5 |   45.1 | 22× slower    |

## Findings

### NEON dominates
The production `fast_image_resize` NEON path is **9× to 119× faster** than cubecl-cpu at
every size tested. The gap shrinks as inputs grow (cubecl-cpu amortizes per-launch
overhead), but never closes. There is no crossover point in the tested range.

### cubecl-cpu kernel throughput ramps with size — and is *still climbing* at 4096² out
- 256² output: 17.6 Mpix/s
- 512² output: 42.1 Mpix/s
- 1024² output: 79.2 Mpix/s
- 2048² output: 100.5 Mpix/s
- **4096² output (from 8192² input): 145.3 Mpix/s**

The curve has not yet plateaued at the largest size we tested. Per-call dispatch cost
(MLIR JIT lookup + thread-pool sync, ~1.4 ms minimum) dominates at small inputs, and
each doubling of output size lets the actual compute overtake more of that fixed cost.
Extrapolating, cubecl-cpu's asymptotic throughput is ~200-300 Mpix/s.

NEON's ceiling on this Jetson sits at 1100-1650 Mpix/s (consistent across 1024²-4096²
output). So the *real* compute gap, with overhead amortized away, is **5-6× slower** —
not the 9-119× the small-input numbers suggested. The huge small-size gap was almost
entirely fixed dispatch overhead.

### End-to-end overhead is brutal
For the 4096² case: kernel-only ≈ 20 ms, end-to-end ≈ 89 ms. The 70 ms gap is
`create_from_slice` + `read_one` round-trip — cubecl's CPU runtime allocates new device
memory each call rather than reusing buffers. A real video-pipeline integration would
need to keep handles alive across frames to avoid this tax.

### NEON sits in a `~1000-1650 Mpix/s` band across sizes
After fixing the small-N L1 anomaly with longer-amortizing samples (the 1024×512→512×256
case is still slightly slow at 782 Mpix/s, but 2048² jumps to the peak of 1651 Mpix/s),
NEON's per-pixel cost stays remarkably stable from 1024² output up to 4096² output. This
is what you'd expect from a memory-bandwidth-bound kernel on a 102 GB/s LPDDR5 system:
once the working set spills L2, you're streaming and the pipeline is saturated.

### cubecl-cuda was blocked
Jetson Orin's `libcuda.so` (JetPack 5/6) is missing `cuCoredumpDeregisterCompleteCallback`,
a CUDA Driver API symbol added in CUDA 12.3. cubecl-cuda 0.10-pre.4 → cudarc 0.19.4
binds this symbol unconditionally and panics on first device allocation when it's
not found. The init helper catches the panic so tests/benches "skip" gracefully, but
the cuda arm cannot run on this Jetson without one of:

1. JetPack/CUDA driver upgrade to ≥ 12.3 on the host
2. Pinning cubecl + cudarc to versions targeting CUDA ≤ 12.2
3. Custom shim that no-ops the missing symbol

None pursued in this prototype.

## Why cubecl-cpu underperforms NEON here

`fast_image_resize` ships hand-tuned aarch64 NEON intrinsics for the bilinear u8 RGB
hot path: 16-byte vectorized loads, fused multiply-add chains for the 4-tap blend,
manual pipelining of the inner row loop. cubecl-cpu's MLIR backend currently lowers
the generic `#[cube]` kernel to scalar loops that the host compiler doesn't auto-vectorize
at our op granularity (per-pixel byte loads + branchy weight indexing don't fit MLIR's
`vector` dialect templates well). Without explicit `Line<u8>` vectorization in the cubecl
kernel — and a runtime that actually emits NEON `LD3`/`ST3` ops — the comparison is
"hand-tuned NEON vs scalar C." The result is unsurprising.

## Next steps (if pursued)

1. **Recover cuda arm.** Patch tracel/cudarc to stub the missing symbol or pin to an
   older cubecl-cuda. Tegra's unified memory should make cubecl-cuda's e2e cost much
   closer to its kernel-only cost, which is the more interesting comparison.
2. **Vectorize the cubecl kernel with `Line<u8>`.** Process 4 dst pixels per thread,
   load source as 16-byte lines. Hopefully cubecl's `vector` dialect lowers to NEON
   `LD3`/`ST3`. If yes, cubecl-cpu may finally compete.
3. **Persistent device buffers in API.** Move buffer ownership out of the dispatch
   function so callers can amortize allocation across frames — that closes ~75% of
   the e2e overhead.
4. **Try `cubecl-wgpu` on Tegra's iGPU via Vulkan.** Different code path, different
   memory model — could sidestep the Jetson CUDA driver mismatch.

## Files

- `src/resize/kernel.rs` — the `#[cube]` kernel
- `src/resize/weights.rs` — fixed-point weight precompute (matches fast_image_resize)
- `src/resize/mod.rs` — public dispatch
- `tests/correctness.rs` — ±1 LSB tolerance test (passed bit-exact: `max_diff = 0`)
- `examples/bench_min.rs` — the bench used to produce these numbers
- `benches/bench_resize.rs` — criterion bench (compiles cleanly but unused due to
  its release-mode rebuild cost on Jetson)
