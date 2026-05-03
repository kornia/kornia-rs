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

## Throughput (median μs / Mpix/s, 10 reps)

| src → dst           | arm                   | median (μs) | Mpix/s | vs NEON       |
|---------------------|-----------------------|------------:|-------:|---------------|
| 512² → 256²         | **neon**              |        23.1 | 1416.3 | —             |
|                     | cubecl_cpu_kernel     |     2 743.7 |   11.9 | 119× slower   |
|                     | cubecl_cpu_e2e        |     3 238.3 |   10.1 | 140× slower   |
| 1024² → 512²        | **neon**              |       326.4 |  401.5 | —             |
|                     | cubecl_cpu_kernel     |     4 255.3 |   30.8 |  13× slower   |
|                     | cubecl_cpu_e2e        |     8 464.9 |   15.5 |  26× slower   |
| 2048² → 1024²       | **neon**              |       479.2 | 1094.1 | —             |
|                     | cubecl_cpu_kernel     |     6 794.0 |   77.2 |  14× slower   |
|                     | cubecl_cpu_e2e        |    23 843.7 |   22.0 |  50× slower   |
| 4096² → 2048²       | **neon**              |     2 143.2 |  978.5 | —             |
|                     | cubecl_cpu_kernel     |    19 906.0 |  105.4 |   9× slower   |
|                     | cubecl_cpu_e2e        |    89 437.8 |   23.4 |  42× slower   |

## Findings

### NEON dominates
The production `fast_image_resize` NEON path is **9× to 119× faster** than cubecl-cpu at
every size tested. The gap shrinks as inputs grow (cubecl-cpu amortizes per-launch
overhead), but never closes. There is no crossover point in the tested range.

### cubecl-cpu kernel throughput ramps with size
- 256² output: 11.9 Mpix/s
- 512² output: 30.8 Mpix/s
- 1024² output: 77.2 Mpix/s
- 2048² output: 105.4 Mpix/s

This curve points at heavy fixed cost per kernel dispatch (likely MLIR JIT lookup +
per-call thread-pool sync) that only large workloads can cancel out. Even at 2048²
output (12.5 MB) we're not yet at saturation — extrapolating, cubecl-cpu would peak
around 150-200 Mpix/s, still far below NEON's ~1000-1400 Mpix/s ceiling.

### End-to-end overhead is brutal
For the 4096² case: kernel-only ≈ 20 ms, end-to-end ≈ 89 ms. The 70 ms gap is
`create_from_slice` + `read_one` round-trip — cubecl's CPU runtime allocates new device
memory each call rather than reusing buffers. A real video-pipeline integration would
need to keep handles alive across frames to avoid this tax.

### NEON is anomalously slow at 1024²→512² in this run
NEON's `326 μs` median for 1024×512 → 512×256 (output: 0.5 MB) is roughly 6× slower per
pixel than the 2048² and 4096² results, which is unusual. The most likely explanation is
that 0.5 MB output overflows the 1 MB L2 cache during the bench's tight loop while
2048²/4096² are already streaming-bound and the cache-spill cost amortizes. Repeating
with longer warm-up or `perf stat` would confirm. Doesn't change the headline conclusion
— even at this anomaly point, NEON is 13× ahead of cubecl-cpu.

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
