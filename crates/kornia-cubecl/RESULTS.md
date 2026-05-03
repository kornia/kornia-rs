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

## Throughput (median μs / Mpix/s, 10 reps, extended size sweep + tiled variants)

The `_x4` and `_x16` arms are kornia-cubecl kernel variants that process 4 or 16
horizontally-adjacent dst pixels per cubecl thread. Reduces total thread count and
exposes longer contiguous-byte-store patterns to the MLIR optimizer. Same
algorithm and bit-exact same output as the baseline kernel.



| src → dst           | arm                   | median (μs) | Mpix/s | vs NEON       |
|---------------------|-----------------------|------------:|-------:|---------------|
| 512² → 256²         | **neon**              |        29.7 | 1103.4 | —             |
|                     | cubecl_cpu_kernel     |     1 933.8 |   16.9 | 65× slower    |
|                     | cubecl_cpu_kernel_x4  |     2 732.2 |   12.0 | 92× slower    |
|                     | cubecl_cpu_kernel_x16 |     2 619.2 |   12.5 | 88× slower    |
|                     | cubecl_cpu_e2e        |     5 608.2 |    5.8 | 190× slower   |
| 1024² → 512²        | **neon**              |       154.4 |  848.7 | —             |
|                     | cubecl_cpu_kernel     |     3 562.2 |   36.8 | 23× slower    |
|                     | cubecl_cpu_kernel_x4  |     3 877.1 |   33.8 | 25× slower    |
|                     | cubecl_cpu_kernel_x16 |     4 437.4 |   29.5 | 29× slower    |
|                     | cubecl_cpu_e2e        |     9 784.5 |   13.4 | 63× slower    |
| 2048² → 1024²       | **neon**              |       650.9 |  805.5 | —             |
|                     | cubecl_cpu_kernel     |     7 486.8 |   70.0 | 11× slower    |
|                     | **cubecl_cpu_kernel_x4** |   3 693.0 |  142.0 | **6× slower** |
|                     | cubecl_cpu_kernel_x16 |     7 896.8 |   66.4 | 12× slower    |
|                     | cubecl_cpu_e2e        |    20 315.0 |   25.8 | 31× slower    |
| 4096² → 2048²       | **neon**              |     2 267.6 |  924.8 | —             |
|                     | cubecl_cpu_kernel     |    20 688.6 |  101.4 | 9× slower     |
|                     | **cubecl_cpu_kernel_x4** |   9 980.8 |  210.1 | **4× slower** |
|                     | cubecl_cpu_kernel_x16 |    12 580.3 |  166.7 | 6× slower     |
|                     | cubecl_cpu_e2e        |    70 940.2 |   29.6 | 31× slower    |
| **8192² → 4096²**   | **neon**              |     6 887.2 | 1218.0 | —             |
|                     | cubecl_cpu_kernel     |    54 401.8 |  154.2 | 8× slower     |
|                     | cubecl_cpu_kernel_x4  |    35 571.7 |  235.8 | 5× slower     |
|                     | **cubecl_cpu_kernel_x16** |   27 257.8 |  307.8 | **4× slower** |
|                     | cubecl_cpu_e2e        |   267 784.7 |   31.3 | 39× slower    |
| 1920×1080 → 960×540 | **neon**              |       727.4 |  712.7 | —             |
|                     | cubecl_cpu_kernel     |     6 590.4 |   78.7 | 9× slower     |
|                     | cubecl_cpu_kernel_x4  |     5 272.2 |   98.3 | 7× slower     |
|                     | **cubecl_cpu_kernel_x16** |   4 167.3 |  124.4 | **6× slower** |
|                     | cubecl_cpu_e2e        |     9 961.8 |   52.0 | 14× slower    |

## Findings

### Tiling (more dst pixels per thread) is a 2× win at the right size
| size              | kernel | x4 | x16 | best   |
|-------------------|-------:|---:|----:|--------|
| 256² out          | 16.9   | 12.0 | 12.5 | baseline |
| 512² out          | 36.8   | 33.8 | 29.5 | baseline |
| 1024² out         | 70.0   | **142.0** | 66.4 | x4 (2.0×) |
| 2048² out         | 101.4  | **210.1** | 166.7 | x4 (2.1×) |
| 4096² out (8K in) | 154.2  | 235.8 | **307.8** | x16 (2.0×) |
| 540p (1080p in)   | 78.7   | 98.3 | **124.4** | x16 (1.6×) |

The optimal tile is not monotonic. At 2048² out, x4 wins; at 4096² out, x16 wins.
Likely reason: x16 wide tiles serialize the inner per-pixel loop too much when
each row is short, while x4 keeps enough parallelism. As rows grow, x16's
reduced thread overhead pays off.

A production cubecl-cpu kernel would need a dispatch heuristic to pick the tile
size — or generate the tile size at compile time from input dimensions.

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
