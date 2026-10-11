# cuda-oxide vs NVRTC: parity report

Every kernel shipped from `kernels-oxide/` gets a section here: byte-exactness
evidence and an A/B against the NVRTC kernel it replaces. Method and checklist:
[`docs/cuda-oxide.md`](cuda-oxide.md). Roadmap: #1135, Phase 2.

| Kernel | Byte-exact vs CPU and NVRTC | oxide / NVRTC (4K, clean clocks) | Status |
|---|---|---|---|
| `gray_from_rgb_u8` | yes (`assert_eq!`, 9 sizes) | **0.998** | shipped behind `cuda-oxide` |
| bilinear resize f32 C3 | yes (2x down/up + 4 odd sizes) | within ±1.7% on 5 cases | spike only, not upstreamed |

## System

| | |
|---|---|
| Date | 2026-10-11 |
| Machine | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| GPU | ga10b, sm_87 |
| L4T | R36.4.3 |
| CUDA / driver | 12.6.68 / 540.4.0 |
| Power mode | MAXN_SUPER (nvpmodel 2) |
| GPU clocks | **not locked** (DVFS 306-1020 MHz; `jetson_clocks` needs sudo, unavailable). The bench records the clock after every round. |
| rustc (kornia) | 1.98.0 |
| rustc (kernel crate) | 1.96.0-nightly (2026-04-02), pinned `nightly-2026-04-03` |
| LLVM `llc` | 21.1.8 |
| cuda-oxide | `f819f23fe035ec1c24b8ec06e738b62a9f377140` |
| PTX | `.target sm_75`, `.version 6.3`, JIT-compiled by the driver |

## `gray_from_rgb_u8`

Port of `GRAY_FROM_RGB_U8_SRC` (`cuda/color/gray.rs`): OpenCV Q14 integer
weights, 4 px/thread via three u32 loads and one u32 store, byte tail for the
last partial quad. Same parameters, same `launch_map` geometry (256-thread 1-D
blocks), same `PREFER_L1` carveout as the NVRTC kernel.

### Correctness

```
test cuda::color::gray::tests::gray_from_rgb_u8_oxide_bit_exact ... ok
test cuda::color::gray::tests::gray_from_rgb_u8_bit_exact_vs_cpu ... ok
test cuda::oxide::tests::ptx_gate_target_is_sm75 ... ok
test cuda::oxide::tests::ptx_gate_has_every_entry ... ok
test cuda::oxide::tests::ptx_gate_no_fma ... ok
test cuda::oxide::tests::ptx_gate_no_generic_memory_ops ... ok
test cuda::oxide::tests::corrupt_ptx_errors_instead_of_panicking ... ok
```

`gray_from_rgb_u8_oxide_bit_exact` asserts oxide == CPU, NVRTC == CPU and
default launcher == oxide for n = 1, 3, 4, 5, 7, 8, 1023, 851, 1920·1080+3 (every
`n mod 4` tail), and that neither kernel writes past `npixels` into a
sentinel-filled oversized `dst`. It calls `ensure_loaded` first, so it cannot
pass by silently comparing NVRTC with itself. Mutation check: changing one Q14
weight in the committed PTX (4899 -> 4900) fails it at n = 1023.

### Throughput

`examples/bench_cuda_oxide.rs`: owned stream, 5 cycles alternating arm order,
each arm 5 rounds × 100 launches per cycle, events + sync per round, min round
per cycle. GB/s counts 3 B read + 1 B written per pixel. Three separate sittings:

| Size | Sitting | NVRTC min (ms) | oxide min (ms) | oxide / NVRTC | Spread (N / o) | GB/s (N / o) | GPU MHz seen |
|---|---|---:|---:|---:|---:|---:|---|
| 3840x2160 | 1 | 0.3472 | 0.3465 | **0.998** | 0.6% / 1.4% | 95.5 / 95.8 | 1020 |
| 3840x2160 | 2 | 0.3476 | 0.3471 | **0.998** | 0.2% / 0.1% | 95.4 / 95.6 | 1020 |
| 3840x2160 | 3 | 0.3479 | 0.3471 | **0.998** | 0.3% / 0.1% | 95.4 / 95.6 | 1020 |
| 1920x1080 | 1 | 0.0913 | 0.0911 | 0.998 | 55% / 56% | 90.9 / 91.0 | 714-1020 |
| 1920x1080 | 2 | 0.0911 | 0.0910 | 0.998 | 48% / 47% | 91.0 / 91.2 | 306-1020 |
| 1920x1080 | 3 | 0.0918 | 0.0914 | 0.996 | 47% / 48% | 90.4 / 90.7 | 714-1020 |
| 320x240 | 1 | 0.0155 | 0.0154 | 0.994 | 11% / 8% | 19.8 / 19.9 | 306 |
| 320x240 | 2 | 0.0081 | 0.0076 | 0.940 | 96% / 103% | 37.9 / 40.3 | 306-714 |
| 320x240 | 3 | 0.0082 | 0.0077 | 0.939 | 95% / 102% | 37.5 / 39.9 | 306-714 |

**Read the 4K rows.** The clock sat at 1020 MHz for every round and the cycle
spread is under 1.5%: oxide is at parity (0.2% faster, inside noise). At 1080p
the minima agree to 0.4% but the spread is DVFS ramping mid-run. At 320x240 the
kernel is launch-bound and too light to ramp the governor (306 MHz in sitting 1),
so those rows measure the clock, not the kernel; do not quote them. All of this
should be re-run with `sudo jetson_clocks` before a final number is cited.

### Instruction mix

From the PTX (nvcc `--fmad=false -arch=compute_87` stands in for NVRTC's
identical options); registers from `ptxas -v --gpu-name sm_87 -O3`.

| | NVRTC | cuda-oxide |
|---|---:|---:|
| instructions (approx) | 220 | 105 |
| `ld.global.nc` (`__ldg`) | 18 | 0 |
| `ld.global` | 0 | 6 |
| `fma` | 0 | 0 |
| registers | 27 | 18 |

The NVRTC count is inflated by nvcc unrolling the byte-tail loop 4x; the hot
4-px path is three word loads and one word store in both. The oxide kernel has
no read-only-cache loads (this cuda-oxide cannot express `__ldg`) and still
matches, consistent with the resize finding that `__ldg` is a no-op on sm_87.

### Target and fallback

The PTX targets `sm_75`; the kernel body is byte-identical to an `sm_87` build
(only `.version`/`.target` differ), and the Orin JITs it with the numbers above.
If the driver rejects the module, the launcher uses the NVRTC kernel.

## Bilinear resize (spike, 2026-09-10)

Branch `spike/cuda-oxide-resize` @ 88541a7, report
`docs/benchmark-cuda-oxide-resize.md` there. Same board, MAXN_SUPER,
`.target sm_87`. Different harness from the one above (wall clock, 200
iterations after 50 warmup, queue-all-then-sync), so do not compare its numbers
with the gray table.

| case (src->dst) | NVRTC `__ldg` | NVRTC no `__ldg` | oxide raw ptr | oxide slices |
|---|---:|---:|---:|---:|
| 1024x1024 -> 512x512 | 0.175 | 0.167 | 0.166 | 0.167 |
| 512x512 -> 1024x1024 | 0.197 | 0.198 | 0.199 | 0.221 |
| 1920x1080 -> 960x540 | 0.323 | 0.323 | 0.323 | 0.325 |
| 1920x1080 -> 3840x2160 | 1.526 | 1.530 | 1.536 | 1.720 |
| 3840x2160 -> 1920x1080 | 1.278 | 1.276 | 1.300 | 1.278 |

(ms/iter.) Byte-exact over 2x down, 2x up and four odd sizes. Two findings that
became porting rules: raw pointers carry no `noalias` (interleaved stores cost
2x on upscale until loads were batched before stores), and slices are slower
than raw pointers (generic loads, bounds branches; 12-13% on upscale).
