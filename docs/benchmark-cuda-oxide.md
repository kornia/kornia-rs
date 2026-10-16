# cuda-oxide kernels: parity and throughput

One row per kernel (and arch), plus a short section. Method:
[`docs/cuda-oxide.md`](cuda-oxide.md). Roadmap: #1135.

| Kernel | Arch | Byte-exact vs CPU | GPU 4K (ms) | CPU 4K (ms) | vs main's NVRTC (4K) |
|---|---|---|---:|---:|---:|
| `gray_from_rgb_u8` | sm_87 (Orin), PTX sm_75 | yes | 0.347 | 1.33-1.45 | 0.999 |

## System

| | |
|---|---|
| Date | 2026-10-14 |
| Machine | Jetson Orin Nano Engineering Reference Developer Kit Super (ga10b, sm_87) |
| L4T / CUDA / driver | R36.4.3 / 12.6.68 / 540.4.0 |
| Power mode | MAXN_SUPER (nvpmodel 2) |
| GPU clocks | **not locked** (DVFS 306-1020 MHz; no sudo for `jetson_clocks`) |
| rustc | 1.98.0 (kornia), nightly-2026-04-03 (kernels-oxide) |
| cuda-oxide / llc | `f819f23` / LLVM 21.1.8 |

## `gray_from_rgb_u8`

OpenCV Q14 (`(4899 R + 9617 G + 1868 B + 8192) >> 14`), 4 px/thread: three u32
loads, one u32 store, byte tail. Replaces main's NVRTC kernel of the same name.

- **Parity:** `gray_from_rgb_u8_bit_exact` (n = 1..8, 1023, 851, 640x480,
  1920x1080+3 with a sentinel past `npixels`; images 1x1, 1x7, 7x1, 1x4099,
  4099x1, 33x17, 255x3, 1025x1, 1919x1081, 4K) and
  `gray_from_rgb_u8_degenerate_npixels`. Changing one weight (4899 -> 4900)
  fails it at n = 1023.
- **Throughput** (`benches/bench_cuda_oxide.rs`, 3 sittings; GPU kernel time, CPU = 6
  threads, NEON): 4K GPU 0.3467-0.3474 ms (95.6 GB/s, 1020 MHz, spread <=4.5%)
  vs CPU 1.45-1.71 ms; 1080p GPU 0.091 ms vs CPU 0.55-0.59 ms. 320x240 is
  launch-bound and the governor sits at 306 MHz for some sittings: do not quote.
- **vs main's NVRTC kernel** (`orin-test-bench.sh --base origin/main
  --cycles 4`, binaries alternated): 4K min 0.3464 vs 0.3469 ms (0.999),
  median of the 4 cycle minima 0.3468 vs 0.3476 ms; spread <=1%. 1080p mins
  agree (0.0913 vs 0.0921 ms) but the clock swung 306-1020 MHz there.
- **Variants tried, not kept** (same sittings, byte-exact): 1 px/thread is
  2.19x slower at 4K (0.761 ms) and 2.1x at 1080p; 8 px/thread (three u64 or
  six u32 loads, all loads first) is 0.99x at 4K and 0.99-1.00x at 1080p,
  under the 10% bar. 4 px/thread already streams at ~96 GB/s.
