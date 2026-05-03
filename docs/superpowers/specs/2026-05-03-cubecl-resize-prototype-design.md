# cubecl resize prototype — design

**Status:** approved (brainstorm)
**Date:** 2026-05-03
**Branch:** `proto/cubecl`
**Owner:** edgarriba

## Goal

Build a standalone `kornia-cubecl` crate hosting a single cubecl-based bilinear resize
kernel for `u8` RGB downscale, and a Criterion benchmark that compares it against the
existing NEON path (`fast_image_resize`, exposed via `kornia_imgproc::resize::resize_fast_rgb`)
across a power-of-two size sweep on Jetson Orin.

The deliverable is a numbers-on-the-table answer to: "for u8 RGB bilinear 2× downscale,
where (if anywhere) does cubecl-cuda or cubecl-cpu beat the production NEON path?"

Note: the NEON baseline is the third-party `fast_image_resize` crate — not a hand-rolled
kornia kernel. This is the actual production path for u8 RGB resize in `kornia-imgproc`,
so beating it is a meaningful (and non-trivial) bar.

## Scope

In:
- bilinear interpolation only
- `u8` RGB (3-channel interleaved) only
- 2× downscale only (square inputs)
- two cubecl runtimes: `cubecl-cuda`, `cubecl-cpu`
- correctness vs `fast_image_resize` reference
- benchmark on 4 sizes: `512²→256²`, `1024²→512²`, `2048²→1024²`, `4096²→2048²`

Out (deliberately deferred):
- `f32` images, other interpolation modes (Nearest, Bicubic, Lanczos), anti-aliased downscale
- mono / RGBA / non-3-channel layouts
- upscale, arbitrary aspect-ratio scale, non-square inputs
- `cubecl-wgpu` backend
- unified-memory / zero-copy CUDA path (Tegra-specific optimization)
- multi-GPU, async pipelining, stream overlap
- Python bindings (`kornia-py`) integration
- promoting kernel into `kornia-imgproc` proper

## Architecture

New crate at `crates/kornia-cubecl/`:

```
crates/kornia-cubecl/
├── Cargo.toml                              # deps: cubecl, cubecl-cuda, cubecl-cpu,
│                                           #       kornia-image, kornia-tensor, thiserror
│                                           # dev-deps: criterion, rand,
│                                           #           kornia-imgproc (for NEON baseline)
├── src/
│   ├── lib.rs                              # re-exports + runtime traits
│   ├── runtime.rs                          # CudaRuntime / CpuRuntime selection helpers
│   └── resize/
│       ├── mod.rs                          # public resize_bilinear_u8_rgb<R: Runtime>
│       ├── kernel.rs                       # #[cube] kernel — runtime-agnostic
│       └── weights.rs                      # CPU-side precompute of (idx, weight×256) tables
├── benches/
│   └── bench_resize.rs                     # Criterion: NEON vs cubecl-cpu vs cubecl-cuda
└── tests/
    └── correctness.rs                      # ±1 LSB vs fast_image_resize on 4 sizes
```

Crate features:
- `cuda` (default) — pulls `cubecl-cuda`, exposes `CudaRuntime` re-export
- `cpu` (default) — pulls `cubecl-cpu`, exposes `CpuRuntime` re-export

Both default-on so `cargo build` does the obvious thing on a Jetson; CI machines without
CUDA can build with `--no-default-features --features cpu`.

## Components

### `runtime.rs`
Thin re-export layer that gives the rest of the crate a single import surface for
runtime types regardless of which features are enabled. Provides
`init_cuda() -> Result<ComputeClient<CudaRuntime>>` and `init_cpu() -> ComputeClient<CpuRuntime>`
helpers used by tests and benches.

### `resize/weights.rs`
Pure CPU code. Given `(src_w, src_h, dst_w, dst_h)`, computes two tables:

- `weights_x: Vec<(u32, u16)>` of length `dst_w` — for each output column,
  the integer source column index `idx` and the fixed-point fractional weight
  `w ∈ [0, 256)` such that the output sample is
  `(256 - w) * src[idx] + w * src[idx + 1]` (rounded, then `>> 8`).
- `weights_y: Vec<(u32, u16)>` of length `dst_h` — same, for rows.

Mirrors `fast_image_resize`'s precompute. Computed once per resize, uploaded to device.

### `resize/kernel.rs`
A single `#[cube] fn resize_bilinear_u8_rgb_kernel` parameterized over `R: Runtime`.
Launch geometry: one thread per output pixel (`dst_w × dst_h` threads, tiled into
2D workgroups — 16×16 default).

Per-thread work:
1. Read this thread's `(out_x, out_y)`.
2. Look up `(sx, wx) = weights_x[out_x]` and `(sy, wy) = weights_y[out_y]`.
3. Read 4 source RGB triplets from `src` at `(sx, sy)`, `(sx+1, sy)`, `(sx, sy+1)`, `(sx+1, sy+1)`.
4. Per channel: compute `top = (256 - wx) * a + wx * b`, `bot = (256 - wx) * c + wx * d`,
   then `out = ((256 - wy) * top + wy * bot + (1 << 15)) >> 16`. Cast to `u8`.
5. Write the RGB triplet to `dst[out_y, out_x]`.

This is the **same fixed-point math** `fast_image_resize` uses, which is what makes
the ±1 LSB tolerance achievable.

### `resize/mod.rs`
Public entry point:

```rust
pub fn resize_bilinear_u8_rgb<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle<R>,
    src_size: ImageSize,
    dst: &Handle<R>,
    dst_size: ImageSize,
) -> Result<(), ResizeError>;
```

The function is **dispatch-only** — caller owns buffer allocation. This is the boundary
that defines the "kernel-only" measurement in the benchmark.

## Data flow (end-to-end timing path)

```
host Vec<u8> ─H2D─▶ device buf ─kernel─▶ device buf ─D2H─▶ host Vec<u8>
                              ▲                                ▲
                              └─weights vecs H2D───────────────┘
```

The benchmark's "end-to-end" arm covers everything in this diagram. The "kernel-only"
arm pre-allocates and pre-uploads device buffers + weights outside the timed region,
and times only the kernel dispatch + sync.

## Benchmark structure

`benches/bench_resize.rs` — single Criterion group `resize_u8_rgb_2x_downscale` with
these arms per size in `{512²→256², 1024²→512², 2048²→1024², 4096²→2048²}`:

| Arm                  | Backend                              | Timed boundary                       |
|----------------------|--------------------------------------|--------------------------------------|
| `neon`               | `kornia_imgproc::resize_fast_rgb`    | full call                            |
| `cubecl_cuda_kernel` | cubecl-cuda                          | dispatch + sync (warm device buffers)|
| `cubecl_cuda_e2e`    | cubecl-cuda                          | H2D + dispatch + sync + D2H          |
| `cubecl_cpu_kernel`  | cubecl-cpu                           | dispatch + sync                      |
| `cubecl_cpu_e2e`     | cubecl-cpu                           | dispatch + sync (no real copy on CPU)|

`Throughput::Elements(dst_w * dst_h)` so Criterion reports pixels/sec.

End-to-end arms allocate fresh device buffers each call AND upload weights each call
— this matches realistic single-shot use where the dst size isn't known in advance.
Kernel-only arms reuse buffers AND weights across iterations (precomputed once before
the timed region).

## Correctness test

`tests/correctness.rs` — for each of the 4 sizes:

1. Generate deterministic random `Image<u8, 3>` with `StdRng::seed_from_u64(0xC0FFEE)`.
2. Run `resize_fast_rgb` → `reference: Vec<u8>`.
3. Run cubecl-cpu → `actual_cpu: Vec<u8>`. Per-channel assert `abs(diff) ≤ 1`;
   total mismatched-channel count ≤ `0.001 * dst_w * dst_h * 3`.
4. Run cubecl-cuda (skipped if `init_cuda()` fails) → same comparison.

If the cuda runtime is unavailable, the test prints a skip message and passes — Jetson
CI matrix doesn't always have working CUDA.

## Error handling

- `init_cuda()` returns `Result` — no driver, no device, etc. → caller decides (bench
  prints "skipping cuda — no device" and continues; test prints skip message and passes).
- Kernel dispatch errors propagate as cubecl's native error type, wrapped in a crate-local
  `ResizeError` enum (variants: `Cubecl(cubecl::Error)`, `BufferSize { expected, got }`,
  `ZeroDimension`).
- Validation in `resize_bilinear_u8_rgb`: assert `src_size`, `dst_size` non-zero;
  assert buffer byte-lengths equal `width * height * 3`.

## Open questions / decisions for implementation phase

These are noted but do not block this design. The implementation plan will resolve them:

- Exact cubecl version to pin in `Cargo.toml` (depends on what's stable at impl time).
- Whether the `#[cube]` kernel's per-pixel loads can use `u32`-aligned reads of packed
  RGB (avoiding 3 single-byte loads). May matter on cuda; almost certainly on cpu.
- Workgroup size — default to 16×16 but worth a quick parameter sweep in the bench
  if CUDA numbers look off.

## Success criteria

The prototype is "done" when:

1. `cargo test -p kornia-cubecl` passes (correctness gate).
2. `cargo bench -p kornia-cubecl` runs to completion on Jetson Orin and emits all 5 arms × 4 sizes = 20 measurements.
3. The bench output is summarized in a short `RESULTS.md` in the crate, listing
   pixel-throughput numbers per arm and the crossover size (if any) where cubecl-cuda
   beats neon for the kernel-only and end-to-end columns.

That summary — not a production-ready feature — is the deliverable that decides whether
to invest further in a cubecl-backed kornia GPU path.
