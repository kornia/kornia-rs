# Colormap Implementation Design
Date: 2026-05-09

## Overview

Add `apply_colormap` to `kornia-rs`: a pure-Rust, SIMD-accelerated colormap function
covering all 21 OpenCV colormaps, exposed to Python via the existing PyO3 bindings.
The Python benchmark target is 10x faster than `cv2.applyColorMap`.

## Scope

- Input: single-channel `u8` grayscale `Image<u8, 1, CpuAllocator>`
- Output: 3-channel RGB `u8` `Image<u8, 3, CpuAllocator>`
- All 21 OpenCV colormaps (AUTUMN, BONE, JET, WINTER, RAINBOW, OCEAN, SUMMER, SPRING,
  COOL, HSV, PINK, HOT, PARULA, MAGMA, INFERNO, PLASMA, VIRIDIS, CIVIDIS, TWILIGHT,
  TURBO, DEEPGREEN)
- No OpenCV runtime dependency — LUT tables hardcoded from OpenCV source

## Data Layout

Each colormap stored as three separate compile-time `[u8; 256]` arrays:

```rust
struct ColormapLut {
    r: [u8; 256],
    g: [u8; 256],
    b: [u8; 256],
}
static JET_LUT: ColormapLut = ColormapLut { r: [...], g: [...], b: [...] };
```

Separate channels are required by the NEON kernel: `vqtbl4q_u8` operates on a contiguous
byte table, and `vst3q_u8` writes interleaved RGB — both need independent per-channel passes.

## Public API

### Low-level (in-place, zero-alloc)

```rust
pub fn apply_colormap(
    src: &Image<u8, 1, CpuAllocator>,
    dst: &mut Image<u8, 3, CpuAllocator>,
    colormap: ColormapType,
) -> Result<(), ImageError>
```

### Type-safe Image API (allocating, ergonomic)

```rust
pub trait ApplyColormap {
    fn apply_colormap(&self, colormap: ColormapType) -> Result<Rgb8, ImageError>;
}
impl ApplyColormap for Gray8 { ... }
```

### Python API

```python
# Function API
rgb = kornia.color.apply_colormap(gray_np, "jet")

# Image API
rgb_img = kornia.Image(gray_np).apply_colormap("turbo")
```

Colormap names are lowercase strings matching OpenCV naming convention.

## Kernel Architecture

Runtime dispatch via the existing `CpuFeatures` singleton (probed once via `OnceLock`):

```
apply_colormap()
    └── CpuFeatures::get()
        ├── [aarch64] has_neon  → apply_colormap_neon()    // 16 px/iter
        ├── [x86_64]  has_avx2  → apply_colormap_avx2()    // 32 px/iter
        └── fallback            → apply_colormap_scalar()   // 1 px/iter
```

### Scalar kernel

```rust
for (pixel, &idx) in dst.chunks_exact_mut(3).zip(src.iter()) {
    pixel[0] = lut.r[idx as usize];
    pixel[1] = lut.g[idx as usize];
    pixel[2] = lut.b[idx as usize];
}
```

### NEON kernel (aarch64, 16 px/iter)

Load the 256-byte channel table across 16 × `uint8x16_t` registers (4 groups of 4).
Each `vqtbl4q_u8` call covers one 64-byte chunk and returns 0 for out-of-range indices.
Four calls OR'd together cover all 256 entries for one channel. Three channel passes +
`vst3q_u8` completes one 16-pixel iteration:

```
// R channel (same structure repeated for G, B)
r0 = vqtbl4q_u8({R[0..64]},   idx)        // hits indices 0..63,   0 elsewhere
r1 = vqtbl4q_u8({R[64..128]},  idx - 64)  // hits indices 64..127, 0 elsewhere
r2 = vqtbl4q_u8({R[128..192]}, idx - 128)
r3 = vqtbl4q_u8({R[192..256]}, idx - 192)
r  = r0 | r1 | r2 | r3

vst3q_u8(dst_ptr, {r, g, b})              // one instruction, interleaved RGB write
```

### AVX2 kernel (x86_64, 32 px/iter)

`_mm_shuffle_epi8` operates on 128-bit lanes (16 bytes). Two 16-pixel passes are combined
into a 32-pixel iteration. Within each 16-pixel pass: split 256-byte table into two
128-byte halves, use high bit of index to select half, blend results:

```
// Per 16-pixel half (run twice, combined into 256-bit store)
lo_mask = idx & 0x7F
hi_flag = (idx >> 7) * 0xFF          // broadcast bit to all 8 positions
r_lo = _mm_shuffle_epi8(R_LUT[0..128],   lo_mask)
r_hi = _mm_shuffle_epi8(R_LUT[128..256], lo_mask)
r    = _mm_blendv_epi8(r_lo, r_hi, hi_flag)   // same for g, b

// Interleave r/g/b for both halves, store 96 bytes with _mm256_storeu_si256 x3
```

## Files

| File | Action | Purpose |
|------|--------|---------|
| `crates/kornia-imgproc/src/color/colormap.rs` | Create | LUT tables, enum, scalar + SIMD kernels, `ApplyColormap` trait |
| `crates/kornia-imgproc/src/color/mod.rs` | Modify | `pub mod colormap;` + re-exports |
| `kornia-py/src/color.rs` | Modify | `#[pyfunction] apply_colormap` + `PyImage` method |
| `kornia-py/src/lib.rs` | Modify | Register new pyfunction |
| `crates/kornia-imgproc/benches/bench_colormap.rs` | Create | Criterion: scalar / NEON / AVX2 / dispatch × 3 sizes |
| `crates/kornia-imgproc/Cargo.toml` | Modify | Add bench entry |
| `examples/bench_colormap.py` | Create | Python: kornia vs cv2, all 21 maps × 3 sizes |

## Benchmark Design

### Criterion (Rust)

Three image sizes: 640×480, 1920×1080, 3840×2160.
Four benchmark groups: `scalar`, `neon`, `avx2`, `dispatch`.
Each kernel called directly (not via dispatch) to isolate per-kernel performance.

### Python benchmark

```python
SIZES = [(640, 480), (1920, 1080), (3840, 2160)]
# 200 iterations each, report median, MP/s, and speedup vs cv2
```

Reports per-colormap speedup ratio and MP/s. Target: ≥10x on Jetson Orin (NEON).

## Success Criteria

- All 21 colormaps produce pixel-identical output to `cv2.applyColorMap` (±0 tolerance)
- Python benchmark shows ≥10x speedup over OpenCV on Jetson Orin at 1920×1080
- Criterion shows NEON kernel ≥8x faster than scalar
- Existing `kornia-imgproc` tests pass (no regressions)
