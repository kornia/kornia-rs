//! CUDA kernel for Bayer demosaic (u8, bilinear, OpenCV-compatible).
//!
//! Mirrors `color/bayer/kernels.rs` bit-for-bit: rounded integer averages
//! (`avg2 = (a+b+1)>>1`, `avg4 = (a+b+c+d+2)>>2`) and replicate-border
//! (clamp-to-edge) addressing.
//!
//! One thread handles a 2×2 quad. The color phase of each quad position is
//! fixed by the [`BayerPattern`] and passed as four uniform cell codes, so the
//! per-position `switch` is warp-uniform — all four patterns share one
//! branch-free kernel body.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

pub use kornia_image::color_spaces::BayerPattern;

use super::{check_len, config_2d, get_kernel, CudaColorError, KernelCell};

static BAYER_SRC: &str = r#"
// Cell codes: 0 = R, 1 = G-on-R-row, 2 = G-on-B-row, 3 = B.
#define CELL_R  0
#define CELL_GR 1
#define CELL_GB 2
#define CELL_B  3

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return min(max(v, lo), hi);
}

__device__ __forceinline__ int at(
    const unsigned char* src, int r, int c, int rows, int cols)
{
    r = clampi(r, 0, rows - 1);
    c = clampi(c, 0, cols - 1);
    return __ldg(&src[r * cols + c]);
}

__device__ __forceinline__ int avg2(int a, int b) {
    return (a + b + 1) >> 1;
}

__device__ __forceinline__ int avg4(int a, int b, int c, int d) {
    return (a + b + c + d + 2) >> 2;
}

// Demosaic one pixel with a (warp-uniform) cell code.
__device__ __forceinline__ void demosaic_px(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int r, int c, int rows, int cols, int cell)
{
    int center = __ldg(&src[r * cols + c]);
    int red, green, blue;
    switch (cell) {
        case CELL_R: {
            green = avg4(at(src, r - 1, c, rows, cols), at(src, r + 1, c, rows, cols),
                         at(src, r, c - 1, rows, cols), at(src, r, c + 1, rows, cols));
            blue  = avg4(at(src, r - 1, c - 1, rows, cols), at(src, r - 1, c + 1, rows, cols),
                         at(src, r + 1, c - 1, rows, cols), at(src, r + 1, c + 1, rows, cols));
            red = center;
            break;
        }
        case CELL_B: {
            green = avg4(at(src, r - 1, c, rows, cols), at(src, r + 1, c, rows, cols),
                         at(src, r, c - 1, rows, cols), at(src, r, c + 1, rows, cols));
            red   = avg4(at(src, r - 1, c - 1, rows, cols), at(src, r - 1, c + 1, rows, cols),
                         at(src, r + 1, c - 1, rows, cols), at(src, r + 1, c + 1, rows, cols));
            blue = center;
            break;
        }
        case CELL_GR: {
            red   = avg2(at(src, r, c - 1, rows, cols), at(src, r, c + 1, rows, cols));
            blue  = avg2(at(src, r - 1, c, rows, cols), at(src, r + 1, c, rows, cols));
            green = center;
            break;
        }
        default: { // CELL_GB
            blue  = avg2(at(src, r, c - 1, rows, cols), at(src, r, c + 1, rows, cols));
            red   = avg2(at(src, r - 1, c, rows, cols), at(src, r + 1, c, rows, cols));
            green = center;
            break;
        }
    }
    int o = (r * cols + c) * 3;
    dst[o]     = (unsigned char)red;
    dst[o + 1] = (unsigned char)green;
    dst[o + 2] = (unsigned char)blue;
}

// One thread per 2×2 quad; cell00..cell11 are the pattern's phase table
// entries (uniform across the launch → no warp divergence).
extern "C" __global__ void rgb_from_bayer_u8(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int rows, int cols,
    int cell00, int cell01, int cell10, int cell11)
{
    int qx = blockIdx.x * blockDim.x + threadIdx.x;
    int qy = blockIdx.y * blockDim.y + threadIdx.y;
    int c = qx * 2;
    int r = qy * 2;
    if (r >= rows || c >= cols) return;

    demosaic_px(src, dst, r, c, rows, cols, cell00);
    if (c + 1 < cols) demosaic_px(src, dst, r, c + 1, rows, cols, cell01);
    if (r + 1 < rows) {
        demosaic_px(src, dst, r + 1, c, rows, cols, cell10);
        if (c + 1 < cols) demosaic_px(src, dst, r + 1, c + 1, rows, cols, cell11);
    }
}
"#;

static BAYER: KernelCell = KernelCell::new();

/// Phase table as kernel cell codes `[row&1][col&1]` — derived from the CPU
/// path's format-defining `phase_table` so the two can never drift
/// (0=R, 1=G-on-R-row, 2=G-on-B-row, 3=B, matching the kernel's CELL_*).
fn cell_codes(pattern: BayerPattern) -> [i32; 4] {
    use crate::color::bayer::kernels::{phase_table, Cell};
    let code = |c: Cell| match c {
        Cell::R => 0,
        Cell::GonRRow => 1,
        Cell::GonBRow => 2,
        Cell::B => 3,
    };
    let t = phase_table(pattern);
    [code(t[0][0]), code(t[0][1]), code(t[1][0]), code(t[1][1])]
}

/// Demosaic a Bayer mosaic device buffer (`rows*cols` u8) to interleaved RGB
/// (`rows*cols*3` u8). Bilinear, replicate borders — bit-exact vs the CPU path.
pub fn launch_rgb_from_bayer_u8(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    rows: usize,
    cols: usize,
    pattern: BayerPattern,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), rows * cols)?;
    check_len("dst", dst.len(), rows * cols * 3)?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let kernel = get_kernel(&BAYER, stream, BAYER_SRC, "rgb_from_bayer_u8")?;
    let [c00, c01, c10, c11] = cell_codes(pattern);
    let (rows_i, cols_i) = (rows as i32, cols as i32);
    let (qw, qh) = ((cols as u32).div_ceil(2), (rows as u32).div_ceil(2));
    let cfg = config_2d(qw, qh);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&rows_i)
        .arg(&cols_i)
        .arg(&c00)
        .arg(&c01)
        .arg(&c10)
        .arg(&c11)
        .launch_cfg(cfg)?;
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};

    #[test]
    fn bayer_demosaic_bit_exact_vs_cpu_all_patterns_and_odd_sizes() {
        let stream = default_stream();
        for (rows, cols) in [(48usize, 64usize), (23, 37), (1, 1), (3, 5)] {
            let mosaic = pattern_u8(rows * cols);
            for pattern in [
                BayerPattern::Rggb,
                BayerPattern::Bggr,
                BayerPattern::Grbg,
                BayerPattern::Gbrg,
            ] {
                let mut cpu = vec![0u8; rows * cols * 3];
                crate::color::bayer::kernels::rgb_from_bayer_scalar(
                    &mosaic, &mut cpu, rows, cols, pattern,
                );

                let d_src = stream.clone_htod(&mosaic).unwrap();
                let mut d_dst = stream.alloc_zeros::<u8>(rows * cols * 3).unwrap();
                launch_rgb_from_bayer_u8(&stream, &d_src, &mut d_dst, rows, cols, pattern).unwrap();
                let gpu: Vec<u8> = stream.clone_dtoh(&d_dst).unwrap();
                stream.synchronize().unwrap();
                assert_eq!(
                    gpu, cpu,
                    "bayer {pattern:?} {rows}x{cols} must be bit-exact"
                );
            }
        }
    }
}
