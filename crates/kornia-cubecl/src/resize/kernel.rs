//! Bilinear u8 RGB resize kernel.
//!
//! One thread per output pixel. Each thread reads 4 source RGB triplets and
//! computes the fixed-point bilinear blend matching `fast_image_resize`'s
//! output to within ±1 LSB.
//!
//! Buffers are `Array<u8>` directly (cubecl 0.10-pre.4 supports u8 as a first-class
//! primitive), avoiding the byte-pack/race-condition complications a u32-packed
//! layout would introduce.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn resize_bilinear_u8_rgb_kernel(
    src: &Array<u8>,
    dst: &mut Array<u8>,
    weights_x_idx: &Array<u32>,
    weights_x_w: &Array<u32>,
    weights_y_idx: &Array<u32>,
    weights_y_w: &Array<u32>,
    #[comptime] src_w: u32,
    #[comptime] dst_w: u32,
    #[comptime] dst_h: u32,
) {
    let out_x = ABSOLUTE_POS_X;
    let out_y = ABSOLUTE_POS_Y;
    if out_x >= dst_w || out_y >= dst_h {
        terminate!();
    }

    let sx = weights_x_idx[usize::cast_from(out_x)];
    let wx = weights_x_w[usize::cast_from(out_x)];
    let sy = weights_y_idx[usize::cast_from(out_y)];
    let wy = weights_y_w[usize::cast_from(out_y)];

    let row_top = sy * src_w * 3u32;
    let row_bot = (sy + 1u32) * src_w * 3u32;
    let off_l = sx * 3u32;
    let off_r = (sx + 1u32) * 3u32;

    let inv_wx = 256u32 - wx;
    let inv_wy = 256u32 - wy;

    let dst_off = (out_y * dst_w + out_x) * 3u32;

    #[unroll]
    for ch in 0u32..3u32 {
        let tl = u32::cast_from(src[usize::cast_from(row_top + off_l + ch)]);
        let tr = u32::cast_from(src[usize::cast_from(row_top + off_r + ch)]);
        let bl = u32::cast_from(src[usize::cast_from(row_bot + off_l + ch)]);
        let br = u32::cast_from(src[usize::cast_from(row_bot + off_r + ch)]);

        let top = inv_wx * tl + wx * tr;
        let bot = inv_wx * bl + wx * br;
        // Round half up via +(1<<15), then >>16 collapses both wx and wy normalization.
        let val = (inv_wy * top + wy * bot + (1u32 << 15)) >> 16u32;

        dst[usize::cast_from(dst_off + ch)] = u8::cast_from(val);
    }
}

/// N-pixels-per-thread variant. Each thread writes N horizontally-adjacent dst
/// pixels (N×3 contiguous bytes) — reduces total thread count N×, exposing a
/// pattern of contiguous byte stores that cubecl-cpu's MLIR backend can
/// (hopefully) fuse into wider stores.
///
/// Launch geometry: dst_w must be divisible by N for the kernel to cover the
/// rightmost column.
#[cube(launch_unchecked)]
pub fn resize_bilinear_u8_rgb_kernel_x4(
    src: &Array<u8>,
    dst: &mut Array<u8>,
    weights_x_idx: &Array<u32>,
    weights_x_w: &Array<u32>,
    weights_y_idx: &Array<u32>,
    weights_y_w: &Array<u32>,
    #[comptime] src_w: u32,
    #[comptime] dst_w: u32,
    #[comptime] dst_h: u32,
) {
    let tile_x = ABSOLUTE_POS_X; // tile index along x; each tile = 4 dst pixels
    let out_y = ABSOLUTE_POS_Y;
    if tile_x * 4u32 >= dst_w || out_y >= dst_h {
        terminate!();
    }

    let sy = weights_y_idx[usize::cast_from(out_y)];
    let wy = weights_y_w[usize::cast_from(out_y)];
    let inv_wy = 256u32 - wy;

    let row_top = sy * src_w * 3u32;
    let row_bot = (sy + 1u32) * src_w * 3u32;

    let dst_row = out_y * dst_w * 3u32;
    let base_x = tile_x * 4u32;

    // Process 4 horizontal output pixels.
    #[unroll]
    for px in 0u32..4u32 {
        let out_x = base_x + px;
        let sx = weights_x_idx[usize::cast_from(out_x)];
        let wx = weights_x_w[usize::cast_from(out_x)];
        let inv_wx = 256u32 - wx;
        let off_l = sx * 3u32;
        let off_r = (sx + 1u32) * 3u32;
        let dst_off = dst_row + out_x * 3u32;

        #[unroll]
        for ch in 0u32..3u32 {
            let tl = u32::cast_from(src[usize::cast_from(row_top + off_l + ch)]);
            let tr = u32::cast_from(src[usize::cast_from(row_top + off_r + ch)]);
            let bl = u32::cast_from(src[usize::cast_from(row_bot + off_l + ch)]);
            let br = u32::cast_from(src[usize::cast_from(row_bot + off_r + ch)]);

            let top = inv_wx * tl + wx * tr;
            let bot = inv_wx * bl + wx * br;
            let val = (inv_wy * top + wy * bot + (1u32 << 15)) >> 16u32;

            dst[usize::cast_from(dst_off + ch)] = u8::cast_from(val);
        }
    }
}

/// Composable primitive: sample one bilinear-interpolated u8 RGB pixel.
/// Returns `(r, g, b)` as a tuple of u32 (each in [0,255]) so callers can
/// do further compute without re-quantizing through u8.
///
/// `out_x`, `out_y` are the dst coordinates; weight tables are looked up here.
/// This is the inner-loop primitive used by both the standalone resize kernel
/// and any fused pipeline that wants to do "resize + something_else" in one pass.
#[cube]
pub fn sample_bilinear_u8_rgb_pixel(
    src: &Array<u8>,
    weights_x_idx: &Array<u32>,
    weights_x_w: &Array<u32>,
    weights_y_idx: &Array<u32>,
    weights_y_w: &Array<u32>,
    src_w: u32,
    out_x: u32,
    out_y: u32,
) -> (u32, u32, u32) {
    let sx = weights_x_idx[usize::cast_from(out_x)];
    let wx = weights_x_w[usize::cast_from(out_x)];
    let sy = weights_y_idx[usize::cast_from(out_y)];
    let wy = weights_y_w[usize::cast_from(out_y)];

    let row_top = sy * src_w * 3u32;
    let row_bot = (sy + 1u32) * src_w * 3u32;
    let off_l = sx * 3u32;
    let off_r = (sx + 1u32) * 3u32;

    let inv_wx = 256u32 - wx;
    let inv_wy = 256u32 - wy;

    let mut out_r = 0u32;
    let mut out_g = 0u32;
    let mut out_b = 0u32;

    #[unroll]
    for ch in 0u32..3u32 {
        let tl = u32::cast_from(src[usize::cast_from(row_top + off_l + ch)]);
        let tr = u32::cast_from(src[usize::cast_from(row_top + off_r + ch)]);
        let bl = u32::cast_from(src[usize::cast_from(row_bot + off_l + ch)]);
        let br = u32::cast_from(src[usize::cast_from(row_bot + off_r + ch)]);
        let top = inv_wx * tl + wx * tr;
        let bot = inv_wx * bl + wx * br;
        let val = (inv_wy * top + wy * bot + (1u32 << 15)) >> 16u32;
        if ch == 0u32 { out_r = val; }
        if ch == 1u32 { out_g = val; }
        if ch == 2u32 { out_b = val; }
    }
    (out_r, out_g, out_b)
}

/// Composable primitive: convert one RGB triple to luma using the
/// ITU-R BT.601 coefficients (matches OpenCV cvtColor's RGB→GRAY).
/// Inputs are u32 in [0,255] (matches the output of `sample_bilinear_u8_rgb_pixel`).
/// Output is u8 in [0,255].
#[cube]
pub fn rgb_to_gray_u8(r: u32, g: u32, b: u32) -> u32 {
    // OpenCV: gray = 0.299*R + 0.587*G + 0.114*B
    // Fixed-point: (R*77 + G*150 + B*29 + 128) >> 8
    (r * 77u32 + g * 150u32 + b * 29u32 + 128u32) >> 8u32
}

/// Composable primitive: normalize a u8 value to f32 in a target range.
/// `gray` in [0,255], output = (gray - mean) / std.
#[cube]
pub fn normalize_u8_to_f32(gray: u32, mean: f32, inv_std: f32) -> f32 {
    let g = f32::cast_from(gray);
    (g - mean) * inv_std
}

/// FUSED kernel: resize bilinear u8 RGB + RGB→gray + normalize to f32, all in one
/// pass. Avoids two intermediate DRAM round-trips (RGB-resized buffer + gray buffer)
/// vs calling the three ops separately.
#[cube(launch_unchecked)]
pub fn resize_to_gray_normalize_kernel(
    src: &Array<u8>,
    dst: &mut Array<f32>,
    weights_x_idx: &Array<u32>,
    weights_x_w: &Array<u32>,
    weights_y_idx: &Array<u32>,
    weights_y_w: &Array<u32>,
    #[comptime] src_w: u32,
    #[comptime] dst_w: u32,
    #[comptime] dst_h: u32,
    mean: f32,
    inv_std: f32,
) {
    let out_x = ABSOLUTE_POS_X;
    let out_y = ABSOLUTE_POS_Y;
    if out_x >= dst_w || out_y >= dst_h {
        terminate!();
    }

    let (r, g, b) = sample_bilinear_u8_rgb_pixel(
        src, weights_x_idx, weights_x_w, weights_y_idx, weights_y_w,
        src_w, out_x, out_y,
    );
    let gray = rgb_to_gray_u8(r, g, b);
    let norm = normalize_u8_to_f32(gray, mean, inv_std);

    let dst_off = out_y * dst_w + out_x;
    dst[usize::cast_from(dst_off)] = norm;
}

/// Standalone: RGB → gray (no resize). Operates pixel-by-pixel on an interleaved
/// u8 RGB buffer, writes a u8 grayscale buffer of length width*height.
#[cube(launch_unchecked)]
pub fn rgb_to_gray_kernel(
    src: &Array<u8>,
    dst: &mut Array<u8>,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }
    let src_off = (y * width + x) * 3u32;
    let r = u32::cast_from(src[usize::cast_from(src_off)]);
    let g = u32::cast_from(src[usize::cast_from(src_off + 1u32)]);
    let b = u32::cast_from(src[usize::cast_from(src_off + 2u32)]);
    let gray = rgb_to_gray_u8(r, g, b);
    let dst_off = y * width + x;
    dst[usize::cast_from(dst_off)] = u8::cast_from(gray);
}

/// Standalone: normalize u8 → f32 with given mean and inv_std.
#[cube(launch_unchecked)]
pub fn normalize_u8_to_f32_kernel(
    src: &Array<u8>,
    dst: &mut Array<f32>,
    #[comptime] count: u32,
    mean: f32,
    inv_std: f32,
) {
    let i = ABSOLUTE_POS_X;
    if i >= count {
        terminate!();
    }
    let g = u32::cast_from(src[usize::cast_from(i)]);
    dst[usize::cast_from(i)] = normalize_u8_to_f32(g, mean, inv_std);
}

/// 16-pixels-per-thread variant. Same idea as x4 but pushed harder. dst_w must
/// be divisible by 16.
#[cube(launch_unchecked)]
pub fn resize_bilinear_u8_rgb_kernel_x16(
    src: &Array<u8>,
    dst: &mut Array<u8>,
    weights_x_idx: &Array<u32>,
    weights_x_w: &Array<u32>,
    weights_y_idx: &Array<u32>,
    weights_y_w: &Array<u32>,
    #[comptime] src_w: u32,
    #[comptime] dst_w: u32,
    #[comptime] dst_h: u32,
) {
    let tile_x = ABSOLUTE_POS_X;
    let out_y = ABSOLUTE_POS_Y;
    if tile_x * 16u32 >= dst_w || out_y >= dst_h {
        terminate!();
    }

    let sy = weights_y_idx[usize::cast_from(out_y)];
    let wy = weights_y_w[usize::cast_from(out_y)];
    let inv_wy = 256u32 - wy;

    let row_top = sy * src_w * 3u32;
    let row_bot = (sy + 1u32) * src_w * 3u32;

    let dst_row = out_y * dst_w * 3u32;
    let base_x = tile_x * 16u32;

    #[unroll]
    for px in 0u32..16u32 {
        let out_x = base_x + px;
        let sx = weights_x_idx[usize::cast_from(out_x)];
        let wx = weights_x_w[usize::cast_from(out_x)];
        let inv_wx = 256u32 - wx;
        let off_l = sx * 3u32;
        let off_r = (sx + 1u32) * 3u32;
        let dst_off = dst_row + out_x * 3u32;

        #[unroll]
        for ch in 0u32..3u32 {
            let tl = u32::cast_from(src[usize::cast_from(row_top + off_l + ch)]);
            let tr = u32::cast_from(src[usize::cast_from(row_top + off_r + ch)]);
            let bl = u32::cast_from(src[usize::cast_from(row_bot + off_l + ch)]);
            let br = u32::cast_from(src[usize::cast_from(row_bot + off_r + ch)]);

            let top = inv_wx * tl + wx * tr;
            let bot = inv_wx * bl + wx * br;
            let val = (inv_wy * top + wy * bot + (1u32 << 15)) >> 16u32;

            dst[usize::cast_from(dst_off + ch)] = u8::cast_from(val);
        }
    }
}
