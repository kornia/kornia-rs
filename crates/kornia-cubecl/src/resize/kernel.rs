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
