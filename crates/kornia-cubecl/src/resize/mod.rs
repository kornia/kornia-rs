pub mod kernel;
pub mod weights;

use crate::error::ResizeError;
use cubecl::prelude::*;
use cubecl::server::Handle;
use kornia_image::ImageSize;

use kernel::{resize_bilinear_u8_rgb_kernel, resize_bilinear_u8_rgb_kernel_x16, resize_bilinear_u8_rgb_kernel_x4};
use weights::compute_axis_weights;

/// Run the bilinear u8 RGB resize kernel on the given runtime.
///
/// The caller owns all device buffers. `src` and `dst` must be u8-typed handles
/// of length `src_size.width * src_size.height * 3` and `dst_size.width * dst_size.height * 3`
/// respectively. Weight tables are computed and uploaded internally each call
/// (kept inside the dispatch boundary so end-to-end callers see the realistic cost).
pub fn resize_bilinear_u8_rgb<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    src_size: ImageSize,
    dst: &Handle,
    dst_size: ImageSize,
) -> Result<(), ResizeError> {
    if src_size.width == 0 || src_size.height == 0 || dst_size.width == 0 || dst_size.height == 0 {
        return Err(ResizeError::ZeroDimension);
    }

    let src_w = src_size.width as u32;
    let src_h = src_size.height as u32;
    let dst_w = dst_size.width as u32;
    let dst_h = dst_size.height as u32;

    let wx = compute_axis_weights(src_w, dst_w);
    let wy = compute_axis_weights(src_h, dst_h);

    let wx_idx: Vec<u32> = wx.iter().map(|w| w.src_idx).collect();
    let wx_w: Vec<u32> = wx.iter().map(|w| w.weight_x256 as u32).collect();
    let wy_idx: Vec<u32> = wy.iter().map(|w| w.src_idx).collect();
    let wy_w: Vec<u32> = wy.iter().map(|w| w.weight_x256 as u32).collect();

    let wx_idx_h = client.create_from_slice(bytemuck::cast_slice(&wx_idx));
    let wx_w_h = client.create_from_slice(bytemuck::cast_slice(&wx_w));
    let wy_idx_h = client.create_from_slice(bytemuck::cast_slice(&wy_idx));
    let wy_w_h = client.create_from_slice(bytemuck::cast_slice(&wy_w));

    let cube_dim = CubeDim::new_2d(16, 16);
    let cube_count = CubeCount::new_2d(dst_w.div_ceil(16), dst_h.div_ceil(16));

    let src_len = (src_w as usize) * (src_h as usize) * 3;
    let dst_len = (dst_w as usize) * (dst_h as usize) * 3;

    unsafe {
        resize_bilinear_u8_rgb_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src.clone(), src_len),
            ArrayArg::from_raw_parts(dst.clone(), dst_len),
            ArrayArg::from_raw_parts(wx_idx_h, wx_idx.len()),
            ArrayArg::from_raw_parts(wx_w_h, wx_w.len()),
            ArrayArg::from_raw_parts(wy_idx_h, wy_idx.len()),
            ArrayArg::from_raw_parts(wy_w_h, wy_w.len()),
            src_w,
            dst_w,
            dst_h,
        );
    }

    Ok(())
}

/// Same as `resize_bilinear_u8_rgb` but launches the 4-pixels-per-thread
/// variant (`resize_bilinear_u8_rgb_kernel_x4`). Requires `dst_size.width % 4 == 0`.
pub fn resize_bilinear_u8_rgb_x4<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    src_size: ImageSize,
    dst: &Handle,
    dst_size: ImageSize,
) -> Result<(), ResizeError> {
    if src_size.width == 0 || src_size.height == 0 || dst_size.width == 0 || dst_size.height == 0 {
        return Err(ResizeError::ZeroDimension);
    }
    if dst_size.width % 4 != 0 {
        return Err(ResizeError::BufferSize { expected: dst_size.width, got: dst_size.width });
    }

    let src_w = src_size.width as u32;
    let src_h = src_size.height as u32;
    let dst_w = dst_size.width as u32;
    let dst_h = dst_size.height as u32;

    let wx = compute_axis_weights(src_w, dst_w);
    let wy = compute_axis_weights(src_h, dst_h);

    let wx_idx: Vec<u32> = wx.iter().map(|w| w.src_idx).collect();
    let wx_w: Vec<u32> = wx.iter().map(|w| w.weight_x256 as u32).collect();
    let wy_idx: Vec<u32> = wy.iter().map(|w| w.src_idx).collect();
    let wy_w: Vec<u32> = wy.iter().map(|w| w.weight_x256 as u32).collect();

    let wx_idx_h = client.create_from_slice(bytemuck::cast_slice(&wx_idx));
    let wx_w_h = client.create_from_slice(bytemuck::cast_slice(&wx_w));
    let wy_idx_h = client.create_from_slice(bytemuck::cast_slice(&wy_idx));
    let wy_w_h = client.create_from_slice(bytemuck::cast_slice(&wy_w));

    let cube_dim = CubeDim::new_2d(16, 16);
    // tile_x covers 4 dst pixels each → tile_count_x = ceil(dst_w / 4)
    let cube_count = CubeCount::new_2d((dst_w / 4).div_ceil(16), dst_h.div_ceil(16));

    let src_len = (src_w as usize) * (src_h as usize) * 3;
    let dst_len = (dst_w as usize) * (dst_h as usize) * 3;

    unsafe {
        resize_bilinear_u8_rgb_kernel_x4::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src.clone(), src_len),
            ArrayArg::from_raw_parts(dst.clone(), dst_len),
            ArrayArg::from_raw_parts(wx_idx_h, wx_idx.len()),
            ArrayArg::from_raw_parts(wx_w_h, wx_w.len()),
            ArrayArg::from_raw_parts(wy_idx_h, wy_idx.len()),
            ArrayArg::from_raw_parts(wy_w_h, wy_w.len()),
            src_w,
            dst_w,
            dst_h,
        );
    }

    Ok(())
}

/// Same as `resize_bilinear_u8_rgb` but launches the 16-pixels-per-thread variant.
/// Requires `dst_size.width % 16 == 0`.
pub fn resize_bilinear_u8_rgb_x16<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    src_size: ImageSize,
    dst: &Handle,
    dst_size: ImageSize,
) -> Result<(), ResizeError> {
    if src_size.width == 0 || src_size.height == 0 || dst_size.width == 0 || dst_size.height == 0 {
        return Err(ResizeError::ZeroDimension);
    }
    if dst_size.width % 16 != 0 {
        return Err(ResizeError::BufferSize { expected: dst_size.width, got: dst_size.width });
    }

    let src_w = src_size.width as u32;
    let src_h = src_size.height as u32;
    let dst_w = dst_size.width as u32;
    let dst_h = dst_size.height as u32;

    let wx = compute_axis_weights(src_w, dst_w);
    let wy = compute_axis_weights(src_h, dst_h);

    let wx_idx: Vec<u32> = wx.iter().map(|w| w.src_idx).collect();
    let wx_w: Vec<u32> = wx.iter().map(|w| w.weight_x256 as u32).collect();
    let wy_idx: Vec<u32> = wy.iter().map(|w| w.src_idx).collect();
    let wy_w: Vec<u32> = wy.iter().map(|w| w.weight_x256 as u32).collect();

    let wx_idx_h = client.create_from_slice(bytemuck::cast_slice(&wx_idx));
    let wx_w_h = client.create_from_slice(bytemuck::cast_slice(&wx_w));
    let wy_idx_h = client.create_from_slice(bytemuck::cast_slice(&wy_idx));
    let wy_w_h = client.create_from_slice(bytemuck::cast_slice(&wy_w));

    let cube_dim = CubeDim::new_2d(16, 16);
    let cube_count = CubeCount::new_2d((dst_w / 16).div_ceil(16), dst_h.div_ceil(16));

    let src_len = (src_w as usize) * (src_h as usize) * 3;
    let dst_len = (dst_w as usize) * (dst_h as usize) * 3;

    unsafe {
        resize_bilinear_u8_rgb_kernel_x16::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src.clone(), src_len),
            ArrayArg::from_raw_parts(dst.clone(), dst_len),
            ArrayArg::from_raw_parts(wx_idx_h, wx_idx.len()),
            ArrayArg::from_raw_parts(wx_w_h, wx_w.len()),
            ArrayArg::from_raw_parts(wy_idx_h, wy_idx.len()),
            ArrayArg::from_raw_parts(wy_w_h, wy_w.len()),
            src_w,
            dst_w,
            dst_h,
        );
    }

    Ok(())
}
