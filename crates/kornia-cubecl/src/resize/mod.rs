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

/// Bundle of pre-uploaded device handles for the bilinear weight tables.
/// Build once via `WeightHandles::new`, reuse across many dispatches with the
/// same (src_w, src_h, dst_w, dst_h). Saves the four `create_from_slice` uploads
/// that `resize_bilinear_u8_rgb` does per call — material at small sizes where
/// dispatch overhead dominates.
pub struct WeightHandles {
    pub wx_idx: Handle,
    pub wx_w: Handle,
    pub wy_idx: Handle,
    pub wy_w: Handle,
    pub wx_len: usize,
    pub wy_len: usize,
}

impl WeightHandles {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, src_size: ImageSize, dst_size: ImageSize) -> Self {
        let wx = compute_axis_weights(src_size.width as u32, dst_size.width as u32);
        let wy = compute_axis_weights(src_size.height as u32, dst_size.height as u32);
        let wx_idx_v: Vec<u32> = wx.iter().map(|w| w.src_idx).collect();
        let wx_w_v: Vec<u32> = wx.iter().map(|w| w.weight_x256 as u32).collect();
        let wy_idx_v: Vec<u32> = wy.iter().map(|w| w.src_idx).collect();
        let wy_w_v: Vec<u32> = wy.iter().map(|w| w.weight_x256 as u32).collect();
        Self {
            wx_idx: client.create_from_slice(bytemuck::cast_slice(&wx_idx_v)),
            wx_w: client.create_from_slice(bytemuck::cast_slice(&wx_w_v)),
            wy_idx: client.create_from_slice(bytemuck::cast_slice(&wy_idx_v)),
            wy_w: client.create_from_slice(bytemuck::cast_slice(&wy_w_v)),
            wx_len: wx_v_len(&wx),
            wy_len: wx_v_len(&wy),
        }
    }
}

fn wx_v_len(v: &[weights::AxisWeight]) -> usize { v.len() }

/// Bilinear u8 RGB resize using PRE-uploaded weight handles. Skips per-call
/// weight uploads — use this in a hot loop where `(src_size, dst_size)` is fixed.
pub fn resize_bilinear_u8_rgb_with_weights<R: Runtime>(
    client: &ComputeClient<R>,
    src: &Handle,
    src_size: ImageSize,
    dst: &Handle,
    dst_size: ImageSize,
    weights: &WeightHandles,
) -> Result<(), ResizeError> {
    if src_size.width == 0 || src_size.height == 0 || dst_size.width == 0 || dst_size.height == 0 {
        return Err(ResizeError::ZeroDimension);
    }
    let src_w = src_size.width as u32;
    let dst_w = dst_size.width as u32;
    let dst_h = dst_size.height as u32;

    let cube_dim = CubeDim::new_2d(16, 16);
    let cube_count = CubeCount::new_2d(dst_w.div_ceil(16), dst_h.div_ceil(16));
    let src_len = (src_size.width) * (src_size.height) * 3;
    let dst_len = (dst_size.width) * (dst_size.height) * 3;

    unsafe {
        resize_bilinear_u8_rgb_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src.clone(), src_len),
            ArrayArg::from_raw_parts(dst.clone(), dst_len),
            ArrayArg::from_raw_parts(weights.wx_idx.clone(), weights.wx_len),
            ArrayArg::from_raw_parts(weights.wx_w.clone(), weights.wx_len),
            ArrayArg::from_raw_parts(weights.wy_idx.clone(), weights.wy_len),
            ArrayArg::from_raw_parts(weights.wy_w.clone(), weights.wy_len),
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
