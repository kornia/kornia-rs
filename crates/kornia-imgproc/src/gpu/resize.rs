//! CubeCL resize kernels (f32, nearest-neighbor and bilinear).
//!
//! Public API: [`launch_resize_nearest_f32`] and [`launch_resize_bilinear_f32`].
//! Both dispatch to a 3-channel specialised kernel for RGB (the common case)
//! and fall back to a dynamic-channel kernel otherwise.
//!
//! Design choices:
//! - 1 thread per output pixel, 1-D flat grid.
//! - Scale factors precomputed on the host (removes 2 float divides per thread).
//! - 3ch kernels: hardcoded loops → CUDA JIT sees literal bounds, better ILP.
//! - Bilinear: 4 weights computed once then applied per-channel, bounding live
//!   register count to ~20 instead of 4×nc color values loaded upfront.
//! - Block size 256: larger tiles improve L1 cache reuse on upscale workloads
//!   where neighbouring output threads share source cache lines.

use cubecl::prelude::*;

const BLOCK_SIZE: u32 = 256;

// ─── nearest-neighbor, dynamic channel count ─────────────────────────────────

/// Source coordinates use half-pixel center alignment:
/// `src_xi = clamp(floor((dst_x + 0.5) * scale_x), 0, src_w − 1)`.
#[cube(launch)]
fn resize_nearest_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    num_channels: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let dw = usize::cast_from(dst_w);
    let dh = usize::cast_from(dst_h);
    if ABSOLUTE_POS >= dw * dh {
        terminate!();
    }
    let dst_x = ABSOLUTE_POS % dw;
    let dst_y = ABSOLUTE_POS / dw;

    // Route f32 → u32 → usize to avoid a direct f32→usize cast on GPU targets.
    let sw = usize::cast_from(src_w);
    let src_xf = (f32::cast_from(dst_x) + f32::new(0.5)) * scale_x;
    let src_yf = (f32::cast_from(dst_y) + f32::new(0.5)) * scale_y;
    let src_xi = min(usize::cast_from(u32::cast_from(src_xf)), sw - 1);
    let src_yi = min(
        usize::cast_from(u32::cast_from(src_yf)),
        usize::cast_from(src_h) - 1,
    );

    let nc = usize::cast_from(num_channels);
    let src_base = (src_yi * sw + src_xi) * nc;
    let dst_base = ABSOLUTE_POS * nc;
    for c in 0..nc {
        dst[dst_base + c] = src[src_base + c];
    }
}

// ─── nearest-neighbor, 3 channels hardcoded ──────────────────────────────────

/// 3-channel specialisation: literal loop bound lets the JIT emit all loads
/// with independent address expressions for better instruction-level parallelism.
#[cube(launch)]
fn resize_nearest_3c_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let dw = usize::cast_from(dst_w);
    let dh = usize::cast_from(dst_h);
    if ABSOLUTE_POS >= dw * dh {
        terminate!();
    }
    let dst_x = ABSOLUTE_POS % dw;
    let dst_y = ABSOLUTE_POS / dw;

    let sw = usize::cast_from(src_w);
    let src_xf = (f32::cast_from(dst_x) + f32::new(0.5)) * scale_x;
    let src_yf = (f32::cast_from(dst_y) + f32::new(0.5)) * scale_y;
    let src_xi = min(usize::cast_from(u32::cast_from(src_xf)), sw - 1);
    let src_yi = min(
        usize::cast_from(u32::cast_from(src_yf)),
        usize::cast_from(src_h) - 1,
    );

    let src_base = (src_yi * sw + src_xi) * 3;
    let dst_base = ABSOLUTE_POS * 3;
    dst[dst_base] = src[src_base];
    dst[dst_base + 1] = src[src_base + 1];
    dst[dst_base + 2] = src[src_base + 2];
}

// ─── bilinear, dynamic channel count ─────────────────────────────────────────

/// Source coordinates use half-pixel center alignment:
/// `sx = clamp((dst_x + 0.5) * scale_x − 0.5, 0, src_w − 1)`.
/// Four neighbouring pixels are blended with precomputed bilinear weights.
#[cube(launch)]
fn resize_bilinear_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    num_channels: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let dw = usize::cast_from(dst_w);
    let dh = usize::cast_from(dst_h);
    if ABSOLUTE_POS >= dw * dh {
        terminate!();
    }
    let dst_x = ABSOLUTE_POS % dw;
    let dst_y = ABSOLUTE_POS / dw;

    let src_w_m1 = f32::cast_from(src_w) - f32::new(1.0);
    let src_h_m1 = f32::cast_from(src_h) - f32::new(1.0);
    let sx_raw = (f32::cast_from(dst_x) + f32::new(0.5)) * scale_x - f32::new(0.5);
    let sy_raw = (f32::cast_from(dst_y) + f32::new(0.5)) * scale_y - f32::new(0.5);
    let sx = max(min(sx_raw, src_w_m1), f32::new(0.0));
    let sy = max(min(sy_raw, src_h_m1), f32::new(0.0));

    let sw = usize::cast_from(src_w);
    let sh = usize::cast_from(src_h);
    let x0u = u32::cast_from(sx);
    let y0u = u32::cast_from(sy);
    let x0 = usize::cast_from(x0u);
    let y0 = usize::cast_from(y0u);
    let x1 = min(x0 + 1, sw - 1);
    let y1 = min(y0 + 1, sh - 1);

    let fx = sx - f32::cast_from(x0u);
    let fy = sy - f32::cast_from(y0u);

    // Precompute 4 weights once; apply to each channel sequentially.
    // Keeps live register count bounded to ~4 weights rather than 4×nc values.
    let w00 = (f32::new(1.0) - fy) * (f32::new(1.0) - fx);
    let w10 = (f32::new(1.0) - fy) * fx;
    let w01 = fy * (f32::new(1.0) - fx);
    let w11 = fy * fx;

    let nc = usize::cast_from(num_channels);
    let r0 = y0 * sw;
    let r1 = y1 * sw;
    let dst_base = ABSOLUTE_POS * nc;

    for c in 0..nc {
        dst[dst_base + c] = w00 * src[(r0 + x0) * nc + c]
            + w10 * src[(r0 + x1) * nc + c]
            + w01 * src[(r1 + x0) * nc + c]
            + w11 * src[(r1 + x1) * nc + c];
    }
}

// ─── bilinear, 3 channels hardcoded ──────────────────────────────────────────

/// 3-channel specialisation: precomputed weights + literal channel count.
#[cube(launch)]
fn resize_bilinear_3c_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let dw = usize::cast_from(dst_w);
    let dh = usize::cast_from(dst_h);
    if ABSOLUTE_POS >= dw * dh {
        terminate!();
    }
    let dst_x = ABSOLUTE_POS % dw;
    let dst_y = ABSOLUTE_POS / dw;

    let src_w_m1 = f32::cast_from(src_w) - f32::new(1.0);
    let src_h_m1 = f32::cast_from(src_h) - f32::new(1.0);
    let sx_raw = (f32::cast_from(dst_x) + f32::new(0.5)) * scale_x - f32::new(0.5);
    let sy_raw = (f32::cast_from(dst_y) + f32::new(0.5)) * scale_y - f32::new(0.5);
    let sx = max(min(sx_raw, src_w_m1), f32::new(0.0));
    let sy = max(min(sy_raw, src_h_m1), f32::new(0.0));

    let sw = usize::cast_from(src_w);
    let sh = usize::cast_from(src_h);
    let x0u = u32::cast_from(sx);
    let y0u = u32::cast_from(sy);
    let x0 = usize::cast_from(x0u);
    let y0 = usize::cast_from(y0u);
    let x1 = min(x0 + 1, sw - 1);
    let y1 = min(y0 + 1, sh - 1);

    let fx = sx - f32::cast_from(x0u);
    let fy = sy - f32::cast_from(y0u);

    let w00 = (f32::new(1.0) - fy) * (f32::new(1.0) - fx);
    let w10 = (f32::new(1.0) - fy) * fx;
    let w01 = fy * (f32::new(1.0) - fx);
    let w11 = fy * fx;

    let r0 = y0 * sw;
    let r1 = y1 * sw;
    let dst_base = ABSOLUTE_POS * 3;

    let b00 = (r0 + x0) * 3;
    let b10 = (r0 + x1) * 3;
    let b01 = (r1 + x0) * 3;
    let b11 = (r1 + x1) * 3;

    dst[dst_base] = w00 * src[b00] + w10 * src[b10] + w01 * src[b01] + w11 * src[b11];
    dst[dst_base + 1] =
        w00 * src[b00 + 1] + w10 * src[b10 + 1] + w01 * src[b01 + 1] + w11 * src[b11 + 1];
    dst[dst_base + 2] =
        w00 * src[b00 + 2] + w10 * src[b10 + 2] + w01 * src[b01 + 2] + w11 * src[b11 + 2];
}

// ─── public launchers ────────────────────────────────────────────────────────

/// Launch the nearest-neighbor resize kernel.
///
/// One thread per output pixel. Dispatches to the 3-channel specialised kernel
/// for RGB images; falls back to the dynamic-channel kernel otherwise.
///
/// # Arguments
///
/// * `client` – CubeCL compute client for the target device.
/// * `src` – device handle: `src_height × src_width × num_channels` f32 values.
/// * `dst` – device handle: `dst_height × dst_width × num_channels` f32 values.
/// * `src_width`, `src_height` – source image dimensions in pixels.
/// * `dst_width`, `dst_height` – output image dimensions in pixels.
/// * `num_channels` – channels per pixel (1 for grayscale, 3 for RGB, 4 for RGBA).
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_nearest_f32<R: Runtime>(
    client: &ComputeClient<R>,
    src: cubecl::server::Handle,
    dst: cubecl::server::Handle,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    num_channels: u32,
) {
    let num_dst_pixels = dst_width as u64 * dst_height as u64;
    if num_dst_pixels == 0 {
        return;
    }
    let num_src_pixels = src_width as usize * src_height as usize;
    let num_dst_pixels = num_dst_pixels as usize;
    let nc = num_channels as usize;
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;
    let num_cubes = num_dst_pixels.div_ceil(BLOCK_SIZE as usize) as u32;

    if num_channels == 3 {
        unsafe {
            resize_nearest_3c_kernel::launch::<R>(
                client,
                CubeCount::Static(num_cubes, 1, 1),
                CubeDim {
                    x: BLOCK_SIZE,
                    y: 1,
                    z: 1,
                },
                ArrayArg::from_raw_parts(src, num_src_pixels * nc),
                ArrayArg::from_raw_parts(dst, num_dst_pixels * nc),
                src_width,
                src_height,
                dst_width,
                dst_height,
                scale_x,
                scale_y,
            )
        }
    } else {
        unsafe {
            resize_nearest_kernel::launch::<R>(
                client,
                CubeCount::Static(num_cubes, 1, 1),
                CubeDim {
                    x: BLOCK_SIZE,
                    y: 1,
                    z: 1,
                },
                ArrayArg::from_raw_parts(src, num_src_pixels * nc),
                ArrayArg::from_raw_parts(dst, num_dst_pixels * nc),
                src_width,
                src_height,
                dst_width,
                dst_height,
                num_channels,
                scale_x,
                scale_y,
            )
        }
    }
}

/// Launch the bilinear resize kernel.
///
/// One thread per output pixel. Dispatches to the 3-channel specialised kernel
/// for RGB images; falls back to the dynamic-channel kernel otherwise.
///
/// # Arguments
///
/// * `client` – CubeCL compute client for the target device.
/// * `src` – device handle: `src_height × src_width × num_channels` f32 values.
/// * `dst` – device handle: `dst_height × dst_width × num_channels` f32 values.
/// * `src_width`, `src_height` – source image dimensions in pixels.
/// * `dst_width`, `dst_height` – output image dimensions in pixels.
/// * `num_channels` – channels per pixel (1 for grayscale, 3 for RGB, 4 for RGBA).
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_f32<R: Runtime>(
    client: &ComputeClient<R>,
    src: cubecl::server::Handle,
    dst: cubecl::server::Handle,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    num_channels: u32,
) {
    let num_dst_pixels = dst_width as u64 * dst_height as u64;
    if num_dst_pixels == 0 {
        return;
    }
    let num_src_pixels = src_width as usize * src_height as usize;
    let num_dst_pixels = num_dst_pixels as usize;
    let nc = num_channels as usize;
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;
    let num_cubes = num_dst_pixels.div_ceil(BLOCK_SIZE as usize) as u32;

    if num_channels == 3 {
        unsafe {
            resize_bilinear_3c_kernel::launch::<R>(
                client,
                CubeCount::Static(num_cubes, 1, 1),
                CubeDim {
                    x: BLOCK_SIZE,
                    y: 1,
                    z: 1,
                },
                ArrayArg::from_raw_parts(src, num_src_pixels * nc),
                ArrayArg::from_raw_parts(dst, num_dst_pixels * nc),
                src_width,
                src_height,
                dst_width,
                dst_height,
                scale_x,
                scale_y,
            )
        }
    } else {
        unsafe {
            resize_bilinear_kernel::launch::<R>(
                client,
                CubeCount::Static(num_cubes, 1, 1),
                CubeDim {
                    x: BLOCK_SIZE,
                    y: 1,
                    z: 1,
                },
                ArrayArg::from_raw_parts(src, num_src_pixels * nc),
                ArrayArg::from_raw_parts(dst, num_dst_pixels * nc),
                src_width,
                src_height,
                dst_width,
                dst_height,
                num_channels,
                scale_x,
                scale_y,
            )
        }
    }
}

#[cfg(test)]
#[cfg(feature = "gpu-cubecl")]
mod tests {
    use super::*;
    use cubecl::prelude::{CubeElement, Runtime};
    use cubecl_cpu::CpuRuntime;

    fn run_nearest(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        nc: u32,
    ) -> Vec<f32> {
        let client = CpuRuntime::client(&Default::default());
        let src_handle = client.create_from_slice(f32::as_bytes(src));
        let dst_len = (dst_w * dst_h * nc) as usize;
        let dst_handle = client.empty(dst_len * std::mem::size_of::<f32>());
        launch_resize_nearest_f32::<CpuRuntime>(
            &client,
            src_handle,
            dst_handle.clone(),
            src_w,
            src_h,
            dst_w,
            dst_h,
            nc,
        );
        let bytes = client.read_one_unchecked(dst_handle);
        f32::from_bytes(&bytes).to_vec()
    }

    fn run_bilinear(
        src: &[f32],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        nc: u32,
    ) -> Vec<f32> {
        let client = CpuRuntime::client(&Default::default());
        let src_handle = client.create_from_slice(f32::as_bytes(src));
        let dst_len = (dst_w * dst_h * nc) as usize;
        let dst_handle = client.empty(dst_len * std::mem::size_of::<f32>());
        launch_resize_bilinear_f32::<CpuRuntime>(
            &client,
            src_handle,
            dst_handle.clone(),
            src_w,
            src_h,
            dst_w,
            dst_h,
            nc,
        );
        let bytes = client.read_one_unchecked(dst_handle);
        f32::from_bytes(&bytes).to_vec()
    }

    /// Nearest-neighbor identity: 1:1 resize must return the same pixels.
    #[test]
    fn test_nearest_identity_1c() {
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let out = run_nearest(&src, 3, 2, 3, 2, 1);
        assert_eq!(out, src);
    }

    /// Nearest-neighbor 2× upscale of a 2×2 grayscale image.
    ///
    /// src (row-major):
    ///   [0, 1]
    ///   [2, 3]
    ///
    /// Each source pixel maps to a 2×2 block in the 4×4 output.
    #[test]
    fn test_nearest_upscale_2x_1c() {
        let src = [0.0_f32, 1.0, 2.0, 3.0];
        let out = run_nearest(&src, 2, 2, 4, 4, 1);
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            2.0, 2.0, 3.0, 3.0,
            2.0, 2.0, 3.0, 3.0,
        ];
        assert_eq!(out, expected);
    }

    /// Nearest-neighbor 2× downscale of a 4×4 grayscale image.
    ///
    /// With half-pixel center mapping each output pixel samples the
    /// center of its 2×2 source block (the lower-right of the four).
    #[test]
    fn test_nearest_downscale_2x_1c() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = run_nearest(&src, 4, 4, 2, 2, 1);
        // (dx=0,dy=0)→sx=floor(0.5*2)=1, sy=1 → src[1*4+1]=5
        // (dx=1,dy=0)→sx=floor(1.5*2)=3, sy=1 → src[1*4+3]=7
        // (dx=0,dy=1)→sx=1, sy=3           → src[3*4+1]=13
        // (dx=1,dy=1)→sx=3, sy=3           → src[3*4+3]=15
        assert_eq!(out, [5.0, 7.0, 13.0, 15.0]);
    }

    /// Bilinear identity: 1:1 resize must exactly reproduce the source.
    #[test]
    fn test_bilinear_identity_1c() {
        let src: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let out = run_bilinear(&src, 3, 2, 3, 2, 1);
        assert_eq!(out, src);
    }

    /// Bilinear upscale of a uniform image: all outputs equal the constant.
    #[test]
    fn test_bilinear_uniform_upscale() {
        let src = [7.0_f32; 4];
        let out = run_bilinear(&src, 2, 2, 4, 4, 1);
        for v in &out {
            assert!((v - 7.0).abs() < 1e-5, "expected 7.0, got {v}");
        }
    }

    /// Bilinear center pixel of a 2× upscale must be a proper blend.
    ///
    /// src:
    ///   [0, 4]
    ///   [8, 12]
    ///
    /// At dst=(col=1, row=1):
    ///   sx = (1+0.5)*0.5 − 0.5 = 0.25, sy = 0.25
    ///   → 0.75*0.75*0 + 0.25*0.75*4 + 0.75*0.25*8 + 0.25*0.25*12 = 3.0
    #[test]
    fn test_bilinear_center_blend() {
        let src = [0.0_f32, 4.0, 8.0, 12.0];
        let out = run_bilinear(&src, 2, 2, 4, 4, 1);
        let center = out[1 * 4 + 1];
        assert!((center - 3.0).abs() < 1e-4, "expected 3.0, got {center}");
    }

    /// Nearest-neighbor identity for a 3-channel (RGB) image.
    #[test]
    fn test_nearest_identity_3c() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = run_nearest(&src, 2, 2, 2, 2, 3);
        assert_eq!(out, src);
    }

    /// Bilinear identity for a 3-channel (RGB) image.
    #[test]
    fn test_bilinear_identity_3c() {
        let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = run_bilinear(&src, 2, 2, 2, 2, 3);
        assert_eq!(out, src);
    }
}
