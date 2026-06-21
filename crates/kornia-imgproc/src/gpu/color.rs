//! CubeCL grayscale conversion kernels.
//!
//! CPU reference: `kornia_imgproc::color::gray_from_rgb` (BT.601 weights).
//! GPU version: one thread per output pixel, 1-D flat grid.

use cubecl::prelude::*;

// BT.601 luma weights encoded as compile-time constants.
const RW: f32 = 0.299;
const GW: f32 = 0.587;
const BW: f32 = 0.114;

const BLOCK_SIZE: u32 = 256;

/// CubeCL kernel: convert a packed RGB buffer to a packed grayscale buffer (f32).
///
/// `src` must have `num_pixels * 3` elements; `dst` must have `num_pixels` elements.
/// Threads beyond `num_pixels` call `terminate!()`.
#[cube(launch)]
fn gray_from_rgb_f32_kernel(src: &Array<f32>, dst: &mut Array<f32>, num_pixels: u32) {
    if ABSOLUTE_POS >= usize::cast_from(num_pixels) {
        terminate!();
    }
    let base = ABSOLUTE_POS * 3;
    let r = src[base];
    let g = src[base + 1];
    let b = src[base + 2];
    dst[ABSOLUTE_POS] = f32::new(RW) * r + f32::new(GW) * g + f32::new(BW) * b;
}

/// Launch the `gray_from_rgb` kernel on the given compute client.
///
/// # Arguments
///
/// * `client` – CubeCL compute client for the target device.
/// * `src` – device handle for the packed RGB buffer (`height * width * 3` f32 values).
/// * `dst` – device handle for the packed grayscale output (`height * width` f32 values).
/// * `width`, `height` – image dimensions in pixels.
///
/// # Panics
///
/// Panics if the kernel launch fails (e.g. driver error).
pub fn launch_gray_from_rgb_f32<R: Runtime>(
    client: &ComputeClient<R>,
    src: cubecl::server::Handle,
    dst: cubecl::server::Handle,
    width: u32,
    height: u32,
) {
    // Use u64 arithmetic to avoid u32 overflow on large images.
    let num_pixels_u64 = width as u64 * height as u64;
    if num_pixels_u64 == 0 {
        return;
    }
    let num_pixels = num_pixels_u64 as usize;
    let num_cubes = num_pixels.div_ceil(BLOCK_SIZE as usize) as u32;

    unsafe {
        gray_from_rgb_f32_kernel::launch::<R>(
            client,
            CubeCount::Static(num_cubes, 1, 1),
            CubeDim {
                x: BLOCK_SIZE,
                y: 1,
                z: 1,
            },
            ArrayArg::from_raw_parts(src, num_pixels * 3),
            ArrayArg::from_raw_parts(dst, num_pixels),
            num_pixels as u32,
        )
    }
}
