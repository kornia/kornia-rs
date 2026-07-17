//! Device adapters for [`remap`](super::remap): routes checked device
//! (src, dst, map_x, map_y) quadruples to the native CUDA remap kernels.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err};
use crate::cuda::remap::{launch_remap_bilinear_cuda, launch_remap_nearest_cuda};
use crate::interpolation::InterpolationMode;

/// Run the CUDA remap for a device-resident f32 triple (src, dst, maps).
///
/// `map_x` and `map_y` are device slices of length `dst_h * dst_w` — the
/// same size as a flattened single-channel output image.  Bilinear and nearest
/// are hardware-accelerated; bicubic and lanczos must be handled by the CPU
/// path (the caller, `interpolation::remap`, falls through for those modes).
pub(super) fn remap_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    map_x: &CudaSlice<f32>,
    map_y: &CudaSlice<f32>,
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err("remap", "3-channel f32 images"));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => launch_remap_bilinear_cuda(
            ctx, stream, s, map_x, map_y, d, src_w, src_h, dst_w, dst_h, None,
        ),
        InterpolationMode::Nearest => launch_remap_nearest_cuda(
            ctx, stream, s, map_x, map_y, d, src_w, src_h, dst_w, dst_h, None,
        ),
        other => Err(crate::cuda::remap::CudaRemapError::Cuda(format!(
            "remap CUDA: {other:?} is not GPU-accelerated — move images to host for this mode"
        ))),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::{remap::remap, InterpolationMode};
    use kornia_image::{Image, ImageError, ImageSize};

    fn identity_maps(w: usize, h: usize) -> (Image<f32, 1>, Image<f32, 1>) {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mx: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|x| x as f32)).collect();
        let my: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect();
        (
            Image::<f32, 1>::new(size, mx).unwrap(),
            Image::<f32, 1>::new(size, my).unwrap(),
        )
    }

    /// `remap` with device images and an identity map must be bit-identical to
    /// the CPU path — the byte-exact contract for the remap kernel.
    #[test]
    #[ignore = "requires CUDA"]
    fn public_remap_device_equals_host() -> Result<(), ImageError> {
        let stream = default_stream();
        let (w, h) = (65, 33);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, pattern_f32(w * h * 3)).unwrap();
        let (mx, my) = identity_maps(w, h);

        let mut cpu_dst = Image::<f32, 3>::from_size_val(size, 0.0)?;
        remap(&src, &mut cpu_dst, &mx, &my, InterpolationMode::Bilinear)?;

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(size, &stream).unwrap();
        let d_mx = mx.to_cuda(&stream).unwrap();
        let d_my = my.to_cuda(&stream).unwrap();
        remap(
            &d_src,
            &mut d_dst,
            &d_mx,
            &d_my,
            InterpolationMode::Bilinear,
        )?;

        let back = d_dst.to_host_owned()?;
        for (i, (c, g)) in cpu_dst.as_slice().iter().zip(back.as_slice()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "remap bilinear element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                c.to_bits(),
                g.to_bits()
            );
        }
        Ok(())
    }

    /// Nearest-neighbor device path is bit-identical to host.
    #[test]
    #[ignore = "requires CUDA"]
    fn public_remap_nearest_device_equals_host() -> Result<(), ImageError> {
        let stream = default_stream();
        let (w, h) = (65, 33);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, pattern_f32(w * h * 3)).unwrap();
        let (mx, my) = identity_maps(w, h);

        let mut cpu_dst = Image::<f32, 3>::from_size_val(size, 0.0)?;
        remap(&src, &mut cpu_dst, &mx, &my, InterpolationMode::Nearest)?;

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(size, &stream).unwrap();
        let d_mx = mx.to_cuda(&stream).unwrap();
        let d_my = my.to_cuda(&stream).unwrap();
        remap(&d_src, &mut d_dst, &d_mx, &d_my, InterpolationMode::Nearest)?;

        let back = d_dst.to_host_owned()?;
        for (i, (c, g)) in cpu_dst.as_slice().iter().zip(back.as_slice()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "remap nearest element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                c.to_bits(),
                g.to_bits()
            );
        }
        Ok(())
    }

    /// Mixed residency — device src/dst but host maps — must be a typed error.
    #[test]
    #[ignore = "requires CUDA"]
    fn device_images_with_host_maps_is_error() {
        let stream = default_stream();
        let (w, h) = (16, 16);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, pattern_f32(w * h * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(size, &stream).unwrap();

        let (mx, my) = identity_maps(w, h); // host maps

        let err = remap(&d_src, &mut d_dst, &mx, &my, InterpolationMode::Bilinear).unwrap_err();
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("device-resident")),
            "expected a Cuda error about device-resident maps, got {err:?}"
        );
    }
}
