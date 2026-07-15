//! Device adapter for [`resize`](super::resize): routes a
//! checked device pair to the native CUDA resize kernels.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err};
use crate::cuda::resize::{
    launch_resize_bilinear_downscale_cuda, launch_resize_nearest_downscale_cuda, PixelMapping,
};
use crate::interpolation::InterpolationMode;

/// Run the CUDA resize for a device-resident f32 pair.
///
/// The kernels are 3-channel only, so `C != 3` errors (never a silent CPU
/// fallback — see `cuda::dispatch`). Output is bit-identical to the CPU
/// [`resize`](super::resize) — the byte-exact contract
/// established with the half-pixel grid and `--fmad=false`, asserted by the
/// parity tests in `cuda::resize`.
pub(super) fn resize_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err("resize", "3-channel f32 images"));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => launch_resize_bilinear_downscale_cuda(
            ctx,
            stream,
            s,
            d,
            src_w,
            src_h,
            dst_w,
            dst_h,
            PixelMapping::HalfPixel,
            None,
        ),
        InterpolationMode::Nearest => launch_resize_nearest_downscale_cuda(
            ctx,
            stream,
            s,
            d,
            src_w,
            src_h,
            dst_w,
            dst_h,
            PixelMapping::HalfPixel,
            None,
        ),
        // validate_interpolation in resize rejects everything else
        // before this adapter is reached.
        mode => return Err(ImageError::UnsupportedInterpolation(mode)),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use crate::resize::resize;
    use kornia_image::{Image, ImageError, ImageSize};

    fn sized(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// The public `resize`, called with device images, must produce
    /// bit-identical output to the same call with host images — for both
    /// modes, downscale and upscale, at non-dyadic sizes.
    #[test]
    fn public_resize_device_equals_host() {
        let stream = default_stream();
        for &((sw, sh), (dw, dh)) in &[((129, 97), (64, 48)), ((63, 41), (127, 90))] {
            let src = Image::<f32, 3>::new(sized(sw, sh), pattern_f32(sw * sh * 3)).unwrap();
            for mode in [InterpolationMode::Bilinear, InterpolationMode::Nearest] {
                let mut cpu_dst = Image::<f32, 3>::from_size_val(sized(dw, dh), 0.0).unwrap();
                resize(&src, &mut cpu_dst, mode).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut d_dst = Image::<f32, 3>::zeros_cuda(sized(dw, dh), &stream).unwrap();
                resize(&d_src, &mut d_dst, mode).unwrap();

                let back = d_dst.to_host_owned().unwrap();
                assert_eq!(
                    back.as_slice(),
                    cpu_dst.as_slice(),
                    "{sw}x{sh}->{dw}x{dh} {mode:?}: device must be bit-identical to host"
                );
            }
        }
    }

    /// Same-size device resize must run on the device (an identity-coefficient
    /// kernel pass), NOT fall into the host memcpy short-circuit — which would
    /// dereference device pointers on the host.
    #[test]
    fn same_size_device_resize_is_safe_and_identity() {
        let stream = default_stream();
        let src = Image::<f32, 3>::new(sized(33, 21), pattern_f32(33 * 21 * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(sized(33, 21), &stream).unwrap();
        resize(&d_src, &mut d_dst, InterpolationMode::Bilinear).unwrap();
        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(back.as_slice(), src.as_slice(), "identity resize");
    }

    /// Mixed residency errors; C != 3 device pairs error instead of silently
    /// falling back to the CPU.
    #[test]
    fn resize_error_semantics() {
        let stream = default_stream();
        let src = Image::<f32, 3>::new(sized(16, 16), pattern_f32(16 * 16 * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut host_dst = Image::<f32, 3>::from_size_val(sized(8, 8), 0.0).unwrap();
        let err = resize(&d_src, &mut host_dst, InterpolationMode::Bilinear).unwrap_err();
        assert!(matches!(err, ImageError::MixedResidency), "got {err:?}");

        let src1 = Image::<f32, 1>::new(sized(16, 16), pattern_f32(16 * 16)).unwrap();
        let d_src1 = src1.to_cuda(&stream).unwrap();
        let mut d_dst1 = Image::<f32, 1>::zeros_cuda(sized(8, 8), &stream).unwrap();
        let err = resize(&d_src1, &mut d_dst1, InterpolationMode::Bilinear).unwrap_err();
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("3-channel")),
            "got {err:?}"
        );
    }
}
