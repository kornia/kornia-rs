//! Device adapters for [`warp_affine`](super::warp_affine) and
//! [`warp_perspective`](super::warp_perspective): route checked device pairs
//! to the native CUDA warp kernels.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err};
use crate::cuda::warp_affine::{launch_warp_affine_bilinear_cuda, launch_warp_affine_nearest_cuda};
use crate::cuda::warp_perspective::{
    launch_warp_perspective_bilinear_cuda, launch_warp_perspective_nearest_cuda,
};
use crate::interpolation::InterpolationMode;

/// Run the CUDA warp-affine for a device-resident f32 pair.
///
/// `m` is the forward matrix — the launchers invert internally, exactly like
/// the CPU [`warp_affine`](super::warp_affine), so it passes straight
/// through. Output is bit-identical to the CPU path (the byte-exact contract
/// asserted by the parity tests in `cuda::warp_affine`).
pub(super) fn warp_affine_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    m: &[f32; 6],
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err("warp_affine", "3-channel f32 images"));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => launch_warp_affine_bilinear_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
        InterpolationMode::Nearest => {
            launch_warp_affine_nearest_cuda(ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None)
        }
        mode => return Err(ImageError::UnsupportedInterpolation(mode)),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

/// Run the CUDA warp-perspective for a device-resident f32 pair.
///
/// `m` is the forward homography — inverted internally like the CPU path.
/// Bit-identical to CPU [`warp_perspective`](super::warp_perspective).
pub(super) fn warp_perspective_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    m: &[f32; 9],
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err(
            "warp_perspective",
            "3-channel f32 images",
        ));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => launch_warp_perspective_bilinear_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
        InterpolationMode::Nearest => launch_warp_perspective_nearest_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
        mode => return Err(ImageError::UnsupportedInterpolation(mode)),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use crate::warp::{warp_affine, warp_perspective};
    use kornia_image::{Image, ImageError, ImageSize};

    fn host_pair(w: usize, h: usize) -> (Image<f32, 3>, Image<f32, 3>) {
        let size = ImageSize {
            width: w,
            height: h,
        };
        (
            Image::<f32, 3>::new(size, pattern_f32(w * h * 3)).unwrap(),
            Image::<f32, 3>::from_size_val(size, 0.0).unwrap(),
        )
    }

    /// The public `warp_affine`, called with device images, must produce
    /// bit-identical output to the same call with host images — the whole
    /// point of the byte-exact program.
    #[test]
    fn public_warp_affine_device_equals_host() {
        let stream = default_stream();
        let (src, mut cpu_dst) = host_pair(129, 97);
        let m = crate::warp::get_rotation_matrix2d((64.0, 48.0), 37.0, 1.3);

        warp_affine(&src, &mut cpu_dst, &m, InterpolationMode::Bilinear).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(src.size(), &stream).unwrap();
        warp_affine(&d_src, &mut d_dst, &m, InterpolationMode::Bilinear).unwrap();

        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(
            back.as_slice(),
            cpu_dst.as_slice(),
            "device warp_affine must be bit-identical to host"
        );
    }

    /// Same for the public `warp_perspective`, with a genuinely projective
    /// homography.
    #[test]
    fn public_warp_perspective_device_equals_host() {
        let stream = default_stream();
        let (src, mut cpu_dst) = host_pair(129, 97);
        let hm = [
            1.03,
            0.05,
            -3.0,
            -0.02,
            0.97,
            4.0,
            2.0 / (97.0 * 129.0),
            1.5 / (129.0 * 97.0),
            1.0,
        ];

        warp_perspective(&src, &mut cpu_dst, &hm, InterpolationMode::Bilinear).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(src.size(), &stream).unwrap();
        warp_perspective(&d_src, &mut d_dst, &hm, InterpolationMode::Bilinear).unwrap();

        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(
            back.as_slice(),
            cpu_dst.as_slice(),
            "device warp_perspective must be bit-identical to host"
        );
    }

    /// Mixed residency is a typed error, not an implicit transfer.
    #[test]
    fn mixed_residency_is_a_typed_error() {
        let stream = default_stream();
        let (src, mut host_dst) = host_pair(16, 16);
        let d_src = src.to_cuda(&stream).unwrap();
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = warp_affine(&d_src, &mut host_dst, &m, InterpolationMode::Nearest).unwrap_err();
        assert!(matches!(err, ImageError::MixedResidency), "got {err:?}");
    }

    /// A device pair with a channel count the kernels don't support errors —
    /// never a silent CPU fallback.
    #[test]
    fn unsupported_channels_error_not_fallback() {
        let stream = default_stream();
        let size = ImageSize {
            width: 16,
            height: 16,
        };
        let src = Image::<f32, 1>::new(size, pattern_f32(16 * 16)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 1>::zeros_cuda(size, &stream).unwrap();
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = warp_affine(&d_src, &mut d_dst, &m, InterpolationMode::Nearest).unwrap_err();
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("3-channel")),
            "got {err:?}"
        );
    }
}
