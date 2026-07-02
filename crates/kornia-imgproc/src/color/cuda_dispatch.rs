//! Residency-aware CUDA dispatch for the `ConvertColor` trait.
//!
//! Enabled by the `gpu-cuda` feature. The `impl_convert!` macro in
//! [`super::convert`] calls [`pair_residency`] on the operands: host pairs run
//! the existing CPU (NEON/AVX2) path unchanged, device pairs are routed to the
//! native CUDA launchers in [`crate::gpu::color_cuda`], and mixed pairs are a
//! typed error — there is **no implicit transfer** in either direction.
//!
//! The stream used for the launch is recovered from the *source* image
//! ([`kornia_tensor::Tensor::cuda_stream`]); device images therefore must be
//! created via the typed helpers (`to_cuda` / `zeros_cuda` on `Image` or the
//! color-space newtypes) so the storage is a typed `CudaResource<T>`.

use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use kornia_image::{Image, ImageError};
use kornia_tensor::MemoryDomain;

use crate::color::yuv::kernels::ChromaOrder;
use crate::gpu::color_cuda::{cie, gray, hsv_hls, misc, swizzle, yuv, CudaColorError};

/// Where a (src, dst) operand pair lives.
pub(crate) enum Residency<'a> {
    /// Both operands host-resident → run the CPU path.
    Host,
    /// Both operands device-resident → launch on the source's stream.
    Device(&'a Arc<CudaStream>),
}

/// True if the image's backing memory is device- (or unified-) resident.
///
/// Uses the storage's [`MemoryDomain`], so it is accurate even when the
/// element type does not match the stored `CudaResource<T>`.
pub(crate) fn is_device<T, const C: usize>(img: &Image<T, C>) -> bool {
    matches!(
        img.0.storage.domain(),
        MemoryDomain::Device { .. } | MemoryDomain::Unified { .. }
    )
}

/// Reject any device-resident operand — used by conversions that have no CUDA
/// kernel (yet), turning what would be an `as_slice` panic into a typed error.
pub(crate) fn ensure_host<T, const C: usize, const D: usize>(
    src: &Image<T, C>,
    dst: &Image<T, D>,
) -> Result<(), ImageError> {
    if is_device(src) || is_device(dst) {
        return Err(ImageError::UnsupportedDevice);
    }
    Ok(())
}

/// Classify a (src, dst) pair: both-host, both-device, or error on a mix.
pub(crate) fn pair_residency<'a, T, const C: usize, const D: usize>(
    src: &'a Image<T, C>,
    dst: &Image<T, D>,
) -> Result<Residency<'a>, ImageError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    match (is_device(src), is_device(dst)) {
        (false, false) => Ok(Residency::Host),
        (true, true) => src
            .0
            .cuda_stream()
            .map(Residency::Device)
            .ok_or_else(|| untyped_device_err("source")),
        _ => Err(ImageError::MixedResidency),
    }
}

fn untyped_device_err(what: &str) -> ImageError {
    ImageError::Cuda(format!(
        "{what} image is device-resident but not backed by a typed CudaResource; \
         create device images via Image::to_cuda_image / zeros_cuda (or the \
         color-space newtype to_cuda / zeros_cuda helpers)"
    ))
}

fn cuda_err(e: CudaColorError) -> ImageError {
    ImageError::Cuda(e.to_string())
}

fn check_same_size<T, const C: usize, const D: usize>(
    src: &Image<T, C>,
    dst: &Image<T, D>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    Ok(())
}

/// Extract the typed device slices from a checked device pair.
macro_rules! device_slices {
    ($src:expr, $dst:expr) => {{
        let s = $src
            .0
            .as_cudaslice()
            .ok_or_else(|| untyped_device_err("source"))?;
        let d = $dst
            .0
            .as_cudaslice_mut()
            .ok_or_else(|| untyped_device_err("destination"))?;
        (s, d)
    }};
}

/// Define a same-size elementwise adapter: size check → slice extraction →
/// launcher call on `npixels = w × h`.
macro_rules! adapter {
    ($(#[$meta:meta])* $name:ident, $t:ty, $cin:literal => $cout:literal, $launcher:path) => {
        $(#[$meta])*
        pub(crate) fn $name(
            src: &Image<$t, $cin>,
            dst: &mut Image<$t, $cout>,
            stream: &Arc<CudaStream>,
        ) -> Result<(), ImageError> {
            check_same_size(src, dst)?;
            let npixels = src.cols() * src.rows();
            let (s, d) = device_slices!(src, dst);
            $launcher(stream, s, d, npixels).map_err(cuda_err)
        }
    };
}

adapter!(gray_from_rgb_u8_cuda, u8, 3 => 1, gray::launch_gray_from_rgb_u8);
adapter!(gray_from_rgb_f32_cuda, f32, 3 => 1, gray::launch_gray_from_rgb_f32);
adapter!(rgb_from_gray_u8_cuda, u8, 1 => 3, gray::launch_rgb_from_gray_u8);
adapter!(rgb_from_gray_f32_cuda, f32, 1 => 3, gray::launch_rgb_from_gray_f32);
adapter!(
    /// Symmetric R/B swap — serves both RGB→BGR and BGR→RGB.
    bgr_from_rgb_u8_cuda, u8, 3 => 3, swizzle::launch_bgr_from_rgb_u8
);
adapter!(bgr_from_rgb_f32_cuda, f32, 3 => 3, swizzle::launch_bgr_from_rgb_f32);
adapter!(rgba_from_rgb_u8_cuda, u8, 3 => 4, swizzle::launch_rgba_from_rgb_u8);
adapter!(rgba_from_rgb_f32_cuda, f32, 3 => 4, swizzle::launch_rgba_from_rgb_f32);
adapter!(bgra_from_rgb_u8_cuda, u8, 3 => 4, swizzle::launch_bgra_from_rgb_u8);
adapter!(bgra_from_rgb_f32_cuda, f32, 3 => 4, swizzle::launch_bgra_from_rgb_f32);

adapter!(sepia_from_rgb_u8_cuda, u8, 3 => 3, misc::launch_sepia_from_rgb_u8);
adapter!(sepia_from_rgb_f32_cuda, f32, 3 => 3, misc::launch_sepia_from_rgb_f32);

adapter!(hsv_from_rgb_f32_cuda, f32, 3 => 3, hsv_hls::launch_hsv_from_rgb_f32);
adapter!(rgb_from_hsv_f32_cuda, f32, 3 => 3, hsv_hls::launch_rgb_from_hsv_f32);
adapter!(hls_from_rgb_f32_cuda, f32, 3 => 3, hsv_hls::launch_hls_from_rgb_f32);
adapter!(rgb_from_hls_f32_cuda, f32, 3 => 3, hsv_hls::launch_rgb_from_hls_f32);

adapter!(linear_rgb_from_rgb_f32_cuda, f32, 3 => 3, cie::launch_linear_rgb_from_rgb_f32);
adapter!(rgb_from_linear_rgb_f32_cuda, f32, 3 => 3, cie::launch_rgb_from_linear_rgb_f32);
adapter!(xyz_from_rgb_f32_cuda, f32, 3 => 3, cie::launch_xyz_from_rgb_f32);
adapter!(rgb_from_xyz_f32_cuda, f32, 3 => 3, cie::launch_rgb_from_xyz_f32);
adapter!(lab_from_rgb_f32_cuda, f32, 3 => 3, cie::launch_lab_from_rgb_f32);
adapter!(rgb_from_lab_f32_cuda, f32, 3 => 3, cie::launch_rgb_from_lab_f32);
adapter!(luv_from_rgb_f32_cuda, f32, 3 => 3, cie::launch_luv_from_rgb_f32);
adapter!(rgb_from_luv_f32_cuda, f32, 3 => 3, cie::launch_rgb_from_luv_f32);

/// Define a YCbCr/YUV-family adapter: fixes direction + chroma order.
macro_rules! ycc_adapter {
    ($name:ident, $t:ty, $launcher:path, $order:expr) => {
        pub(crate) fn $name(
            src: &Image<$t, 3>,
            dst: &mut Image<$t, 3>,
            stream: &Arc<CudaStream>,
        ) -> Result<(), ImageError> {
            check_same_size(src, dst)?;
            let npixels = src.cols() * src.rows();
            let (s, d) = device_slices!(src, dst);
            $launcher(stream, s, d, npixels, $order).map_err(cuda_err)
        }
    };
}

ycc_adapter!(
    ycbcr_from_rgb_u8_cuda,
    u8,
    yuv::launch_ycc_from_rgb_u8,
    ChromaOrder::YCrCb
);
ycc_adapter!(
    rgb_from_ycbcr_u8_cuda,
    u8,
    yuv::launch_rgb_from_ycc_u8,
    ChromaOrder::YCrCb
);
ycc_adapter!(
    yuv_from_rgb_u8_cuda,
    u8,
    yuv::launch_ycc_from_rgb_u8,
    ChromaOrder::YuvCbCr
);
ycc_adapter!(
    rgb_from_yuv_u8_cuda,
    u8,
    yuv::launch_rgb_from_ycc_u8,
    ChromaOrder::YuvCbCr
);
ycc_adapter!(
    ycbcr_from_rgb_f32_cuda,
    f32,
    yuv::launch_ycc_from_rgb_f32,
    ChromaOrder::YCrCb
);
ycc_adapter!(
    rgb_from_ycbcr_f32_cuda,
    f32,
    yuv::launch_rgb_from_ycc_f32,
    ChromaOrder::YCrCb
);
ycc_adapter!(
    yuv_from_rgb_f32_cuda,
    f32,
    yuv::launch_ycc_from_rgb_f32,
    ChromaOrder::YuvCbCr
);
ycc_adapter!(
    rgb_from_yuv_f32_cuda,
    f32,
    yuv::launch_rgb_from_ycc_f32,
    ChromaOrder::YuvCbCr
);

/// RGBA8/BGRA8 → RGB8 with optional background blend (shared body).
fn strip_alpha_u8_cuda(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    stream: &Arc<CudaStream>,
    swapped: bool,
    background: Option<[u8; 3]>,
) -> Result<(), ImageError> {
    check_same_size(src, dst)?;
    let npixels = src.cols() * src.rows();
    let (s, d) = device_slices!(src, dst);
    swizzle::launch_rgb_from_rgba_u8(stream, s, d, npixels, swapped, background).map_err(cuda_err)
}

/// RGBA8 → RGB8, alpha dropped (`bg: None` trait path).
pub(crate) fn rgb_from_rgba_u8_cuda(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    strip_alpha_u8_cuda(src, dst, stream, false, None)
}

/// BGRA8 → RGB8, alpha dropped (`bg: None` trait path).
pub(crate) fn rgb_from_bgra_u8_cuda(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    strip_alpha_u8_cuda(src, dst, stream, true, None)
}

/// RGBA8 → RGB8 with optional uniform background blend
/// (`ConvertColorWithBackground` device path).
pub(crate) fn rgb_from_rgba_bg_u8_cuda(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    background: Option<[u8; 3]>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    strip_alpha_u8_cuda(src, dst, stream, false, background)
}

/// BGRA8 → RGB8 with optional uniform background blend
/// (`ConvertColorWithBackground` device path).
pub(crate) fn rgb_from_bgra_bg_u8_cuda(
    src: &Image<u8, 4>,
    dst: &mut Image<u8, 3>,
    background: Option<[u8; 3]>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    strip_alpha_u8_cuda(src, dst, stream, true, background)
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod tests {
    use kornia_image::color_spaces::{Gray8, Grayf64, Rgb8, Rgbf64};
    use kornia_image::ImageSize;

    use crate::color::ConvertColor;
    use crate::gpu::color_cuda::test_utils::{default_stream, pattern_u8};

    const SIZE: ImageSize = ImageSize {
        width: 640,
        height: 480,
    };

    #[test]
    fn device_convert_matches_cpu_and_residency_rules() {
        let stream = default_stream();
        let n = SIZE.width * SIZE.height;
        let rgb = Rgb8::from_size_vec(SIZE, pattern_u8(n * 3)).unwrap();

        // CPU reference through the same trait path.
        let mut gray_cpu = Gray8::from_size_val(SIZE, 0).unwrap();
        rgb.convert(&mut gray_cpu).unwrap();

        // Device path: upload, convert, download.
        let rgb_d = rgb.to_cuda(&stream).unwrap();
        let mut gray_d = Gray8::zeros_cuda(SIZE, &stream).unwrap();
        rgb_d.convert(&mut gray_d).unwrap();
        let gray_back = gray_d.to_host(&stream).unwrap();

        assert_eq!(
            gray_back.as_slice(),
            gray_cpu.as_slice(),
            "device ConvertColor must match CPU bit-for-bit (u8 gray)"
        );

        // Mixed residency: device src + host dst → MixedResidency.
        let mut gray_host = Gray8::from_size_val(SIZE, 0).unwrap();
        let err = rgb_d.convert(&mut gray_host).unwrap_err();
        assert!(
            matches!(err, kornia_image::ImageError::MixedResidency),
            "mixed pair must fail with MixedResidency, got {err:?}"
        );
    }

    #[test]
    fn unported_op_on_device_errors_not_panics() {
        let stream = default_stream();
        // f64 gray has no CUDA kernel — its host-only arm must reject device
        // operands with UnsupportedDevice instead of panicking in as_slice.
        let rgb = Rgbf64::from_size_val(SIZE, 0.5).unwrap();
        let rgb_d = rgb.to_cuda(&stream).unwrap();
        let mut gray_d = Grayf64::zeros_cuda(SIZE, &stream).unwrap();
        let err = rgb_d.convert(&mut gray_d).unwrap_err();
        assert!(
            matches!(err, kornia_image::ImageError::UnsupportedDevice),
            "un-ported op with device operands must be UnsupportedDevice, got {err:?}"
        );
    }
}
