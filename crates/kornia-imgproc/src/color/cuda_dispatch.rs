//! Residency-aware CUDA dispatch for the `ConvertColor` trait.
//!
//! Enabled by the `cuda` feature. The `impl_convert!` macro in
//! [`super::convert`] calls [`pair_residency`] on the operands: host pairs run
//! the existing CPU (NEON/AVX2) path unchanged, device pairs are routed to the
//! native CUDA launchers in [`crate::cuda::color`], and mixed pairs are a
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
use crate::cuda::color::{cie, gray, hsv_hls, misc, swizzle, yuv};

/// Where a (src, dst) operand pair lives.
pub(crate) enum Residency {
    /// Both operands host-resident → run the CPU path.
    Host,
    /// Both operands device-resident → launch through the [`DeviceExec`].
    Device(DeviceExec),
}

/// A checked device execution context: the stream to launch on plus the
/// cross-stream fence obligations when src and dst live on different streams.
///
/// CUDA gives no implicit ordering between streams, so for a cross-stream
/// pair [`pair_residency`] records an event on the destination's stream and
/// makes the launch stream wait on it (the destination's pending writes —
/// e.g. `zeros_cuda`'s async memset — complete first), and [`finish`]
/// records the launch on an event the destination's stream then waits on
/// (subsequent destination-stream reads see the converted pixels).
pub(crate) struct DeviceExec {
    stream: Arc<CudaStream>,
    /// Destination stream to fence back to, when different from `stream`.
    fence_back: Option<Arc<CudaStream>>,
}

impl DeviceExec {
    /// Build the execution context for a (launch, destination) stream pair —
    /// the single home of the cross-stream fence protocol. Same device
    /// required; same stream = no fence; different streams = pre-fence now
    /// (launch stream waits for the destination's pending work) and a
    /// post-fence obligation discharged by [`run`](Self::run).
    fn for_streams(launch: &Arc<CudaStream>, dst: &Arc<CudaStream>) -> Result<Self, ImageError> {
        if launch.context().ordinal() != dst.context().ordinal() {
            return Err(ImageError::DeviceMismatch);
        }
        if launch.cu_stream() == dst.cu_stream() {
            return Ok(DeviceExec {
                stream: launch.clone(),
                fence_back: None,
            });
        }
        let ev = dst.record_event(None).map_err(driver_err)?;
        launch.wait(&ev).map_err(driver_err)?;
        Ok(DeviceExec {
            stream: launch.clone(),
            fence_back: Some(dst.clone()),
        })
    }

    /// Launch through `f` and complete the cross-stream ordering — fusing the
    /// launch and the post-fence so forgetting the fence is unrepresentable.
    pub(crate) fn run(
        self,
        f: impl FnOnce(&Arc<CudaStream>) -> Result<(), ImageError>,
    ) -> Result<(), ImageError> {
        f(&self.stream)?;
        if let Some(dst_stream) = self.fence_back {
            let ev = self.stream.record_event(None).map_err(driver_err)?;
            dst_stream.wait(&ev).map_err(driver_err)?;
        }
        Ok(())
    }
}

fn driver_err(e: cudarc::driver::DriverError) -> ImageError {
    ImageError::Cuda(e.to_string())
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

/// Classify a (src, dst) pair: both-host, both-device, or error on a mix.
///
/// Device pairs must live on the **same device** (cross-device errors with
/// [`ImageError::DeviceMismatch`]). Different **streams** on the same device
/// are supported via event fences — see [`DeviceExec`]. Streams are compared
/// by raw `CUstream` handle (every `ctx.default_stream()` call returns a
/// fresh `Arc` over the same null handle, so `Arc` identity is wrong).
pub(crate) fn pair_residency<T, const C: usize, const D: usize>(
    src: &Image<T, C>,
    dst: &Image<T, D>,
) -> Result<Residency, ImageError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    match (is_device(src), is_device(dst)) {
        (false, false) => Ok(Residency::Host),
        (true, true) => {
            let s_stream = src
                .0
                .cuda_stream()
                .ok_or_else(|| untyped_device_err("source"))?;
            let d_stream = dst
                .0
                .cuda_stream()
                .ok_or_else(|| untyped_device_err("destination"))?;
            Ok(Residency::Device(DeviceExec::for_streams(
                s_stream, d_stream,
            )?))
        }
        _ => Err(ImageError::MixedResidency),
    }
}

/// Build a [`DeviceExec`] for a known-device source stream and a destination
/// image (used by non-`Image` sources like `DeviceVideoFrame`). Same
/// same-device + event-fence rules as [`pair_residency`].
pub(crate) fn device_exec_for<T>(
    src_stream: &Arc<CudaStream>,
    dst: &kornia_tensor::Tensor<T, 3>,
) -> Result<DeviceExec, ImageError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    let d_stream = dst
        .cuda_stream()
        .ok_or_else(|| untyped_device_err("destination"))?;
    DeviceExec::for_streams(src_stream, d_stream)
}

fn untyped_device_err(what: &str) -> ImageError {
    ImageError::Cuda(format!(
        "{what} image is device-resident but not backed by a typed CudaResource; \
         create device images via Image::to_cuda / zeros_cuda (or the \
         color-space newtype to_cuda / zeros_cuda helpers)"
    ))
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
            $launcher(stream, s, d, npixels).map_err(ImageError::from)
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

adapter!(gray_from_rgb_f64_cuda, f64, 3 => 1, gray::launch_gray_from_rgb_f64);
adapter!(rgb_from_gray_f64_cuda, f64, 1 => 3, gray::launch_rgb_from_gray_f64);
adapter!(hsv_from_rgb_f64_cuda, f64, 3 => 3, hsv_hls::launch_hsv_from_rgb_f64);
adapter!(rgb_from_hsv_f64_cuda, f64, 3 => 3, hsv_hls::launch_rgb_from_hsv_f64);
adapter!(hls_from_rgb_f64_cuda, f64, 3 => 3, hsv_hls::launch_hls_from_rgb_f64);
adapter!(rgb_from_hls_f64_cuda, f64, 3 => 3, hsv_hls::launch_rgb_from_hls_f64);
adapter!(linear_rgb_from_rgb_f64_cuda, f64, 3 => 3, cie::launch_linear_rgb_from_rgb_f64);
adapter!(rgb_from_linear_rgb_f64_cuda, f64, 3 => 3, cie::launch_rgb_from_linear_rgb_f64);
adapter!(xyz_from_rgb_f64_cuda, f64, 3 => 3, cie::launch_xyz_from_rgb_f64);
adapter!(rgb_from_xyz_f64_cuda, f64, 3 => 3, cie::launch_rgb_from_xyz_f64);
adapter!(lab_from_rgb_f64_cuda, f64, 3 => 3, cie::launch_lab_from_rgb_f64);
adapter!(rgb_from_lab_f64_cuda, f64, 3 => 3, cie::launch_rgb_from_lab_f64);
adapter!(luv_from_rgb_f64_cuda, f64, 3 => 3, cie::launch_luv_from_rgb_f64);
adapter!(rgb_from_luv_f64_cuda, f64, 3 => 3, cie::launch_rgb_from_luv_f64);

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
            $launcher(stream, s, d, npixels, $order).map_err(ImageError::from)
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
    ycbcr_from_rgb_f64_cuda,
    f64,
    yuv::launch_ycc_from_rgb_f64,
    ChromaOrder::YCrCb
);
ycc_adapter!(
    rgb_from_ycbcr_f64_cuda,
    f64,
    yuv::launch_rgb_from_ycc_f64,
    ChromaOrder::YCrCb
);
ycc_adapter!(
    yuv_from_rgb_f64_cuda,
    f64,
    yuv::launch_ycc_from_rgb_f64,
    ChromaOrder::YuvCbCr
);
ycc_adapter!(
    rgb_from_yuv_f64_cuda,
    f64,
    yuv::launch_rgb_from_ycc_f64,
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

/// Bayer mosaic → RGB8 demosaic (device path of `rgb_from_bayer`).
pub(crate) fn rgb_from_bayer_u8_cuda(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 3>,
    pattern: kornia_image::color_spaces::BayerPattern,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    check_same_size(src, dst)?;
    let (rows, cols) = (src.rows(), src.cols());
    let (s, d) = device_slices!(src, dst);
    crate::cuda::color::bayer::launch_rgb_from_bayer_u8(stream, s, d, rows, cols, pattern)
        .map_err(ImageError::from)
}

/// Gray8 → RGB8 colormap application (device path of `apply_colormap`).
pub(crate) fn apply_colormap_u8_cuda(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 3>,
    colormap: crate::color::ColormapType,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    check_same_size(src, dst)?;
    let npixels = src.cols() * src.rows();
    let (s, d) = device_slices!(src, dst);
    misc::launch_apply_colormap_u8(stream, s, d, npixels, colormap).map_err(ImageError::from)
}

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
    swizzle::launch_rgb_from_rgba_u8(stream, s, d, npixels, swapped, background)
        .map_err(ImageError::from)
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use kornia_image::color_spaces::{Gray8, Grayf64, Rgb8, Rgbf64};
    use kornia_image::ImageSize;

    use crate::color::ConvertColor;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};

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
    fn cross_stream_pairs_are_event_fenced() {
        use cudarc::driver::CudaContext;

        let ctx = CudaContext::new(0).unwrap();
        let s1 = ctx.default_stream();
        let s2 = ctx.new_stream().unwrap();

        let rgb = Rgb8::from_size_vec(SIZE, pattern_u8(SIZE.width * SIZE.height * 3)).unwrap();
        let mut gray_cpu = Gray8::from_size_val(SIZE, 0).unwrap();
        rgb.convert(&mut gray_cpu).unwrap();

        // src on the default stream, dst on an explicit stream: the dispatch
        // must fence (dst's async zero-memset before the kernel; the kernel
        // before dst-stream reads) and produce the exact CPU result.
        let rgb_d = rgb.to_cuda(&s1).unwrap();
        let mut gray_other = Gray8::zeros_cuda(SIZE, &s2).unwrap();
        rgb_d.convert(&mut gray_other).unwrap();
        let back = gray_other.to_host(&s2).unwrap();
        assert_eq!(
            back.as_slice(),
            gray_cpu.as_slice(),
            "cross-stream fenced convert must match CPU bit-for-bit"
        );

        // Two separate default_stream() Arcs are the SAME stream (null handle)
        // — no fence needed, must dispatch fine.
        let mut gray_default = Gray8::zeros_cuda(SIZE, &ctx.default_stream()).unwrap();
        rgb_d.convert(&mut gray_default).unwrap();
    }

    #[test]
    fn f64_device_conversions_match_cpu_oracle() {
        use kornia_image::color_spaces::{Hsvf64, Labf64, YCbCrf64};

        let stream = default_stream();
        let n = SIZE.width * SIZE.height;
        // f64 RGB in [0,1].
        let data: Vec<f64> = pattern_u8(n * 3)
            .iter()
            .map(|&b| b as f64 / 255.0)
            .collect();
        let rgb = Rgbf64::from_size_vec(SIZE, data).unwrap();
        let rgb_d = rgb.to_cuda(&stream).unwrap();

        // gray (exact-form weighted sum) — tight tolerance.
        let mut g_cpu = Grayf64::from_size_val(SIZE, 0.0).unwrap();
        rgb.convert(&mut g_cpu).unwrap();
        let mut g_d = Grayf64::zeros_cuda(SIZE, &stream).unwrap();
        rgb_d.convert(&mut g_d).unwrap();
        let g_back = g_d.to_host_owned().unwrap();
        let diff = g_back
            .as_slice()
            .iter()
            .zip(g_cpu.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        assert!(diff <= 1e-12, "f64 gray max diff {diff}");

        // ycbcr / hsv / lab within 1e-6 (constant-representation and libm
        // differences dominate; still far tighter than the f32 paths).
        macro_rules! check {
            ($dst_ty:ty, $tol:expr, $label:literal) => {{
                let mut cpu = <$dst_ty>::from_size_val(SIZE, 0.0).unwrap();
                rgb.convert(&mut cpu).unwrap();
                let mut dev = <$dst_ty>::zeros_cuda(SIZE, &stream).unwrap();
                rgb_d.convert(&mut dev).unwrap();
                let back = dev.to_host_owned().unwrap();
                let diff = back
                    .as_slice()
                    .iter()
                    .zip(cpu.as_slice())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max);
                assert!(diff <= $tol, "{} f64 max diff {diff}", $label);
            }};
        }
        check!(YCbCrf64, 1e-12, "ycbcr");
        check!(Hsvf64, 1e-6, "hsv");
        check!(Labf64, 1e-6, "lab");
    }
}
