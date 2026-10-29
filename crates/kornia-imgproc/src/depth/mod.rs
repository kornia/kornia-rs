//! Depth-image filtering: a median blur that ignores missing measurements.
//!
//! # Valid depth
//!
//! One rule decides which samples are measurements, for every op here, on
//! CPU and CUDA alike, and is public so other crates can apply the same
//! rule ([`is_valid_depth_u16`], [`is_valid_depth_f32`]):
//!
//! * `u16`: valid iff `d != 0`.
//! * `f32`: valid iff `d.is_finite() && d > 0.0`. `0.0`, `-0.0`, negative
//!   values, `±inf` and `NaN` are all missing measurements.
//!
//! Invalid samples never take part in a median. Where an op produces an
//! invalid sample (a median window with no valid tap) it writes the
//! canonical marker `0` (`u16`) or `NaN` (`f32`).
//!
//! Borders replicate (`BORDER_REPLICATE`, as OpenCV `medianBlur` does):
//! out-of-image taps repeat the nearest edge pixel, so a window always has
//! `ksize²` taps and a replicated tap is counted as often as it appears.

pub(crate) mod kernels;

use kornia_image::{Image, ImageError};

use crate::color::kernel_common::{check_size, sealed};

/// True when a `u16` depth sample is a measurement: `d != 0`.
///
/// The validity rule of every `u16` op in this module (see the module
/// docs). `0` is the missing-measurement marker that depth sensors write
/// where they measured nothing.
///
/// # Arguments
///
/// * `d` - Depth sample, in the caller's depth units.
///
/// # Returns
///
/// `true` if `d` is a measurement, `false` if it is missing.
///
/// # Errors
///
/// This function is infallible.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::depth::is_valid_depth_u16;
///
/// assert!(is_valid_depth_u16(1200));
/// assert!(!is_valid_depth_u16(0));
/// ```
#[inline]
pub fn is_valid_depth_u16(d: u16) -> bool {
    d != 0
}

/// True when an `f32` depth sample is a measurement: finite and `> 0`.
///
/// The validity rule of every `f32` op in this module (see the module
/// docs). `NaN`, `±inf`, `0.0`, `-0.0` and negative values are missing
/// measurements: none of them is a physical depth, and treating them as
/// values would let a hole marker (`0.0` or `NaN`, depending on the
/// producer) or an overflow (`inf`) leak into medians and comparisons.
///
/// # Arguments
///
/// * `d` - Depth sample, in the caller's depth units.
///
/// # Returns
///
/// `true` if `d` is a measurement, `false` if it is missing.
///
/// # Errors
///
/// This function is infallible.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::depth::is_valid_depth_f32;
///
/// assert!(is_valid_depth_f32(1.25));
/// for missing in [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
///     assert!(!is_valid_depth_f32(missing));
/// }
/// ```
#[inline]
pub fn is_valid_depth_f32(d: f32) -> bool {
    d.is_finite() && d > 0.0
}

/// Depth pixel types: `u16` (valid iff [`is_valid_depth_u16`], written
/// invalid marker `0`) and `f32` (valid iff [`is_valid_depth_f32`], written
/// invalid marker `NaN`). Sealed.
pub trait DepthPixel: sealed::Sealed + Copy + Send + Sync + 'static {
    #[doc(hidden)]
    fn median_blur_depth_impl<const C: usize>(
        src: &Image<Self, C>,
        dst: &mut Image<Self, C>,
        ksize: usize,
    ) -> Result<(), ImageError>;
}

/// Median blur of a depth image that excludes missing measurements.
///
/// Each output sample is the median of the **valid** samples
/// ([`is_valid_depth_u16`] / [`is_valid_depth_f32`]) in the `ksize × ksize`
/// window around it, per channel. A pixel whose window holds no valid
/// sample is written as the invalid marker (`0` / `NaN`); an invalid pixel
/// with valid neighbours is filled with their median. Every output sample
/// is therefore either valid or the canonical marker: an `f32` input hole
/// encoded as `0.0`, a negative value or `±inf` comes out as `NaN` or as a
/// filled value, never as itself.
///
/// Reference: OpenCV `cv::medianBlur` (`CV_16U` / `CV_32F`, ksize 3 or 5,
/// `BORDER_REPLICATE`). On an image with no invalid sample the output is
/// identical to `cv2.medianBlur` (pinned in the tests). Departures from
/// OpenCV:
///
/// * OpenCV treats `0` (and, for `f32`, negative and infinite values) as
///   values and leaves NaN behaviour unspecified; here all of them are
///   missing measurements and are excluded from the window.
/// * With an even number `n` of valid samples the upper median (rank
///   `n / 2` of the sorted valid samples) is returned, not the mean of the
///   two middle values: the output is always one of the input samples, so
///   the filter never invents a depth between two surfaces.
///
/// The `u8` [`crate::filter::median_blur`] is unchanged by this op.
///
/// # Arguments
///
/// * `src` - Depth image, `u16` or `f32`, any channel count (each channel is
///   filtered independently).
/// * `dst` - Pre-allocated output with the same size as `src`.
/// * `ksize` - Window side, 3 or 5.
///
/// # Returns
///
/// `Ok(())` on success; `dst` holds the filtered image.
/// An image with zero width or height is a valid no-op.
///
/// # Errors
///
/// * [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
/// * [`ImageError::InvalidKernelLength`] if `ksize` is not 3 or 5.
/// * [`ImageError::MixedResidency`] / [`ImageError::Cuda`] for device images
///   (feature `cuda`) on mixed residency or a launch failure.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::depth::median_blur_depth;
///
/// let size = ImageSize { width: 3, height: 3 };
/// // Centre is a missing measurement, two neighbours are missing too.
/// let src = Image::<u16, 1>::new(size, vec![10, 0, 12, 11, 0, 13, 0, 14, 15])?;
/// let mut dst = Image::<u16, 1>::from_size_val(size, 0)?;
/// median_blur_depth(&src, &mut dst, 3)?;
/// // Valid samples around the centre: 10, 11, 12, 13, 14, 15 -> rank 3.
/// assert_eq!(dst.as_slice()[4], 13);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn median_blur_depth<T: DepthPixel, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    ksize: usize,
) -> Result<(), ImageError> {
    T::median_blur_depth_impl(src, dst, ksize)
}

macro_rules! impl_depth_pixel {
    ($t:ty) => {
        impl DepthPixel for $t {
            fn median_blur_depth_impl<const C: usize>(
                src: &Image<$t, C>,
                dst: &mut Image<$t, C>,
                ksize: usize,
            ) -> Result<(), ImageError> {
                check_size(src, dst)?;
                if ksize != 3 && ksize != 5 {
                    return Err(ImageError::InvalidKernelLength(ksize, ksize));
                }
                crate::try_device!(src, dst, |stream| cuda_adapters::median_blur_depth(
                    src, dst, ksize, stream
                ));
                let (w, h) = (src.cols(), src.rows());
                kernels::median_blur_depth::<$t, C>(
                    src.as_slice(),
                    dst.as_slice_mut(),
                    w,
                    h,
                    ksize,
                );
                Ok(())
            }
        }
    };
}

impl_depth_pixel!(u16);
impl_depth_pixel!(f32);

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use std::sync::Arc;

    use cudarc::driver::CudaStream;
    use kornia_image::{Image, ImageError};

    use crate::cuda::depth::{launch_median_blur_depth, CudaDepthPixel};
    use crate::cuda::dispatch::{device_slices, untyped_device_err};

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn median_blur_depth<T: CudaDepthPixel, const C: usize>(
        src: &Image<T, C>,
        dst: &mut Image<T, C>,
        ksize: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_median_blur_depth(ctx, stream, s, d, src.cols(), src.rows(), C, ksize).map_err(err)
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
