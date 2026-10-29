//! Depth-image filtering: a median blur that ignores missing measurements,
//! and detection / removal of flying pixels at depth discontinuities.
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
//! Invalid samples never take part in a median or a comparison. Where an op
//! produces an invalid sample (a median window with no valid tap, a masked
//! pixel) it writes the canonical marker `0` (`u16`) or `NaN` (`f32`);
//! [`invalidate_masked_depth`] copies unmasked pixels, holes included,
//! unchanged.
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

    #[doc(hidden)]
    fn flying_pixel_mask_impl(
        src: &Image<Self, 1>,
        mask: &mut Image<u8, 1>,
        abs_threshold: f32,
        rel_threshold: f32,
        min_valid: usize,
    ) -> Result<(), ImageError>;

    #[doc(hidden)]
    fn invalidate_masked_depth_impl(
        src: &Image<Self, 1>,
        mask: &Image<u8, 1>,
        dst: &mut Image<Self, 1>,
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

/// Flag flying pixels: valid depth samples that disagree with the valid
/// samples around them.
///
/// For every valid pixel `d` ([`is_valid_depth_u16`] /
/// [`is_valid_depth_f32`]), let `m` be the median of the valid samples in
/// its 3×3 window (the pixel itself included, same median as
/// [`median_blur_depth`] with `ksize = 3`) and `n` their count. The pixel is
/// flagged (`255`) when
///
/// * `n < min_valid` (too little support), or
/// * `|d - m| > max(abs_threshold, rel_threshold * d)`.
///
/// Every other pixel, including every invalid one (already missing, so not
/// a flying pixel), gets `0`.
///
/// A clean step between two surfaces flags nothing: on either side of the
/// step the 3×3 window majority lies on the pixel's own surface. A pixel
/// whose value lies between the two surfaces (a mixed measurement), or a
/// one-pixel spike, is flagged.
///
/// Reference: this is the median-deviation rule commonly used to remove
/// flying pixels from depth maps (compare a sample with a robust local
/// estimate and reject large depth-relative deviations); there is no single
/// library function to match, so the rule above is the specification.
/// Borders replicate, so a corner pixel counts its own value four times in
/// `n` (see the module docs).
///
/// # Arguments
///
/// * `src` - Single-channel depth image, `u16` or `f32`, in the caller's
///   depth units.
/// * `mask` - Pre-allocated `u8` output with the same size as `src`;
///   `255` = flagged, `0` = kept.
/// * `abs_threshold` - Absolute deviation limit, in depth units; `>= 0`.
/// * `rel_threshold` - Deviation limit as a fraction of the pixel's depth
///   (e.g. `0.05` for 5 %); `>= 0`.
/// * `min_valid` - Minimum number of valid taps (of 9, replicated taps
///   included) a valid pixel needs to be kept; `0` or `1` disables the
///   test, values above 9 flag every valid pixel.
///
/// A threshold of `0.0` lets the other one decide alone; `f32::INFINITY`
/// in either disables the deviation test (only `min_valid` flags then).
///
/// # Returns
///
/// `Ok(())` on success; `mask` holds the flags.
/// An image with zero width or height is a valid no-op.
///
/// # Errors
///
/// * [`ImageError::InvalidImageSize`] if `src` and `mask` sizes differ.
/// * [`ImageError::InvalidThreshold`] if `abs_threshold` or
///   `rel_threshold` is NaN or negative (either would silently disable the
///   deviation test), on host and device alike.
/// * [`ImageError::MixedResidency`] / [`ImageError::Cuda`] for device images
///   (feature `cuda`) on mixed residency or a launch failure.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::depth::flying_pixel_mask;
///
/// let size = ImageSize { width: 3, height: 3 };
/// // A 1000 mm surface with one sample stranded at 1500 mm.
/// let mut depth = vec![1000u16; 9];
/// depth[4] = 1500;
/// let src = Image::<u16, 1>::new(size, depth)?;
/// let mut mask = Image::<u8, 1>::from_size_val(size, 0)?;
/// flying_pixel_mask(&src, &mut mask, 40.0, 0.025, 4)?;
/// assert_eq!(mask.as_slice()[4], 255);
/// assert_eq!(mask.as_slice().iter().filter(|&&v| v != 0).count(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn flying_pixel_mask<T: DepthPixel>(
    src: &Image<T, 1>,
    mask: &mut Image<u8, 1>,
    abs_threshold: f32,
    rel_threshold: f32,
    min_valid: usize,
) -> Result<(), ImageError> {
    T::flying_pixel_mask_impl(src, mask, abs_threshold, rel_threshold, min_valid)
}

/// Remove flagged pixels from a depth image.
///
/// `dst` is `src` with every pixel whose `mask` value is non-zero replaced
/// by the invalid marker: `0` for `u16` and `NaN` for `f32` (`NaN` rather
/// than `0.0` so the hole is unmistakable to consumers that only test for
/// `NaN`; both are invalid under [`is_valid_depth_f32`]). Unmasked pixels
/// are copied unchanged, so an input hole keeps its original encoding.
///
/// # Arguments
///
/// * `src` - Single-channel depth image, `u16` or `f32`.
/// * `mask` - Mask with the same size as `src`, e.g. from
///   [`flying_pixel_mask`]; any non-zero value removes the pixel.
/// * `dst` - Pre-allocated output with the same size as `src`.
///
/// # Returns
///
/// `Ok(())` on success; `dst` holds the cleaned depth image.
/// An image with zero width or height is a valid no-op.
///
/// # Errors
///
/// * [`ImageError::InvalidImageSize`] if `mask` or `dst` differ in size from
///   `src`.
/// * [`ImageError::MixedResidency`] / [`ImageError::Cuda`] for device images
///   (feature `cuda`) when the three images are not all host or all device,
///   or on a launch failure.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::depth::invalidate_masked_depth;
///
/// let size = ImageSize { width: 2, height: 1 };
/// let src = Image::<f32, 1>::new(size, vec![1.0, 2.0])?;
/// let mask = Image::<u8, 1>::new(size, vec![0, 255])?;
/// let mut dst = Image::<f32, 1>::from_size_val(size, 0.0)?;
/// invalidate_masked_depth(&src, &mask, &mut dst)?;
/// assert_eq!(dst.as_slice()[0], 1.0);
/// assert!(dst.as_slice()[1].is_nan());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn invalidate_masked_depth<T: DepthPixel>(
    src: &Image<T, 1>,
    mask: &Image<u8, 1>,
    dst: &mut Image<T, 1>,
) -> Result<(), ImageError> {
    T::invalidate_masked_depth_impl(src, mask, dst)
}

/// `Ok` for a usable threshold: not NaN and `>= 0` (`+inf` allowed).
fn check_threshold(name: &'static str, value: f32) -> Result<(), ImageError> {
    // NaN fails the comparison.
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ImageError::InvalidThreshold { name, value })
    }
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

            fn flying_pixel_mask_impl(
                src: &Image<$t, 1>,
                mask: &mut Image<u8, 1>,
                abs_threshold: f32,
                rel_threshold: f32,
                min_valid: usize,
            ) -> Result<(), ImageError> {
                check_size(src, mask)?;
                check_threshold("abs_threshold", abs_threshold)?;
                check_threshold("rel_threshold", rel_threshold)?;
                crate::try_device!(src, mask, |stream| cuda_adapters::flying_pixel_mask(
                    src,
                    mask,
                    abs_threshold,
                    rel_threshold,
                    min_valid,
                    stream
                ));
                let (w, h) = (src.cols(), src.rows());
                kernels::flying_pixel_mask::<$t>(
                    src.as_slice(),
                    mask.as_slice_mut(),
                    w,
                    h,
                    abs_threshold,
                    rel_threshold,
                    min_valid,
                );
                Ok(())
            }

            fn invalidate_masked_depth_impl(
                src: &Image<$t, 1>,
                mask: &Image<u8, 1>,
                dst: &mut Image<$t, 1>,
            ) -> Result<(), ImageError> {
                check_size(src, mask)?;
                check_size(src, dst)?;
                #[cfg(feature = "cuda")]
                if let crate::cuda::dispatch::Residency::Device(exec) =
                    crate::cuda::dispatch::pair_residency(src, dst)?
                {
                    return exec.run(|stream| {
                        cuda_adapters::invalidate_masked_depth(src, mask, dst, stream)
                    });
                }
                #[cfg(feature = "cuda")]
                if crate::cuda::dispatch::is_device(mask) {
                    return Err(ImageError::MixedResidency);
                }
                kernels::remove_flagged::<$t>(src.as_slice(), mask.as_slice(), dst.as_slice_mut());
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

    use crate::cuda::depth::{
        launch_flying_pixel_mask, launch_median_blur_depth, launch_remove_flagged, CudaDepthPixel,
    };
    use crate::cuda::dispatch::{device_slices, is_device, untyped_device_err};

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

    pub(super) fn flying_pixel_mask<T: CudaDepthPixel>(
        src: &Image<T, 1>,
        mask: &mut Image<u8, 1>,
        abs_threshold: f32,
        rel_threshold: f32,
        min_valid: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, m) = device_slices!(src, mask);
        launch_flying_pixel_mask(
            ctx,
            stream,
            s,
            m,
            src.cols(),
            src.rows(),
            abs_threshold,
            rel_threshold,
            min_valid,
        )
        .map_err(err)
    }

    /// `src` / `dst` are a checked device pair launching on `stream`; the
    /// mask is the third operand and is fenced onto `stream` here when it
    /// lives on another stream of the same device.
    pub(super) fn invalidate_masked_depth<T: CudaDepthPixel>(
        src: &Image<T, 1>,
        mask: &Image<u8, 1>,
        dst: &mut Image<T, 1>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        if !is_device(mask) {
            return Err(ImageError::MixedResidency);
        }
        let mask_stream = mask
            .0
            .cuda_stream()
            .ok_or_else(|| untyped_device_err("mask"))?;
        if mask_stream.context().ordinal() != stream.context().ordinal() {
            return Err(ImageError::DeviceMismatch);
        }
        if mask_stream.cu_stream() != stream.cu_stream() {
            let ev = mask_stream.record_event(None).map_err(err)?;
            stream.wait(&ev).map_err(err)?;
        }
        let m = mask
            .0
            .as_cudaslice()
            .ok_or_else(|| untyped_device_err("mask"))?;
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_remove_flagged(ctx, stream, s, m, d, src.cols(), src.rows()).map_err(err)
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
