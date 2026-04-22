//! Image resize — dispatcher + per-algorithm kernels.
//!
//! Per-arch row primitives live in [`kernels`]; the algorithm files above
//! it (bilinear, nearest, pyramid, separable, fused) stay free of
//! `#[cfg(target_arch)]` pairs and call into kernels through a thin seam.
//!
//! # Dispatch cascade (u8 path, in [`resize_fast_u8_aa`])
//!
//! 1. Bilinear + C=3 + exact 2× downscale → [`pyramid::pyrdown_2x_rgb_u8`].
//! 2. Bilinear + C=3 + exact 2× upscale   → [`pyramid::pyrup_2x_rgb_u8`].
//! 3. Nearest (any C, any ratio)          → [`nearest::resize_nearest_u8`].
//! 4. Bilinear (C ∈ {1, 3, 4})            → [`bilinear::resize_bilinear_u8_nch`].
//! 5. Bicubic / Lanczos                   → [`separable::resize_separable_u8`]
//!    (with `antialias` controlling PIL vs OpenCV kernel semantics).

mod bilinear;
mod common;
mod fused;
mod kernels;
mod nearest;
mod pyramid;
mod separable;

pub use fused::{resize_normalize_to_tensor_u8_to_f32, NormalizeParams};

use crate::{
    interpolation::{
        grid::meshgrid_from_fn, interpolate_pixel_fast, validate_interpolation, InterpolationMode,
    },
    parallel,
};
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

use common::FilterKind;

/// Resize an image to a new size.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports any number of channels and data types.
///
/// # Arguments
///
/// * `src` - The input image container.
/// * `dst` - The output image container.
/// * `optional_args` - Optional arguments for the resize operation.
///
/// # Returns
///
/// The resized image with the new size.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::resize::resize_native;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///     width: 2,
///     height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
///
/// resize_native(
///     &image,
///     &mut image_resized,
///     InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_native<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;

    // Short-circuit when the resize is a no-op.
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());
    let step_x = if dst.cols() > 1 {
        (src.cols() - 1) as f32 / (dst.cols() - 1) as f32
    } else {
        0.0
    };
    let step_y = if dst.rows() > 1 {
        (src.rows() - 1) as f32 / (dst.rows() - 1) as f32
    } else {
        0.0
    };
    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        Ok((x as f32 * step_x, y as f32 * step_y))
    })?;

    parallel::par_iter_rows_resample(dst, &map_x, &map_y, |&x, &y, dst_pixel| {
        for (k, pixel) in dst_pixel.iter_mut().enumerate() {
            *pixel = interpolate_pixel_fast(src, x, y, k, interpolation);
        }
    });

    Ok(())
}

/// Resize a 3-channel u8 image. Convenience wrapper around [`resize_fast_u8`].
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::resize::resize_fast_rgb;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3, _>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0u8; 4 * 5 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///   width: 2,
///   height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0, CpuAllocator).unwrap();
///
/// resize_fast_rgb(
///   &image,
///   &mut image_resized,
///   InterpolationMode::Nearest,
/// )
/// .unwrap();
///
/// assert_eq!(image_resized.num_channels(), 3);
/// assert_eq!(image_resized.size().width, 2);
/// assert_eq!(image_resized.size().height, 3);
/// ```
pub fn resize_fast_rgb<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 3, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8::<3, _, _>(src, dst, interpolation)
}

/// Generic fast u8 resize for 1/3/4-channel images (PIL-quality antialiased downscale).
pub fn resize_fast_u8<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<C, _, _>(src, dst, interpolation, true)
}

/// Generic fast u8 resize with explicit antialias control.
///
/// `antialias=true` (default via [`resize_fast_u8`]) matches PIL / torchvision
/// `antialias=True` — the cubic/lanczos kernel is widened by the downscale
/// factor to pre-filter aliasing.
///
/// `antialias=false` matches OpenCV `INTER_CUBIC` / `INTER_LANCZOS4` semantics
/// — fixed 4-tap / 8-tap kernel regardless of scale. Much faster at strong
/// downscale but does not anti-alias.
///
/// `Nearest` and `Bilinear` are unaffected by this flag.
pub fn resize_fast_u8_aa<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    let (src_w, src_h) = (src.cols(), src.rows());
    let (dst_w, dst_h) = (dst.cols(), dst.rows());

    if matches!(interpolation, InterpolationMode::Bilinear)
        && C == 3
        && src_w == dst_w * 2
        && src_h == dst_h * 2
        && src_w >= 2
        && src_h >= 2
    {
        pyramid::pyrdown_2x_rgb_u8(src.as_slice(), dst.as_slice_mut(), src_w, src_h);
        return Ok(());
    }

    if matches!(interpolation, InterpolationMode::Bilinear)
        && C == 3
        && dst_w == src_w * 2
        && dst_h == src_h * 2
        && src_w >= 2
        && src_h >= 2
    {
        pyramid::pyrup_2x_rgb_u8(src.as_slice(), dst.as_slice_mut(), src_w, src_h);
        return Ok(());
    }

    if matches!(interpolation, InterpolationMode::Nearest) && src_w >= 1 && src_h >= 1 {
        nearest::resize_nearest_u8::<C>(
            src.as_slice(),
            src_w,
            src_h,
            dst.as_slice_mut(),
            dst_w,
            dst_h,
        );
        return Ok(());
    }

    if matches!(interpolation, InterpolationMode::Bilinear)
        && src_w >= 2
        && src_h >= 2
        && (C == 1 || C == 3 || C == 4)
    {
        bilinear::resize_bilinear_u8_nch::<C>(
            src.as_slice(),
            src_w,
            src_h,
            dst.as_slice_mut(),
            dst_w,
            dst_h,
        );
        return Ok(());
    }

    resize_fast_impl(src, dst, interpolation, antialias)
}

/// Resize a 1-channel u8 image. Convenience wrapper around [`resize_fast_u8`].
pub fn resize_fast_mono<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8::<1, _, _>(src, dst, interpolation)
}

/// Grayscale resize with explicit antialias control. See [`resize_fast_u8_aa`].
pub fn resize_fast_mono_aa<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<1, _, _>(src, dst, interpolation, antialias)
}

/// RGB resize with explicit antialias control. See [`resize_fast_u8_aa`].
pub fn resize_fast_rgb_aa<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, 3, A1>,
    dst: &mut Image<u8, 3, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<3, _, _>(src, dst, interpolation, antialias)
}

/// Slow-path per-interpolation dispatch for cases that didn't take the
/// fast paths in [`resize_fast_u8_aa`] (primarily bicubic / lanczos and
/// non-exact-2× bilinear on unusual channel counts).
fn resize_fast_impl<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    if !(C == 1 || C == 3 || C == 4) {
        return Err(ImageError::UnsupportedChannelCount(C));
    }
    let (src_w, src_h) = (src.cols(), src.rows());
    let (dst_w, dst_h) = (dst.cols(), dst.rows());
    match interpolation {
        InterpolationMode::Nearest => {
            nearest::resize_nearest_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
            );
        }
        InterpolationMode::Bilinear => {
            bilinear::resize_bilinear_u8_nch::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
            );
        }
        InterpolationMode::Bicubic => {
            separable::resize_separable_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
                FilterKind::Cubic,
                antialias,
            );
        }
        InterpolationMode::Lanczos => {
            separable::resize_separable_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
                FilterKind::Lanczos3,
                antialias,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::{CpuAllocator, TensorError};

    #[test]
    fn resize_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            (0..3 * 4 * 3).map(|x| x as f32).collect::<Vec<f32>>(),
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        assert_eq!(
            image_resized.as_slice(),
            [
                0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5, 27.0, 28.0, 29.0,
                33.0, 34.0, 35.0
            ]
        );

        Ok(())
    }

    #[test]
    fn resize_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 1, _>::from_size_val(new_size, 0.0f32, CpuAllocator)?;

        super::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 1);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        assert_eq!(image_resized.as_slice(), image_resized.as_slice());

        Ok(())
    }

    #[test]
    fn meshgrid() -> Result<(), TensorError> {
        let (map_x, map_y) =
            crate::interpolation::grid::meshgrid_from_fn(2, 3, |x, y| Ok((x as f32, y as f32)))?;

        assert_eq!(map_x.shape, [3, 2]);
        assert_eq!(map_y.shape, [3, 2]);
        assert_eq!(map_x.get([0, 0]), Some(&0.0));
        assert_eq!(map_x.get([0, 1]), Some(&1.0));
        assert_eq!(map_y.get([0, 0]), Some(&0.0));
        assert_eq!(map_y.get([2, 0]), Some(&2.0));

        Ok(())
    }

    #[test]
    fn resize_native_unsupported_interpolation() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32; 4],
            CpuAllocator,
        )?;

        let mut dst = Image::<_, 1, _>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0f32,
            CpuAllocator,
        )?;

        let err = super::resize_native(&image, &mut dst, super::InterpolationMode::Bicubic);
        assert!(err.is_err());

        let err = super::resize_native(&image, &mut dst, super::InterpolationMode::Lanczos);
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn resize_fast() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0u8; 4 * 5 * 3],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3, _>::from_size_val(new_size, 0, CpuAllocator)?;

        super::resize_fast_rgb(
            &image,
            &mut image_resized,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);
        Ok(())
    }

    /// Sanity-check the exact-2× bilinear upscale fast path on sizes that
    /// exercise both the NEON bulk loop (src_w ≥ 17) and the pure-tail path
    /// (small widths).
    #[test]
    fn resize_fast_2x_upscale() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};

        for (w, h) in [(2, 2), (3, 4), (17, 9), (32, 5), (33, 6)] {
            let data: Vec<u8> = (0..w * h * 3).map(|x| (x % 251) as u8).collect();
            let image = Image::<_, 3, _>::new(
                ImageSize {
                    width: w,
                    height: h,
                },
                data,
                CpuAllocator,
            )?;
            let mut dst = Image::<_, 3, _>::from_size_val(
                ImageSize {
                    width: 2 * w,
                    height: 2 * h,
                },
                0u8,
                CpuAllocator,
            )?;
            super::resize_fast_rgb(&image, &mut dst, super::InterpolationMode::Bilinear)?;

            // Corner pixels must be preserved exactly (f=0/1 clamps on both axes).
            let src_slice = image.as_slice();
            let dst_slice = dst.as_slice();
            let dst_w = 2 * w;
            let dst_h = 2 * h;
            for ch in 0..3 {
                assert_eq!(dst_slice[ch], src_slice[ch], "top-left wxh={w}x{h} ch={ch}");
                assert_eq!(
                    dst_slice[(dst_w - 1) * 3 + ch],
                    src_slice[(w - 1) * 3 + ch],
                    "top-right wxh={w}x{h} ch={ch}"
                );
                assert_eq!(
                    dst_slice[(dst_h - 1) * dst_w * 3 + ch],
                    src_slice[(h - 1) * w * 3 + ch],
                    "bottom-left wxh={w}x{h} ch={ch}"
                );
                assert_eq!(
                    dst_slice[((dst_h - 1) * dst_w + (dst_w - 1)) * 3 + ch],
                    src_slice[((h - 1) * w + (w - 1)) * 3 + ch],
                    "bottom-right wxh={w}x{h} ch={ch}"
                );
            }
        }
        Ok(())
    }
}
