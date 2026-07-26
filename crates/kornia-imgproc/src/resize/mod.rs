//! Image resize — dispatcher + per-algorithm kernels.
//!
//! Per-arch row primitives (scalar + aarch64 NEON + x86_64 AVX2) live in
//! [`kernels`]; the algorithm files above it (bilinear, nearest, pyramid,
//! separable, fused) stay free of `#[cfg(target_arch)]` pairs and call into
//! kernels through a thin seam.
//!
//! # Fused ops first
//!
//! When the resize feeds a model input, prefer the **fused** entry points
//! ([`resize_normalize_to_tensor_u8_to_f32`],
//! [`resize_normalize_to_tensor_u8_to_f32_bilinear`]) over resize-then-
//! normalize: one pass, every input byte touched once, no intermediate u8
//! requantization. The plain resizers below are for when a u8 image is the
//! actual product.
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
#[cfg(feature = "cuda")]
mod cuda;
mod fused;
mod kernels;
mod nearest;
mod opencv_compat;
mod pyramid;
mod separable;

pub use opencv_compat::{resize_opencv_f32, resize_opencv_u8};

pub use fused::{
    resize_normalize_to_tensor_u8_to_f32, resize_normalize_to_tensor_u8_to_f32_bilinear,
    resize_normalize_to_tensor_u8_to_f32_nearest, resize_normalize_to_tensor_u8_to_f32_separable,
    NormalizeParams,
};

use crate::{
    interpolation::{grid::meshgrid_from_fn, validate_interpolation, InterpolationMode},
    parallel,
};
use kornia_image::{Image, ImageError};

use common::FilterKind;

/// Resize an image to a new size.
///
/// The function resizes an image to a new size using the specified interpolation mode.
/// It supports any number of channels and data types.
///
/// Sampling uses the **half-pixel** (pixel-center) grid,
/// `sx = (x + 0.5) * src/dst - 0.5`, the convention shared by OpenCV, Pillow,
/// ONNX `Resize`, PyTorch (`align_corners=False`), and NVIDIA VPI, and by this
/// crate's CUDA resize kernels. Earlier releases used align-corners, which
/// produced different pixel values for the same inputs.
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
/// use kornia_imgproc::resize::resize;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///     width: 2,
///     height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0.0).unwrap();
///
/// resize(
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
pub fn resize<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;

    // Device pairs route to the CUDA kernels (bit-identical output — see the
    // byte-exact contract below). This must run BEFORE the same-size
    // short-circuit: `as_slice_mut` on a device image would be a host access
    // of device memory. Mixed host/device pairs are a typed error; there is
    // no implicit transfer in either direction.
    #[cfg(feature = "cuda")]
    if let crate::cuda::dispatch::Residency::Device(exec) =
        crate::cuda::dispatch::pair_residency(src, dst)?
    {
        return exec.run(|stream| cuda::resize_f32_cuda(src, dst, interpolation, stream));
    }

    // Short-circuit when the resize is a no-op.
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    // Lanczos is separable on both backends (the CUDA pipeline is H-then-V
    // with host-built tables); the direct per-pixel sampler below would give
    // a different — and slower — result. Bicubic stays per-pixel: the kernel
    // is direct 4×4 too.
    if interpolation == InterpolationMode::Lanczos {
        crate::interpolation::lanczos::resize_lanczos_separable(src, dst);
        return Ok(());
    }

    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());

    // Half-pixel (pixel-center) sampling grid: `sx = (x + 0.5) * scale - 0.5`,
    // clamped to the source. This is the convention of OpenCV, Pillow, ONNX
    // `Resize` (default `coordinate_transformation_mode = half_pixel`),
    // PyTorch's recommended `align_corners=False`, and NVIDIA VPI — and the
    // grid the CUDA resize kernels in this crate already sample on, so CPU and
    // GPU agree. Earlier releases used align-corners
    // (`sx = x * (src-1)/(dst-1)`), which shifts content relative to every
    // mainstream library; outputs changed accordingly.
    // The grid is separable: evaluate each axis once (dst_w + dst_h values)
    // and let the per-pixel closure be two table loads.
    //
    // BYTE-EXACT CONTRACT with the CUDA resize kernels: the coordinate is
    // computed as `a*x + b` with `a = src/dst` and `b = 0.5*a - 0.5` — the
    // exact f32 expression `cuda::resize::PixelMapping::HalfPixel.coeffs`
    // feeds the kernels, which evaluate `a*x + b` as an uncontracted
    // multiply-add (`--fmad=false`). Same ops, same roundings, identical
    // coordinates — the CPU/GPU parity tests assert bit-equality on top of
    // this. Algebraically equivalent forms like `(x + 0.5)*a - 0.5` round
    // differently; don't "simplify" one side without the other.
    let axis_lut = |src_len: usize, dst_len: usize| -> Vec<f32> {
        let a = src_len as f32 / dst_len as f32;
        let b = 0.5 * a - 0.5;
        let max = (src_len - 1) as f32;
        (0..dst_len)
            .map(|i| (a * i as f32 + b).clamp(0.0, max))
            .collect()
    };
    let xs = axis_lut(src.cols(), dst_cols);
    let ys = axis_lut(src.rows(), dst_rows);
    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| Ok((xs[x], ys[y])))?;

    // One monomorphic pixel loop per mode: dispatching inside the loop makes
    // the closure body carry all four samplers and deoptimizes the hot
    // bilinear path (measured +25-65% on the CPU resize bench).
    macro_rules! run {
        ($sampler:path) => {
            parallel::par_iter_rows_resample(dst, &map_x, &map_y, |&x, &y, dst_pixel| {
                for (k, pixel) in dst_pixel.iter_mut().enumerate() {
                    *pixel = $sampler(src, x, y, k);
                }
            })
        };
    }
    match interpolation {
        InterpolationMode::Bilinear => run!(crate::interpolation::bilinear_interpolation),
        InterpolationMode::Nearest => run!(crate::interpolation::nearest_neighbor_interpolation),
        InterpolationMode::Bicubic => run!(crate::interpolation::bicubic_sample),
        // Handled by the separable branch above.
        InterpolationMode::Lanczos => unreachable!(),
    }

    Ok(())
}

/// Resize a 3-channel u8 image. Convenience wrapper around [`resize_fast_u8`].
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::resize::resize_fast_rgb;
/// use kornia_imgproc::interpolation::InterpolationMode;
///
/// let image = Image::<_, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0u8; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let new_size = ImageSize {
///   width: 2,
///   height: 3,
/// };
///
/// let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0).unwrap();
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
pub fn resize_fast_rgb(
    src: &Image<u8, 3>,
    dst: &mut Image<u8, 3>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8::<3>(src, dst, interpolation)
}

/// Generic fast u8 resize for 1/3/4-channel images (PIL-quality antialiased downscale).
pub fn resize_fast_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<C>(src, dst, interpolation, true)
}

/// Which u8 kernel a `(mode, geometry, channel count)` combination resolves
/// to — the SINGLE routing decision consumed by both the CPU cascade
/// ([`resize_fast_u8_aa`]) and the CUDA cascade
/// (`cuda::resize_fast_u8_cuda`), so the two sides cannot drift: a device
/// pair always runs the GPU twin of the kernel a host pair would run.
/// Kernel-support errors (channel counts, degenerate bilinear sources) are
/// decided here too, so the error VARIANT is residency-independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResizeU8Path {
    /// Exact-2× RGB box downscale (bilinear fast path).
    PyrDown2xRgb,
    /// Exact-2× RGB upscale (bilinear fast path).
    PyrUp2xRgb,
    /// Nearest gather (any channel count).
    Nearest,
    /// Generic Q14 bilinear (C ∈ {1, 3, 4}, source at least 2×2).
    Bilinear,
    /// Two-pass separable Q14 (bicubic / lanczos, C ∈ {1, 3, 4}).
    Separable(FilterKind),
}

pub(crate) fn resize_u8_path(
    channels: usize,
    mode: InterpolationMode,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<ResizeU8Path, ImageError> {
    use InterpolationMode as I;
    Ok(match mode {
        I::Bilinear
            if channels == 3
                && src_w == dst_w * 2
                && src_h == dst_h * 2
                && src_w >= 2
                && src_h >= 2 =>
        {
            ResizeU8Path::PyrDown2xRgb
        }
        I::Bilinear
            if channels == 3
                && dst_w == src_w * 2
                && dst_h == src_h * 2
                && src_w >= 2
                && src_h >= 2 =>
        {
            ResizeU8Path::PyrUp2xRgb
        }
        I::Nearest => ResizeU8Path::Nearest,
        I::Bilinear => {
            if !(channels == 1 || channels == 3 || channels == 4) {
                return Err(ImageError::UnsupportedChannelCount(channels));
            }
            // A 1-pixel axis has no second bilinear tap; previously the CPU
            // fall-through PANICKED on this (LUT offset underflow) while the
            // GPU returned a typed error. Both sides now share this error.
            if src_w < 2 || src_h < 2 {
                return Err(ImageError::InvalidImageSize(src_w, src_h, 2, 2));
            }
            ResizeU8Path::Bilinear
        }
        I::Bicubic | I::Lanczos => {
            if !(channels == 1 || channels == 3 || channels == 4) {
                return Err(ImageError::UnsupportedChannelCount(channels));
            }
            ResizeU8Path::Separable(if mode == I::Bicubic {
                FilterKind::Cubic
            } else {
                FilterKind::Lanczos3
            })
        }
    })
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
pub fn resize_fast_u8_aa<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    // Device pairs route to the CUDA u8 kernels (bit-identical output — the
    // coordinate/weight tables come from the same host builders the CPU
    // uses). Mixed host/device pairs are a typed error; there is no implicit
    // transfer in either direction.
    #[cfg(feature = "cuda")]
    if let crate::cuda::dispatch::Residency::Device(exec) =
        crate::cuda::dispatch::pair_residency(src, dst)?
    {
        return exec
            .run(|stream| cuda::resize_fast_u8_cuda(src, dst, interpolation, antialias, stream));
    }

    let (src_w, src_h) = (src.cols(), src.rows());
    let (dst_w, dst_h) = (dst.cols(), dst.rows());

    match resize_u8_path(C, interpolation, src_w, src_h, dst_w, dst_h)? {
        ResizeU8Path::PyrDown2xRgb => {
            pyramid::pyrdown_2x_rgb_u8(src.as_slice(), dst.as_slice_mut(), src_w, src_h);
        }
        ResizeU8Path::PyrUp2xRgb => {
            pyramid::pyrup_2x_rgb_u8(src.as_slice(), dst.as_slice_mut(), src_w, src_h);
        }
        ResizeU8Path::Nearest => {
            nearest::resize_nearest_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
            );
        }
        ResizeU8Path::Bilinear => {
            bilinear::resize_bilinear_u8_nch::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
            );
        }
        ResizeU8Path::Separable(filt) => {
            separable::resize_separable_u8::<C>(
                src.as_slice(),
                src_w,
                src_h,
                dst.as_slice_mut(),
                dst_w,
                dst_h,
                filt,
                antialias,
            );
        }
    }
    Ok(())
}

/// Resize a 1-channel u8 image. Convenience wrapper around [`resize_fast_u8`].
pub fn resize_fast_mono(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    resize_fast_u8::<1>(src, dst, interpolation)
}

/// Grayscale resize with explicit antialias control. See [`resize_fast_u8_aa`].
pub fn resize_fast_mono_aa(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<1>(src, dst, interpolation, antialias)
}

/// RGB resize with explicit antialias control. See [`resize_fast_u8_aa`].
pub fn resize_fast_rgb_aa(
    src: &Image<u8, 3>,
    dst: &mut Image<u8, 3>,
    interpolation: InterpolationMode,
    antialias: bool,
) -> Result<(), ImageError> {
    resize_fast_u8_aa::<3>(src, dst, interpolation, antialias)
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::TensorError;

    #[test]
    fn resize_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            (0..3 * 4 * 3).map(|x| x as f32).collect::<Vec<f32>>(),
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0.0)?;

        super::resize(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.size().width, 2);
        assert_eq!(image_resized.size().height, 3);

        // Half-pixel grid: sx = (x+0.5)*3/2 - 0.5 ∈ {0.25, 1.75},
        // sy = (y+0.5)*4/3 - 0.5 ∈ {1/6, 1.5, 17/6}. The source value is
        // linear in position (v = 9y + 3x + c), so bilinear reproduces it
        // exactly up to float error: v = 9*sy + 3*sx + c.
        let expected = [
            2.25, 3.25, 4.25, 6.75, 7.75, 8.75, 14.25, 15.25, 16.25, 18.75, 19.75, 20.75, 26.25,
            27.25, 28.25, 30.75, 31.75, 32.75,
        ];
        for (i, (got, want)) in image_resized.as_slice().iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "pixel {i}: got {got}, want {want}"
            );
        }

        Ok(())
    }

    #[test]
    fn resize_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 1>::from_size_val(new_size, 0.0f32)?;

        super::resize(
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

    /// All four interpolation modes are supported since the bicubic/lanczos
    /// CPU paths landed.
    #[test]
    fn resize_supports_all_modes() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32; 4],
        )?;

        let mut dst = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0.0f32,
        )?;

        super::resize(&image, &mut dst, super::InterpolationMode::Bicubic)?;
        super::resize(&image, &mut dst, super::InterpolationMode::Lanczos)?;
        Ok(())
    }

    #[test]
    fn resize_fast() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0u8; 4 * 5 * 3],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_resized = Image::<_, 3>::from_size_val(new_size, 0)?;

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
            let image = Image::<_, 3>::new(
                ImageSize {
                    width: w,
                    height: h,
                },
                data,
            )?;
            let mut dst = Image::<_, 3>::from_size_val(
                ImageSize {
                    width: 2 * w,
                    height: 2 * h,
                },
                0u8,
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
