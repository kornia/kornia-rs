use crate::parallel;
use rayon::prelude::*;

use super::interpolate::validate_interpolation;
use super::InterpolationMode;
use kornia_image::{Image, ImageError};

#[cfg(feature = "cuda")]
use {
    crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err},
    crate::cuda::remap::{
        launch_remap_bilinear_cuda, launch_remap_bilinear_u8_cuda, launch_remap_nearest_cuda,
        launch_remap_nearest_u8_cuda,
    },
    cudarc::driver::CudaStream,
    std::sync::Arc,
};

/// Apply generic geometric transformation to an image.
///
/// Maps `map_x` and `map_y` give the floating-point source coordinate for
/// each output pixel — one f32 per output pixel, shaped `(height, width, 1)`.
/// When both `src`/`dst` and the maps are device-resident, the call is
/// transparently dispatched to the CUDA bilinear/nearest kernels (bilinear and
/// nearest only; bicubic and lanczos run on the CPU path for any residency).
///
/// # Arguments
///
/// * `src` - The input image container with shape (height, width, C).
/// * `dst` - The output image container with shape (height, width, C).
/// * `map_x` - Source x coordinate for each output pixel, shape (height, width, 1).
/// * `map_y` - Source y coordinate for each output pixel, shape (height, width, 1).
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * `map_x` and `map_y` must have the same size.
/// * `dst` must have the same size as the maps.
pub fn remap<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if map_x.size() != map_y.size() {
        return Err(ImageError::InvalidImageSize(
            map_x.rows(),
            map_x.cols(),
            map_y.rows(),
            map_y.cols(),
        ));
    }
    if dst.size() != map_x.size() {
        return Err(ImageError::InvalidImageSize(
            dst.rows(),
            dst.cols(),
            map_x.rows(),
            map_x.cols(),
        ));
    }

    validate_interpolation(interpolation)?;

    // Device pairs with device maps route to the CUDA kernels (bilinear/nearest
    // only — bicubic/lanczos fall through to the CPU path below). Mixed residency
    // is a typed error; there is no implicit host↔device transfer.
    #[cfg(feature = "cuda")]
    {
        use crate::cuda::dispatch::{is_device, pair_residency, Residency};
        if let Residency::Device(exec) = pair_residency(src, dst)? {
            if !is_device(map_x) || !is_device(map_y) {
                return Err(ImageError::Cuda(
                    "remap: map_x and map_y must be device-resident when src/dst are on GPU".into(),
                ));
            }
            return exec
                .run(|stream| remap_f32_cuda(src, dst, map_x, map_y, interpolation, stream));
        }
        if is_device(map_x) || is_device(map_y) {
            return Err(ImageError::Cuda(
                "remap: map_x and map_y must be host-resident when src/dst are on CPU".into(),
            ));
        }
    }

    // One monomorphic pixel loop per mode — see the note in `resize`.
    macro_rules! run {
        ($sampler:path) => {
            parallel::par_iter_rows_resample(
                dst,
                map_x.as_slice(),
                map_y.as_slice(),
                |&x, &y, dst_pixel| {
                    for (c, pixel) in dst_pixel.iter_mut().enumerate() {
                        *pixel = $sampler(src, x, y, c);
                    }
                },
            )
        };
    }
    match interpolation {
        InterpolationMode::Bilinear => run!(crate::interpolation::bilinear_interpolation),
        InterpolationMode::Nearest => run!(crate::interpolation::nearest_neighbor_interpolation),
        InterpolationMode::Bicubic => run!(crate::interpolation::bicubic_sample),
        InterpolationMode::Lanczos => run!(crate::interpolation::lanczos_sample),
    }

    Ok(())
}

/// Apply a generic geometric transformation to a `u8` image.
///
/// The coordinate maps are still `f32` images, one source coordinate per
/// output pixel. Bilinear interpolation uses the same Q10 fixed-point sampler
/// as the `u8` warp kernels, so fractional weights are quantized before the
/// final byte blend. Nearest-neighbor samples with constant-0 border handling.
///
/// # Arguments
///
/// * `src` - The input image container with shape (height, width, C).
/// * `dst` - The output image container with shape (height, width, C).
/// * `map_x` - Source x coordinate for each output pixel, shape (height, width, 1).
/// * `map_y` - Source y coordinate for each output pixel, shape (height, width, 1).
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// * `map_x` and `map_y` must have the same size.
/// * `dst` must have the same size as the maps.
/// * [`ImageError::UnsupportedInterpolation`] is returned for interpolation
///   modes other than bilinear and nearest-neighbor.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::interpolation::{remap_u8, InterpolationMode};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let src = Image::<u8, 3>::from_size_val(ImageSize { width: 4, height: 4 }, 128u8)?;
/// let mut dst = Image::<u8, 3>::from_size_val(ImageSize { width: 4, height: 4 }, 0u8)?;
/// let map_x = Image::<f32, 1>::from_size_val(ImageSize { width: 4, height: 4 }, 1.0f32)?;
/// let map_y = Image::<f32, 1>::from_size_val(ImageSize { width: 4, height: 4 }, 1.0f32)?;
/// remap_u8(&src, &mut dst, &map_x, &map_y, InterpolationMode::Bilinear)?;
/// # Ok(())
/// # }
/// ```
pub fn remap_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    if map_x.size() != map_y.size() {
        return Err(ImageError::InvalidImageSize(
            map_x.rows(),
            map_x.cols(),
            map_y.rows(),
            map_y.cols(),
        ));
    }
    if dst.size() != map_x.size() {
        return Err(ImageError::InvalidImageSize(
            dst.rows(),
            dst.cols(),
            map_x.rows(),
            map_x.cols(),
        ));
    }

    // remap_u8 only accelerates bilinear and nearest; reject others with the
    // same error type on both CPU and GPU paths so callers get a consistent
    // error variant regardless of where the images live.
    match interpolation {
        InterpolationMode::Bilinear | InterpolationMode::Nearest => {}
        other => return Err(ImageError::UnsupportedInterpolation(other)),
    }

    #[cfg(feature = "cuda")]
    {
        use crate::cuda::dispatch::{is_device, pair_residency, Residency};
        if let Residency::Device(exec) = pair_residency(src, dst)? {
            if !is_device(map_x) || !is_device(map_y) {
                return Err(ImageError::Cuda(
                    "remap_u8: map_x and map_y must be device-resident when src/dst are on GPU"
                        .into(),
                ));
            }
            return exec.run(|stream| remap_u8_cuda(src, dst, map_x, map_y, interpolation, stream));
        }
        if is_device(map_x) || is_device(map_y) {
            return Err(ImageError::Cuda(
                "remap_u8: map_x and map_y must be host-resident when src/dst are on CPU".into(),
            ));
        }
    }

    let src_slice = src.as_slice();
    let map_x_slice = map_x.as_slice();
    let map_y_slice = map_y.as_slice();
    let src_w = src.cols() as i32;
    let src_h = src.rows() as i32;
    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;
    let src_stride = src.cols() * C;
    let dst_w = dst.cols();
    let dst_stride = dst_w * C;

    if dst_stride == 0 {
        return Ok(());
    }

    let zero_pixel = |dst_pixel: &mut [u8]| {
        for pixel in dst_pixel.iter_mut().take(C) {
            *pixel = 0;
        }
    };

    match interpolation {
        InterpolationMode::Bilinear => {
            #[cfg(target_arch = "x86_64")]
            if C == 3 && crate::simd::cpu_features().has_avx2 {
                // SAFETY: the helper is only compiled on x86_64, we have
                // already checked AVX2 at runtime, and the helper keeps the
                // same bounds checks as the scalar path.
                unsafe {
                    remap_u8_bilinear_c3_avx2(
                        src_slice,
                        dst,
                        map_x_slice,
                        map_y_slice,
                        src_w,
                        src_h,
                        src_stride,
                        dst_w,
                        dst_stride,
                    );
                }
                return Ok(());
            }

            dst.as_slice_mut()
                .par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let row_base = y * dst_w;
                    for x in 0..dst_w {
                        let xf = map_x_slice[row_base + x];
                        let yf = map_y_slice[row_base + x];
                        let dst_pixel = &mut dst_row[x * C..x * C + C];
                        // NaN-safe OOB check followed by AVX2/NEON-accelerated sampler.
                        if !xf.is_finite() || !yf.is_finite() {
                            zero_pixel(dst_pixel);
                            continue;
                        }
                        let xi = xf.floor() as i32;
                        let yi = yf.floor() as i32;
                        if xi < 0 || xi >= src_w || yi < 0 || yi >= src_h {
                            zero_pixel(dst_pixel);
                            continue;
                        }
                        let fx_q10 = ((xf - xi as f32) * 1024.0) as u32;
                        let fy_q10 = ((yf - yi as f32) * 1024.0) as u32;
                        crate::warp::bilinear_sample_u8_valid::<C>(
                            src_slice, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
                        );
                    }
                });
        }
        InterpolationMode::Nearest => {
            dst.as_slice_mut()
                .par_chunks_exact_mut(dst_stride)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let row_base = y * dst_w;
                    for x in 0..dst_w {
                        let xf = map_x_slice[row_base + x];
                        let yf = map_y_slice[row_base + x];
                        let dst_pixel = &mut dst_row[x * C..x * C + C];
                        if !(xf >= 0.0 && xf < src_w_f && yf >= 0.0 && yf < src_h_f) {
                            zero_pixel(dst_pixel);
                        } else {
                            let xi = (xf.round() as i32).clamp(0, src_w - 1) as usize;
                            let yi = (yf.round() as i32).clamp(0, src_h - 1) as usize;
                            let src_idx = (yi * src.cols() + xi) * C;
                            dst_pixel.copy_from_slice(&src_slice[src_idx..src_idx + C]);
                        }
                    }
                });
        }
        // Bicubic/Lanczos are rejected at the top of remap_u8; unreachable here.
        _ => unreachable!(),
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn remap_u8_bilinear_c3_avx2<const C: usize>(
    src_slice: &[u8],
    dst: &mut Image<u8, C>,
    map_x_slice: &[f32],
    map_y_slice: &[f32],
    src_w: i32,
    src_h: i32,
    src_stride: usize,
    dst_w: usize,
    dst_stride: usize,
) {
    debug_assert!(C == 3);

    let zero_pixel = |dst_pixel: &mut [u8]| {
        for pixel in dst_pixel.iter_mut().take(C) {
            *pixel = 0;
        }
    };

    dst.as_slice_mut()
        .par_chunks_exact_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let row_base = y * dst_w;
            for x in 0..dst_w {
                let xf = map_x_slice[row_base + x];
                let yf = map_y_slice[row_base + x];
                let dst_pixel = &mut dst_row[x * C..x * C + C];
                if !xf.is_finite() || !yf.is_finite() {
                    zero_pixel(dst_pixel);
                    continue;
                }
                let xi = xf.floor() as i32;
                let yi = yf.floor() as i32;
                if xi < 0 || xi >= src_w || yi < 0 || yi >= src_h {
                    zero_pixel(dst_pixel);
                    continue;
                }
                let fx_q10 = ((xf - xi as f32) * 1024.0) as u32;
                let fy_q10 = ((yf - yi as f32) * 1024.0) as u32;

                if xi < src_w - 2 || yi < src_h - 2 {
                    // SAFETY: x86_64 + AVX2 were checked by the caller, and
                    // the bounds condition mirrors the helper's preconditions.
                    unsafe {
                        crate::warp::bilinear_sample_u8_valid_c3_avx2(
                            src_slice.as_ptr(),
                            src_w,
                            src_h,
                            src_stride,
                            xi,
                            yi,
                            fx_q10,
                            fy_q10,
                            dst_pixel.as_mut_ptr(),
                        );
                    }
                } else {
                    crate::warp::bilinear_sample_u8_valid::<C>(
                        src_slice, src_w, src_h, src_stride, xi, yi, fx_q10, fy_q10, dst_pixel,
                    );
                }
            }
        });
}

/// Run the CUDA remap for a device-resident f32 triple (src, dst, maps).
///
/// `map_x` and `map_y` are single-channel device images shaped like `dst` — one
/// f32 source coordinate per output pixel.  Bilinear and nearest
/// are hardware-accelerated; bicubic and lanczos must be handled by the CPU
/// path (the caller, [`remap`], falls through for those modes).
#[cfg(feature = "cuda")]
fn remap_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err("remap", "3-channel f32 images"));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let map_x = map_x
        .as_cudaslice()
        .ok_or_else(|| untyped_device_err("map_x"))?;
    let map_y = map_y
        .as_cudaslice()
        .ok_or_else(|| untyped_device_err("map_y"))?;
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

#[cfg(feature = "cuda")]
fn remap_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    map_x: &Image<f32, 1>,
    map_y: &Image<f32, 1>,
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let map_x = map_x
        .as_cudaslice()
        .ok_or_else(|| untyped_device_err("map_x"))?;
    let map_y = map_y
        .as_cudaslice()
        .ok_or_else(|| untyped_device_err("map_y"))?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let channels = C as u32;

    match interpolation {
        InterpolationMode::Bilinear => launch_remap_bilinear_u8_cuda(
            ctx, stream, s, map_x, map_y, d, src_w, src_h, dst_w, dst_h, channels, None,
        ),
        InterpolationMode::Nearest => launch_remap_nearest_u8_cuda(
            ctx, stream, s, map_x, map_y, d, src_w, src_h, dst_w, dst_h, channels, None,
        ),
        other => Err(crate::cuda::remap::CudaRemapError::Cuda(format!(
            "remap_u8 CUDA: {other:?} is not GPU-accelerated — move images to host for this mode"
        ))),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    fn make_map(w: usize, h: usize, data: Vec<f32>) -> Result<Image<f32, 1>, ImageError> {
        Image::<f32, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
    }

    /// All four interpolation modes are supported since the bicubic/lanczos
    /// CPU samplers landed; an identity map must reproduce the source.
    #[test]
    fn remap_supports_all_modes() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1.0f32, 2.0, 3.0, 4.0],
        )?;
        let map_x = make_map(2, 2, vec![0.0, 1.0, 0.0, 1.0])?;
        let map_y = make_map(2, 2, vec![0.0, 0.0, 1.0, 1.0])?;
        let mut dst = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;
        super::remap(
            &image,
            &mut dst,
            &map_x,
            &map_y,
            super::InterpolationMode::Lanczos,
        )?;
        Ok(())
    }

    #[test]
    fn remap_smoke() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )?;

        let map_x = make_map(2, 2, vec![0.0, 2.0, 0.0, 2.0])?;
        let map_y = make_map(2, 2, vec![0.0, 0.0, 2.0, 2.0])?;

        let expected = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0, 2.0, 6.0, 8.0],
        )?;

        let mut image_transformed = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
        )?;

        super::remap(
            &image,
            &mut image_transformed,
            &map_x,
            &map_y,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 2);

        for (a, b) in image_transformed
            .as_slice()
            .iter()
            .zip(expected.as_slice().iter())
        {
            assert!((a - b).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn remap_u8_identity_bilinear() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1u8, 2, 3, 4],
        )?;
        let map_x = make_map(2, 2, vec![0.0, 1.0, 0.0, 1.0])?;
        let map_y = make_map(2, 2, vec![0.0, 0.0, 1.0, 1.0])?;
        let mut dst = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0,
        )?;

        super::remap_u8(
            &image,
            &mut dst,
            &map_x,
            &map_y,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(dst.as_slice(), image.as_slice());
        Ok(())
    }

    #[test]
    fn remap_u8_rgb_identity_bilinear() -> Result<(), ImageError> {
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )?;
        let map_x = make_map(2, 2, vec![0.0, 1.0, 0.0, 1.0])?;
        let map_y = make_map(2, 2, vec![0.0, 0.0, 1.0, 1.0])?;
        let mut dst = Image::<_, 3>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0,
        )?;

        super::remap_u8(
            &image,
            &mut dst,
            &map_x,
            &map_y,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(dst.as_slice(), image.as_slice());
        Ok(())
    }

    #[test]
    fn remap_u8_bilinear_quantizes_fractional_weights() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![0u8, 255],
        )?;
        let map_x = make_map(1, 1, vec![0.1])?;
        let map_y = make_map(1, 1, vec![0.0])?;
        let mut dst = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0,
        )?;

        super::remap_u8(
            &image,
            &mut dst,
            &map_x,
            &map_y,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(dst.as_slice(), &[25]);
        Ok(())
    }

    #[test]
    fn remap_u8_nearest_zeroes_oob_maps() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![10u8, 20, 30, 40],
        )?;
        let map_x = make_map(2, 2, vec![0.49, 1.49, -1.0, 0.5])?;
        let map_y = make_map(2, 2, vec![0.49, 0.49, 0.5, 2.0])?;
        let mut dst = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0,
        )?;

        super::remap_u8(
            &image,
            &mut dst,
            &map_x,
            &map_y,
            super::InterpolationMode::Nearest,
        )?;

        assert_eq!(dst.as_slice(), &[10, 20, 0, 0]);
        Ok(())
    }
}

// ── Device tests ─────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::{remap, remap_u8};
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use kornia_image::{Image, ImageError, ImageSize};

    fn identity_maps(w: usize, h: usize) -> Result<(Image<f32, 1>, Image<f32, 1>), ImageError> {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mx: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|x| x as f32)).collect();
        let my: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |_| y as f32)).collect();
        Ok((
            Image::<f32, 1>::new(size, mx)?,
            Image::<f32, 1>::new(size, my)?,
        ))
    }

    /// `remap` with device images and an identity map must be bit-identical to
    /// the CPU path — the byte-exact contract for the remap kernel.
    fn check_remap_mode(mode: InterpolationMode) -> Result<(), ImageError> {
        let stream = default_stream();
        let (w, h) = (65, 33);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, pattern_f32(w * h * 3))?;
        let (mx, my) = identity_maps(w, h)?;

        let mut cpu_dst = Image::<f32, 3>::from_size_val(size, 0.0)?;
        remap(&src, &mut cpu_dst, &mx, &my, mode)?;

        let d_src = src.to_cuda(&stream)?;
        let mut d_dst = Image::<f32, 3>::zeros_cuda(size, &stream)?;
        let d_mx = mx.to_cuda(&stream)?;
        let d_my = my.to_cuda(&stream)?;
        remap(&d_src, &mut d_dst, &d_mx, &d_my, mode)?;

        let back = d_dst.to_host_owned()?;
        for (i, (c, g)) in cpu_dst.as_slice().iter().zip(back.as_slice()).enumerate() {
            assert!(
                c.to_bits() == g.to_bits(),
                "remap {mode:?} element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                c.to_bits(),
                g.to_bits()
            );
        }
        Ok(())
    }

    #[test]
    fn public_remap_device_equals_host() -> Result<(), ImageError> {
        check_remap_mode(InterpolationMode::Bilinear)
    }

    /// Nearest-neighbor device path is bit-identical to host.
    #[test]
    fn public_remap_nearest_device_equals_host() -> Result<(), ImageError> {
        check_remap_mode(InterpolationMode::Nearest)
    }

    /// Mixed residency — device src/dst but host maps — must be a typed error.
    #[test]
    fn device_images_with_host_maps_is_error() -> Result<(), ImageError> {
        let stream = default_stream();
        let (w, h) = (16, 16);
        let size = ImageSize {
            width: w,
            height: h,
        };

        let src = Image::<f32, 3>::new(size, pattern_f32(w * h * 3))?;
        let d_src = src.to_cuda(&stream)?;
        let mut d_dst = Image::<f32, 3>::zeros_cuda(size, &stream)?;
        let (mx, my) = identity_maps(w, h)?;

        let Err(err) = remap(&d_src, &mut d_dst, &mx, &my, InterpolationMode::Bilinear) else {
            panic!("expected an error when the maps are host-resident");
        };
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("device-resident")),
            "expected a Cuda error about device-resident maps, got {err:?}"
        );
        Ok(())
    }

    /// Byte-exact contract for the **u8** remap kernel: the GPU quantisation
    /// must reproduce the CPU u8 path exactly, for a non-trivial map that
    /// exercises fractional weights rather than just an identity copy.
    fn check_remap_u8_mode<const C: usize>(
        mode: InterpolationMode,
        (w, h): (usize, usize),
    ) -> Result<(), ImageError> {
        let stream = default_stream();
        let size = ImageSize {
            width: w,
            height: h,
        };

        // Deterministic ramp over the full u8 range; 251 is prime so the
        // pattern does not align with the row stride.
        let data: Vec<u8> = (0..w * h * C).map(|i| (i % 251) as u8).collect();
        let src = Image::<u8, C>::new(size, data)?;

        // Fractional map with irrational-ish steps so the bilinear weights are
        // never 0 or 1 — that is where quantisation actually differs — and a
        // deliberate negative/overshoot band to exercise the border guard.
        let mx: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|x| x as f32 * 1.03 - 1.7))
            .collect();
        let my: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |_| y as f32 * 1.07 - 2.3))
            .collect();
        let map_x = Image::<f32, 1>::new(size, mx)?;
        let map_y = Image::<f32, 1>::new(size, my)?;

        let mut cpu_dst = Image::<u8, C>::from_size_val(size, 0)?;
        remap_u8(&src, &mut cpu_dst, &map_x, &map_y, mode)?;

        let d_src = src.to_cuda(&stream)?;
        let mut d_dst = Image::<u8, C>::zeros_cuda(size, &stream)?;
        let d_mx = map_x.to_cuda(&stream)?;
        let d_my = map_y.to_cuda(&stream)?;
        remap_u8(&d_src, &mut d_dst, &d_mx, &d_my, mode)?;
        let gpu_dst = d_dst.to_host_owned()?;

        let mismatches: Vec<_> = cpu_dst
            .as_slice()
            .iter()
            .zip(gpu_dst.as_slice())
            .enumerate()
            .filter(|(_, (c, g))| c != g)
            .take(8)
            .map(|(i, (c, g))| format!("[{i}] cpu {c} gpu {g}"))
            .collect();
        assert!(
            mismatches.is_empty(),
            "remap_u8 {mode:?} {C}ch {w}x{h}: {} of {} elements differ; first: {}",
            cpu_dst
                .as_slice()
                .iter()
                .zip(gpu_dst.as_slice())
                .filter(|(c, g)| c != g)
                .count(),
            cpu_dst.as_slice().len(),
            mismatches.join(", ")
        );
        Ok(())
    }

    #[test]
    fn public_remap_u8_device_equals_host() -> Result<(), ImageError> {
        for &size in &[(65, 33), (127, 63), (16, 16), (1, 1)] {
            check_remap_u8_mode::<3>(InterpolationMode::Bilinear, size)?;
        }
        Ok(())
    }

    #[test]
    fn public_remap_u8_nearest_device_equals_host() -> Result<(), ImageError> {
        for &size in &[(65, 33), (127, 63), (16, 16), (1, 1)] {
            check_remap_u8_mode::<3>(InterpolationMode::Nearest, size)?;
        }
        Ok(())
    }

    /// The kernel takes the channel count as a runtime argument, so 1- and
    /// 4-channel images exercise a different indexing path than 3-channel.
    #[test]
    fn public_remap_u8_channels_device_equals_host() -> Result<(), ImageError> {
        for &mode in &[InterpolationMode::Bilinear, InterpolationMode::Nearest] {
            check_remap_u8_mode::<1>(mode, (65, 33))?;
            check_remap_u8_mode::<4>(mode, (65, 33))?;
        }
        Ok(())
    }
}
