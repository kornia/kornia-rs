use crate::parallel;

use super::interpolate::validate_interpolation;
use super::InterpolationMode;
use kornia_image::{Image, ImageError};

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
            let mx = map_x.as_cudaslice().ok_or_else(|| {
                ImageError::Cuda("remap: cannot extract map_x device slice".into())
            })?;
            let my = map_y.as_cudaslice().ok_or_else(|| {
                ImageError::Cuda("remap: cannot extract map_y device slice".into())
            })?;
            return exec.run(|stream| {
                super::cuda::remap_f32_cuda(src, dst, mx, my, interpolation, stream)
            });
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
}
