use crate::device::KorniaDevice;
use crate::interpolation::{interpolate_pixel_fast, validate_interpolation, InterpolationMode};
use crate::parallel;

#[cfg(feature = "gpu")]
use crate::gpu::warp::warp_affine_kernel;
#[cfg(feature = "gpu")]
use cubecl::{client::ComputeClient, prelude::*};

/// Inverts a 2x3 affine transformation matrix.
///
/// Arguments:
///
/// * `m` - The 2x3 affine transformation matrix.
///
/// Returns:
///
/// The inverted 2x3 affine transformation matrix.
pub fn invert_affine_transform(m: &[f32; 6]) -> [f32; 6] {
    let (a, b, c, d, e, f) = (m[0], m[1], m[2], m[3], m[4], m[5]);

    // follow OpenCV: check for determinant == 0
    // https://github.com/opencv/opencv/blob/4.9.0/modules/imgproc/src/imgwarp.cpp#L2765
    let determinant = a * e - b * d;
    let inv_determinant = if determinant != 0.0 {
        1.0 / determinant
    } else {
        0.0
    };

    let new_a = e * inv_determinant;
    let new_b = -b * inv_determinant;
    let new_d = -d * inv_determinant;
    let new_e = a * inv_determinant;
    let new_c = -(new_a * c + new_b * f);
    let new_f = -(new_d * c + new_e * f);

    [new_a, new_b, new_c, new_d, new_e, new_f]
}

/// Returns a 2x3 rotation matrix for a 2D rotation around a center point.
///
/// The rotation matrix is defined as:
///
/// | alpha  beta  tx |
/// | -beta  alpha ty |
///
/// where:
///
/// alpha = scale * cos(angle)
/// beta = scale * sin(angle)
/// tx = (1 - alpha) * center.x - beta * center.y
/// ty = beta * center.x + (1 - alpha) * center.y
///
/// # Arguments
///
/// * `center` - The center point of the rotation.
/// * `angle` - The angle of rotation in degrees.
/// * `scale` - The scale factor.
///
/// # Example
///
/// ```
/// use kornia_imgproc::warp::get_rotation_matrix2d;
///
/// let center = (0.0, 0.0);
/// let angle = 90.0;
/// let scale = 1.0;
/// let rotation_matrix = get_rotation_matrix2d(center, angle, scale);
/// ```
pub fn get_rotation_matrix2d(center: (f32, f32), angle: f32, scale: f32) -> [f32; 6] {
    let angle = angle * std::f32::consts::PI / 180.0f32;
    let alpha = scale * angle.cos();
    let beta = scale * angle.sin();

    let tx = (1.0 - alpha) * center.0 - beta * center.1;
    let ty = beta * center.0 + (1.0 - alpha) * center.1;

    [alpha, beta, tx, -beta, alpha, ty]
}

/// Applies an affine transformation to a point.
fn transform_point(x: f32, y: f32, m: &[f32; 6]) -> (f32, f32) {
    let u = m[0] * x + m[1] * y + m[2];
    let v = m[3] * x + m[4] * y + m[5];
    (u, v)
}

/// Applies an affine transformation to an image.
///
/// # Arguments
///
/// * `src` - The input image with shape (height, width, channels).
/// * `dst` - The output image with shape (height, width, channels).
/// * `m` - The 2x3 affine transformation matrix.
/// * `interpolation` - The interpolation mode to use.
/// * `device` - The target computation device `KorniaDevice`.
///
/// # Returns
///
/// The output image with shape (new_height, new_width, channels).
///
/// # Errors
///
/// Returns [`ImageError::InvalidInterpolation`] if the interpolation is not supported.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::interpolation::InterpolationMode;
/// use kornia_imgproc::warp::warp_affine;
/// use kornia_imgproc::KorniaDevice;
///
/// let src = Image::<_, 3, _>::from_size_val(
///    ImageSize {
///       width: 4,
///      height: 5,
///  },
///  1f32,
///  CpuAllocator
/// ).unwrap();
///
/// let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let new_size = ImageSize {
///    width: 4,
///   height: 5,
/// };
///
/// let mut dst = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
///
/// warp_affine(&src, &mut dst, &m, InterpolationMode::Nearest, KorniaDevice::Cpu).unwrap();
///
/// assert_eq!(dst.size().width, 4);
/// assert_eq!(dst.size().height, 5);
/// ```
pub fn warp_affine<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    m: &[f32; 6],
    interpolation: InterpolationMode,
    device: KorniaDevice,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;

    // invert affine transform matrix to find corresponding positions in src from dst
    let m_inv = invert_affine_transform(m);

    match device {
        KorniaDevice::Cpu => {
            // apply affine transformation without pre-allocating coordinate maps
            parallel::par_iter_rows_spatial_mapping(
                dst,
                |x, y| transform_point(x as f32, y as f32, &m_inv),
                |x, y, dst_pixel| {
                    // check if the position is within the bounds of the src image
                    if x >= 0.0f32 && x < src.cols() as f32 && y >= 0.0f32 && y < src.rows() as f32
                    {
                        // interpolate the pixel value for each channel
                        dst_pixel.iter_mut().enumerate().for_each(|(k, pixel)| {
                            *pixel = interpolate_pixel_fast(src, x, y, k, interpolation);
                        });
                    }
                },
            );
        }
        #[cfg(feature = "gpu")]
        KorniaDevice::Gpu(device_id) => {
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
            let client = cubecl::Runtime::client(&WgpuDevice::default());

            // Allocate tensors on the GPU
            let src_handle = client.create(src.as_slice());
            // Need a mutable slice to overwrite data
            let mut dst_vec =
                vec![0.0f32; dst.size().width * dst.size().height * dst.num_channels()];
            let dst_handle = client.create(&dst_vec);
            let m_inv_handle = client.create(&m_inv);

            // SAFETY: The WGPU buffers created match the sizes and strides of the source and destination images.
            // The GPU kernel inherently bounds-checks ABSOLUTE_POS coordinates before executing reads and writes.
            unsafe {
                warp_affine_kernel::launch::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::default(),
                    TensorArg::from_raw_parts(
                        src_handle.clone(),
                        vec![1, src.rows(), src.cols(), C],
                        vec![src.rows() * src.cols() * C, src.cols() * C, C, 1],
                        1,
                    ),
                    TensorArg::from_raw_parts(
                        dst_handle.clone(),
                        vec![1, dst.rows(), dst.cols(), C],
                        vec![dst.rows() * dst.cols() * C, dst.cols() * C, C, 1],
                        1,
                    ),
                    TensorArg::from_raw_parts(m_inv_handle, vec![6], vec![1], 1),
                    src.cols() as u32,
                    src.rows() as u32,
                    dst.cols() as u32,
                    dst.rows() as u32,
                    C as u32,
                );
            }

            let dst_bytes = client.read(dst_handle.binding());
            let dst_data: &[f32] = bytemuck::cast_slice(&dst_bytes);
            dst.as_mut_slice().copy_from_slice(dst_data);
        }
        #[cfg(not(feature = "gpu"))]
        KorniaDevice::Gpu(_) => {
            panic!("Please enable the `gpu` feature flag to compute on GPU.");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;
    #[test]
    fn warp_affine_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3, _>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5 * 3],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::<_, 3, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_affine(
            &image,
            &mut image_transformed,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            super::InterpolationMode::Bilinear,
            crate::KorniaDevice::Cpu,
        )?;

        assert_eq!(image_transformed.num_channels(), 3);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        Ok(())
    }

    #[test]
    fn warp_affine_unsupported_interpolation() -> Result<(), ImageError> {
        let src = Image::<_, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0f32,
            CpuAllocator,
        )?;
        let mut dst = Image::<_, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0f32,
            CpuAllocator,
        )?;
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = super::warp_affine(
            &src,
            &mut dst,
            &m,
            super::InterpolationMode::Bicubic,
            crate::KorniaDevice::Cpu,
        );
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn warp_affine_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_affine(
            &image,
            &mut image_transformed,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            super::InterpolationMode::Nearest,
            crate::KorniaDevice::Cpu,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        Ok(())
    }

    #[test]
    fn warp_affine_correctness_identity() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            (0..20).map(|x| x as f32).collect(),
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 4,
            height: 5,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_affine(
            &image,
            &mut image_transformed,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            super::InterpolationMode::Nearest,
            crate::KorniaDevice::Cpu,
        )?;

        assert_eq!(image_transformed.as_slice(), image.as_slice());
        assert_eq!(image_transformed.size(), image.size());

        Ok(())
    }

    #[test]
    fn warp_affine_correctness_rot90() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32, 1.0f32, 2.0f32, 3.0f32],
            CpuAllocator,
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 2,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_affine(
            &image,
            &mut image_transformed,
            &super::get_rotation_matrix2d((0.5, 0.5), 90.0, 1.0),
            super::InterpolationMode::Nearest,
            crate::KorniaDevice::Cpu,
        )?;

        assert_eq!(
            image_transformed.as_slice(),
            &[1.0f32, 3.0f32, 0.0f32, 2.0f32]
        );

        Ok(())
    }
}
