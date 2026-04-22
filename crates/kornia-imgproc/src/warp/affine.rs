use std::f32::consts::PI;

use kornia_image::allocator::ImageAllocator;
use kornia_image::{Image, ImageError};

use super::kernels::process_affine_span;
use crate::interpolation::{interpolate_pixel_fast, validate_interpolation, InterpolationMode};
use crate::parallel;

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
    let angle = angle * PI / 180.0f32;
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
///
/// # Returns
///
/// The output image with shape (new_height, new_width, channels).
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::interpolation::InterpolationMode;
/// use kornia_imgproc::warp::warp_affine;
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
/// warp_affine(&src, &mut dst, &m, InterpolationMode::Nearest).unwrap();
///
/// assert_eq!(dst.size().width, 4);
/// assert_eq!(dst.size().height, 5);
/// ```
pub fn warp_affine<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    m: &[f32; 6],
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;

    // invert affine transform matrix to find corresponding positions in src from dst
    let m_inv = invert_affine_transform(m);

    // apply affine transformation without pre-allocating coordinate maps
    parallel::par_iter_rows_spatial_mapping(
        dst,
        |x, y| transform_point(x as f32, y as f32, &m_inv),
        |x, y, dst_pixel| {
            // check if the position is within the bounds of the src image
            if x >= 0.0f32 && x < src.cols() as f32 && y >= 0.0f32 && y < src.rows() as f32 {
                // interpolate the pixel value for each channel
                dst_pixel.iter_mut().enumerate().for_each(|(k, pixel)| {
                    *pixel = interpolate_pixel_fast(src, x, y, k, interpolation);
                });
            }
        },
    );

    Ok(())
}

/// u8 warp-affine with bilinear interpolation — direct u8 path avoiding
/// f32 round-trip. Uses rowwise incremental coordinate update and Q10
/// fixed-point bilinear weights.
///
/// `m` is the forward 2x3 transform; this function inverts it.
pub fn warp_affine_u8<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    m: &[f32; 6],
) -> Result<(), ImageError> {
    use rayon::prelude::*;
    let m_inv = invert_affine_transform(m);
    let src_w = src.cols() as i32;
    let src_h = src.rows() as i32;
    let src_stride = src.cols() * C;
    let dst_w = dst.cols();
    let dst_stride = dst_w * C;
    let src_slice = src.as_slice();

    // Q16 fixed-point coords for the inner loop: replaces per-pixel
    // `.floor() as i32` (frintm+fcvtzs) with a single arithmetic shift.
    const Q: i32 = 16;
    const Q_SCALE: f32 = (1 << Q) as f32;
    let dsx = m_inv[0];
    let dsy = m_inv[3];
    let dsx_q = (dsx * Q_SCALE) as i32;
    let dsy_q = (dsy * Q_SCALE) as i32;
    // Valid iff `0 <= xi < src_w`, i.e. src coord in `[0, src_w)`. The
    // sampler clamps `xi+1` to `src_w-1` (BORDER_REPLICATE), so exact-edge
    // integer coords (fx=0) produce `src[yi, xi]` — matching the f32
    // reference kernel's identity behavior. Same for y.
    let sx_upper = src_w as f32;
    let sy_upper = src_h as f32;

    dst.as_slice_mut()
        .par_chunks_exact_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let y_f = y as f32;
            let sx0 = m_inv[1] * y_f + m_inv[2];
            let sy0 = m_inv[4] * y_f + m_inv[5];

            // Analytical valid-x range so the inner loop is branch-free.
            // Solve: 0 <= dsx*x + sx0 < sx_upper; same for y; intersect.
            let (mut x_lo, mut x_hi) = (0i64, dst_w as i64);
            // For the x-axis constraint: 0 <= dsx*x + sx0 < sx_upper.
            if dsx.abs() < 1e-12 {
                if !(sx0 >= 0.0 && sx0 < sx_upper) {
                    x_lo = 0;
                    x_hi = 0;
                }
            } else {
                let a = (-sx0) / dsx;
                let b = (sx_upper - sx0) / dsx;
                let (lo_f, hi_f) = if dsx > 0.0 { (a, b) } else { (b, a) };
                // x in [ceil(lo_f), ceil(hi_f)): strict < hi means the first
                // invalid x is ceil(hi_f) when hi_f is non-integer, else hi_f.
                let new_lo = lo_f.ceil().max(0.0) as i64;
                let new_hi = hi_f.ceil().min(dst_w as f32) as i64;
                x_lo = x_lo.max(new_lo);
                x_hi = x_hi.min(new_hi);
            }
            if dsy.abs() < 1e-12 {
                if !(sy0 >= 0.0 && sy0 < sy_upper) {
                    x_lo = 0;
                    x_hi = 0;
                }
            } else {
                let a = (-sy0) / dsy;
                let b = (sy_upper - sy0) / dsy;
                let (lo_f, hi_f) = if dsy > 0.0 { (a, b) } else { (b, a) };
                let new_lo = lo_f.ceil().max(0.0) as i64;
                let new_hi = hi_f.ceil().min(dst_w as f32) as i64;
                x_lo = x_lo.max(new_lo);
                x_hi = x_hi.min(new_hi);
            }
            if x_lo > x_hi {
                x_lo = 0;
                x_hi = 0;
            }

            // Left/right invalid ranges: zero-fill via memset.
            let x_lo_u = x_lo as usize;
            let x_hi_u = x_hi as usize;
            dst_row[..x_lo_u * C].fill(0);
            dst_row[x_hi_u * C..].fill(0);

            if x_lo_u >= x_hi_u {
                return;
            }

            // Q16 coord at x_lo.
            let sx_q_lo = ((sx0 + dsx * x_lo_u as f32) * Q_SCALE) as i32;
            let sy_q_lo = ((sy0 + dsy * x_lo_u as f32) * Q_SCALE) as i32;

            // Branch-free inner loop over the valid region (dispatched
            // to the best backend by the kernels module).
            process_affine_span::<C>(
                src_slice, src_w, src_h, src_stride, dst_row, x_lo_u, x_hi_u, sx_q_lo, sy_q_lo,
                dsx_q, dsy_q,
            );
        });

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
        let err = super::warp_affine(&src, &mut dst, &m, super::InterpolationMode::Bicubic);
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
        )?;

        assert_eq!(
            image_transformed.as_slice(),
            &[1.0f32, 3.0f32, 0.0f32, 2.0f32]
        );

        Ok(())
    }
}
