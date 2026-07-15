use std::f32::consts::PI;

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

use super::kernels::process_affine_span;
use crate::interpolation::{validate_interpolation, InterpolationMode};

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
/// use kornia_imgproc::interpolation::InterpolationMode;
/// use kornia_imgproc::warp::warp_affine;
///
/// let src = Image::<_, 3>::from_size_val(
///    ImageSize {
///       width: 4,
///      height: 5,
///  },
///  1f32,
/// ).unwrap();
///
/// let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
/// let new_size = ImageSize {
///    width: 4,
///   height: 5,
/// };
///
/// let mut dst = Image::<_, 3>::from_size_val(new_size, 0.0).unwrap();
///
/// warp_affine(&src, &mut dst, &m, InterpolationMode::Nearest).unwrap();
///
/// assert_eq!(dst.size().width, 4);
/// assert_eq!(dst.size().height, 5);
/// ```
pub fn warp_affine<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    m: &[f32; 6],
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;

    let m_inv = invert_affine_transform(m);

    let src_w = src.cols();
    let src_h = src.rows();
    let dst_w = dst.cols();
    let row_len = dst_w * C;

    // Per-column increments (affine: same for every row).
    let dsx = m_inv[0];
    let dsy = m_inv[3];
    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;

    // Analytical valid-x range: solve 0 <= dsx*x + sx0 < src_w and
    // 0 <= dsy*x + sy0 < src_h for x, intersected to [x_lo, x_hi) — see
    // `warp::span` for the inclusive/strict boundary rules.
    //
    // 1e-6 degenerate-step threshold covers f32 trig imprecision (e.g.
    // cos(π/2) ≈ −4.4e-8); assumes source step ≥ ~1e-6 px/col (extreme
    // downscale > 1e6:1 is not a use case here).
    let compute_range = |sx0: f32, sy0: f32| -> (usize, usize) {
        super::span::affine_valid_span([(dsx, sx0, src_w_f), (dsy, sy0, src_h_f)], dst_w, 1e-6)
    };

    // 16 rows per Rayon task — same granularity as parallel.rs — keeps spawn
    // overhead low without sacrificing work-stealing balance.
    const ROWS_PER_TASK: usize = 16;
    let src_slice = src.as_slice();

    // Lift the interpolation branch outside the parallel loop so each task is
    // branch-free in its inner loop.
    macro_rules! run_rows {
        ($chunk:expr, $y_base:expr, $inner:expr) => {
            for (dy, dst_row) in $chunk.chunks_exact_mut(row_len).enumerate() {
                let y_f = ($y_base + dy) as f32;
                let sx0 = m_inv[1] * y_f + m_inv[2];
                let sy0 = m_inv[4] * y_f + m_inv[5];
                let (x_lo, x_hi) = compute_range(sx0, sy0);
                dst_row[..x_lo * C].fill(0.0);
                dst_row[x_hi * C..].fill(0.0);
                if x_lo < x_hi {
                    let mut sx = sx0 + dsx * x_lo as f32;
                    let mut sy = sy0 + dsy * x_lo as f32;
                    $inner(dst_row, x_lo, x_hi, &mut sx, &mut sy);
                }
            }
        };
    }

    match interpolation {
        InterpolationMode::Nearest => {
            dst.as_slice_mut()
                .par_chunks_mut(row_len * ROWS_PER_TASK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    run_rows!(
                        chunk,
                        ci * ROWS_PER_TASK,
                        |dst_row: &mut [f32],
                         x_lo: usize,
                         x_hi: usize,
                         sx: &mut f32,
                         sy: &mut f32| {
                            for dst_pixel in dst_row[x_lo * C..x_hi * C].chunks_exact_mut(C) {
                                let xi = sx.round().clamp(0.0, src_w_f - 1.0) as usize;
                                let yi = sy.round().clamp(0.0, src_h_f - 1.0) as usize;
                                dst_pixel.copy_from_slice(&src_slice[(yi * src_w + xi) * C..][..C]);
                                *sx += dsx;
                                *sy += dsy;
                            }
                        }
                    );
                });
        }
        InterpolationMode::Bilinear => {
            dst.as_slice_mut()
                .par_chunks_mut(row_len * ROWS_PER_TASK)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    run_rows!(
                        chunk,
                        ci * ROWS_PER_TASK,
                        |dst_row: &mut [f32],
                         x_lo: usize,
                         x_hi: usize,
                         sx: &mut f32,
                         sy: &mut f32| {
                            for dst_pixel in dst_row[x_lo * C..x_hi * C].chunks_exact_mut(C) {
                                let sx_c = sx.clamp(0.0, src_w_f - 1.0);
                                let sy_c = sy.clamp(0.0, src_h_f - 1.0);
                                let x0 = sx_c as usize;
                                let y0 = sy_c as usize;
                                let x1 = (x0 + 1).min(src_w - 1);
                                let y1 = (y0 + 1).min(src_h - 1);
                                let fx = sx_c - x0 as f32;
                                let fy = sy_c - y0 as f32;
                                let w00 = (1.0 - fy) * (1.0 - fx);
                                let w10 = (1.0 - fy) * fx;
                                let w01 = fy * (1.0 - fx);
                                let w11 = fy * fx;
                                let b00 = (y0 * src_w + x0) * C;
                                let b10 = (y0 * src_w + x1) * C;
                                let b01 = (y1 * src_w + x0) * C;
                                let b11 = (y1 * src_w + x1) * C;
                                for k in 0..C {
                                    dst_pixel[k] = w00 * src_slice[b00 + k]
                                        + w10 * src_slice[b10 + k]
                                        + w01 * src_slice[b01 + k]
                                        + w11 * src_slice[b11 + k];
                                }
                                *sx += dsx;
                                *sy += dsy;
                            }
                        }
                    );
                });
        }
        // validate_interpolation at the top of this function rejects every mode
        // other than Nearest and Bilinear, so this arm is unreachable.
        _ => unreachable!(),
    }

    Ok(())
}

/// u8 warp-affine with bilinear interpolation — direct u8 path avoiding
/// f32 round-trip. Uses rowwise incremental coordinate update and Q10
/// fixed-point bilinear weights.
///
/// `m` is the forward 2x3 transform; this function inverts it.
pub fn warp_affine_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
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

            // Analytical valid-x range so the inner loop is branch-free:
            // 0 <= dsx*x + sx0 < sx_upper (same for y), intersected — see
            // `warp::span` for the inclusive/strict boundary rules.
            let (x_lo_u, x_hi_u) = super::span::affine_valid_span(
                [(dsx, sx0, sx_upper), (dsy, sy0, sy_upper)],
                dst_w,
                1e-12,
            );
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

    /// A destination pixel whose source coordinate lands *exactly* on the top or
    /// left edge (`sx == 0.0` / `sy == 0.0`) is inside the image and must be
    /// sampled, not zero-filled.
    ///
    /// `compute_range` derives the valid column span analytically. When the step
    /// along an axis is negative, the inclusive/strict ends of the interval swap;
    /// an earlier revision applied `ceil` to both, which dropped the valid column
    /// whenever the boundary fell on an exact integer — zeroing a real pixel.
    /// Exact-integer boundaries are the common case for axis-aligned transforms,
    /// so this is not a corner case.
    ///
    /// Here a horizontal flip maps dst x=3 to src x=0.0 exactly; that column must
    /// carry the source's first column, not zeros.
    #[test]
    fn warp_affine_samples_pixels_exactly_on_the_edge() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 4,
            height: 2,
        };
        // Distinct non-zero values so a zero-fill is unmistakable.
        let src = Image::<f32, 1>::new(size, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
        let mut dst = Image::<f32, 1>::from_size_val(size, -1.0)?;

        // Horizontal flip: dst_x = 3 - src_x, so dst x=3 <- src x=0 (dsx = -1).
        let m = [-1.0, 0.0, 3.0, 0.0, 1.0, 0.0];
        super::warp_affine(
            &src,
            &mut dst,
            &m,
            crate::interpolation::InterpolationMode::Nearest,
        )?;

        assert_eq!(
            dst.as_slice(),
            &[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0],
            "flip must sample every column, including the one landing exactly on src x=0"
        );
        Ok(())
    }

    #[test]
    fn warp_affine_smoke_ch3() -> Result<(), ImageError> {
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5 * 3],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::<_, 3>::from_size_val(new_size, 0.0)?;

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
        let src = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0f32,
        )?;
        let mut dst = Image::<_, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0f32,
        )?;
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = super::warp_affine(&src, &mut dst, &m, super::InterpolationMode::Bicubic);
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn warp_affine_smoke_ch1() -> Result<(), ImageError> {
        use kornia_image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::<_, 1>::from_size_val(new_size, 0.0)?;

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
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            (0..20).map(|x| x as f32).collect(),
        )?;

        let new_size = ImageSize {
            width: 4,
            height: 5,
        };

        let mut image_transformed = Image::<_, 1>::from_size_val(new_size, 0.0)?;

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
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32, 1.0f32, 2.0f32, 3.0f32],
        )?;

        let new_size = ImageSize {
            width: 2,
            height: 2,
        };

        let mut image_transformed = Image::<_, 1>::from_size_val(new_size, 0.0)?;

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
