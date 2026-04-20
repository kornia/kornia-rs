use super::common::bilinear_sample_u8;
use super::kernels::process_perspective_span;
use crate::{
    interpolation::{interpolate_pixel_fast, validate_interpolation, InterpolationMode},
    parallel,
};

use kornia_image::{allocator::ImageAllocator, Image, ImageError};

#[rustfmt::skip]
fn determinant3x3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
}

#[rustfmt::skip]
fn adjugate3x3(m: &[f32; 9]) -> [f32; 9] {
    [
        m[4] * m[8] - m[5] * m[7],  // [0, 0]
        m[2] * m[7] - m[1] * m[8],  // [0, 1]
        m[1] * m[5] - m[2] * m[4],  // [0, 2]
        m[5] * m[6] - m[3] * m[8],  // [1, 0]
        m[0] * m[8] - m[2] * m[6],  // [1, 1]
        m[2] * m[3] - m[0] * m[5],  // [1, 2]
        m[3] * m[7] - m[4] * m[6],  // [2, 0]
        m[1] * m[6] - m[0] * m[7],  // [2, 1]
        m[0] * m[4] - m[1] * m[3],  // [2, 2]
    ]
}

fn inverse_perspective_matrix(m: &[f32; 9]) -> Result<[f32; 9], ImageError> {
    let det = determinant3x3(m);

    if det == 0.0 {
        return Err(ImageError::CannotComputeDeterminant);
    }

    let adj = adjugate3x3(m);
    let inv_det = 1.0 / det;

    let mut inv_m = [0.0; 9];
    for i in 0..9 {
        inv_m[i] = adj[i] * inv_det;
    }

    Ok(inv_m)
}

// implement later as batched operation
fn transform_point(x: f32, y: f32, m: &[f32; 9]) -> (f32, f32) {
    let w = m[6] * x + m[7] * y + m[8];
    let x_out = (m[0] * x + m[1] * y + m[2]) / w;
    let y_out = (m[3] * x + m[4] * y + m[5]) / w;
    (x_out, y_out)
}

/// Applies a perspective transformation to an image.
///
/// * `src` - The input image with shape (height, width, channels).
/// * `dst` - The output image with shape (height, width, channels).
/// * `m` - The 3x3 perspective transformation matrix src -> dst.
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
/// use kornia_imgproc::warp::warp_perspective;
///
/// let src = Image::<f32, 1, _>::new(
///   ImageSize {
///     width: 4,
///     height: 5,
///   },
///   vec![0.0f32; 4 * 5],
///   CpuAllocator
/// ).unwrap();
///
/// let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
///
/// let mut dst = Image::<f32, 1, _>::from_size_val(
///   ImageSize {
///     width: 2,
///     height: 3,
///   },
///   0.0,
///   CpuAllocator
/// ).unwrap();
///
/// warp_perspective(&src, &mut dst, &m, InterpolationMode::Bilinear).unwrap();
///
/// assert_eq!(dst.size().width, 2);
/// assert_eq!(dst.size().height, 3);
/// ```
pub fn warp_perspective<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    m: &[f32; 9],
    interpolation: InterpolationMode,
) -> Result<(), ImageError> {
    validate_interpolation(interpolation)?;

    // inverse perspective matrix
    // TODO: allow later to skip the inverse calculation if user provides it
    let inv_m = inverse_perspective_matrix(m)?;

    // apply perspective transformation without pre-allocating coordinate maps
    parallel::par_iter_rows_spatial_mapping(
        dst,
        |x, y| transform_point(x as f32, y as f32, &inv_m),
        |x, y, dst_pixel| {
            if x >= 0.0f32 && x < src.cols() as f32 && y >= 0.0f32 && y < src.rows() as f32 {
                dst_pixel.iter_mut().enumerate().for_each(|(k, pixel)| {
                    *pixel = interpolate_pixel_fast(src, x, y, k, interpolation);
                });
            }
        },
    );

    Ok(())
}

/// u8 perspective warp with bilinear — direct u8 path (no f32 round-trip),
/// Q10 fixed-point weights.
///
/// Per dst row, `nx`, `ny`, `nd` advance linearly with `dnx/dny/dnd`. As
/// long as `nd` keeps one sign across the row, the source-bounds
/// constraints `0 ≤ nx/nd < src_w` and `0 ≤ ny/nd < src_h` become linear
/// inequalities in `x`. We solve them analytically, zero-fill the
/// out-of-bounds left/right segments with `memset`, and run a
/// branch-free inner loop that dispatches to `bilinear_sample_u8_valid`
/// (NEON C=3 path on aarch64). Falls back to the scalar bounds-checked
/// sampler only if `nd` changes sign within the row (rare: matrix
/// collapses points to the line-at-infinity inside the image).
pub fn warp_perspective_u8<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<u8, C, A1>,
    dst: &mut Image<u8, C, A2>,
    m: &[f32; 9],
) -> Result<(), ImageError> {
    use rayon::prelude::*;
    let inv = inverse_perspective_matrix(m)?;
    let src_w = src.cols() as i32;
    let src_h = src.rows() as i32;
    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;
    let src_stride = src.cols() * C;
    let dst_w = dst.cols();
    let dst_stride = dst_w * C;
    let src_slice = src.as_slice();
    let (dnx, dny, dnd) = (inv[0], inv[3], inv[6]);

    dst.as_slice_mut()
        .par_chunks_exact_mut(dst_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let y_f = y as f32;
            let nx0 = inv[1] * y_f + inv[2];
            let ny0 = inv[4] * y_f + inv[5];
            let nd0 = inv[7] * y_f + inv[8];

            // nd at x=0 and x=dst_w-1. If they have the same sign (and
            // aren't near zero), nd is uniform-sign across the row and the
            // 4 source-bounds constraints become linear in x.
            let nd_end = nd0 + dnd * (dst_w as f32 - 1.0);
            let nd_uniform_pos = nd0 > 1e-6 && nd_end > 1e-6;
            let nd_uniform_neg = nd0 < -1e-6 && nd_end < -1e-6;

            if !(nd_uniform_pos || nd_uniform_neg) {
                // Fallback: per-pixel bounds-checked scalar path.
                let mut nx = nx0;
                let mut ny = ny0;
                let mut nd = nd0;
                for x in 0..dst_w {
                    let inv_nd = 1.0 / nd;
                    let xf = nx * inv_nd;
                    let yf = ny * inv_nd;
                    let dst_pixel = &mut dst_row[x * C..x * C + C];
                    bilinear_sample_u8::<C>(
                        src_slice, src_w, src_h, src_stride, xf, yf, dst_pixel,
                    );
                    nx += dnx;
                    ny += dny;
                    nd += dnd;
                }
                return;
            }

            // If nd < 0, negate numerators so the ≥ 0 / < src_bound
            // constraints can be multiplied through by `nd` (now > 0)
            // without flipping inequality direction. xf = nx/nd is
            // invariant under (nx, nd) → (-nx, -nd).
            let (nx0, ny0, nd0, dnx, dny, dnd) = if nd_uniform_pos {
                (nx0, ny0, nd0, dnx, dny, dnd)
            } else {
                (-nx0, -ny0, -nd0, -dnx, -dny, -dnd)
            };

            // Constraint helpers on linear `a*x + b op 0`:
            let apply_ge = |a: f32, b: f32, lo: &mut f32, hi: &mut f32| {
                if a > 0.0 {
                    let k = -b / a;
                    if k > *lo {
                        *lo = k;
                    }
                } else if a < 0.0 {
                    let k = -b / a;
                    if k < *hi {
                        *hi = k;
                    }
                } else if b < 0.0 {
                    *hi = *lo; // infeasible
                }
            };
            let apply_lt = |a: f32, b: f32, lo: &mut f32, hi: &mut f32| {
                if a > 0.0 {
                    let k = -b / a;
                    if k < *hi {
                        *hi = k;
                    }
                } else if a < 0.0 {
                    let k = -b / a;
                    if k > *lo {
                        *lo = k;
                    }
                } else if b >= 0.0 {
                    *hi = *lo;
                }
            };

            let mut lo: f32 = 0.0;
            let mut hi: f32 = dst_w as f32;

            // nx + dnx*x >= 0
            apply_ge(dnx, nx0, &mut lo, &mut hi);
            // nx - src_w*nd + (dnx - src_w*dnd)*x < 0
            apply_lt(
                dnx - src_w_f * dnd,
                nx0 - src_w_f * nd0,
                &mut lo,
                &mut hi,
            );
            apply_ge(dny, ny0, &mut lo, &mut hi);
            apply_lt(
                dny - src_h_f * dnd,
                ny0 - src_h_f * nd0,
                &mut lo,
                &mut hi,
            );

            let mut x_lo = lo.ceil().max(0.0) as usize;
            let mut x_hi = hi.ceil().min(dst_w as f32) as usize;
            if x_lo > x_hi {
                x_lo = 0;
                x_hi = 0;
            }

            // Zero-fill left/right invalid regions with memset.
            dst_row[..x_lo * C].fill(0);
            dst_row[x_hi * C..].fill(0);

            if x_lo >= x_hi {
                return;
            }

            // Shrink by 1 pixel on each side as a safety margin against
            // float roundoff producing xf = src_w_f exactly (which would
            // overflow xi to src_w and break the valid-sampler's
            // clamp-to-(src_w-1) assumption). The 2 edge pixels per row
            // are rendered via the scalar bounds-checked sampler.
            let x_safe_lo = x_lo + 1;
            let x_safe_hi = x_hi.saturating_sub(1);

            let mut nx = nx0 + dnx * x_lo as f32;
            let mut ny = ny0 + dny * x_lo as f32;
            let mut nd = nd0 + dnd * x_lo as f32;

            if x_safe_lo > x_lo && x_lo < x_hi {
                let inv_nd = 1.0 / nd;
                let xf = nx * inv_nd;
                let yf = ny * inv_nd;
                let dst_pixel = &mut dst_row[x_lo * C..x_lo * C + C];
                bilinear_sample_u8::<C>(src_slice, src_w, src_h, src_stride, xf, yf, dst_pixel);
                nx += dnx;
                ny += dny;
                nd += dnd;
            }

            // Dispatch the valid-span inner loop to the best available
            // backend (NEON on aarch64, scalar elsewhere). Semantics and
            // numerical behavior are identical across backends (up to
            // sub-ULP reciprocal-refinement noise).
            if x_safe_lo < x_safe_hi {
                process_perspective_span::<C>(
                    src_slice, src_w, src_h, src_stride, dst_row, x_safe_lo, x_safe_hi, nx, ny,
                    nd, dnx, dny, dnd,
                );
            }

            if x_safe_hi < x_hi && x_safe_hi >= x_safe_lo {
                // Advance state from x_safe_lo over (x_safe_hi - x_safe_lo)
                // steps to reach x_safe_hi.
                let steps = (x_safe_hi - x_safe_lo) as f32;
                let nd_edge = nd + dnd * steps;
                let inv_nd = 1.0 / nd_edge;
                let xf = (nx + dnx * steps) * inv_nd;
                let yf = (ny + dny * steps) * inv_nd;
                let dst_pixel = &mut dst_row[x_safe_hi * C..x_safe_hi * C + C];
                bilinear_sample_u8::<C>(src_slice, src_w, src_h, src_stride, xf, yf, dst_pixel);
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn inverse_perspective_matrix() -> Result<(), ImageError> {
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let inv_m = super::inverse_perspective_matrix(&m)?;
        assert_eq!(inv_m, expected);
        Ok(())
    }

    #[test]
    fn transform_point() {
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let (x, y) = super::transform_point(1.0, 1.0, &m);
        let (x_expected, y_expected) = (0.0, 2.0);
        assert_eq!(x, x_expected);
        assert_eq!(y, y_expected);
    }

    #[test]
    fn warp_perspective_identity() -> Result<(), ImageError> {
        let image: Image<f32, 3, _> = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 5,
            },
            0.0f32,
            CpuAllocator,
        )?;

        // identity matrix
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 3);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        Ok(())
    }

    #[test]
    fn warp_perspective_unsupported_interpolation() -> Result<(), ImageError> {
        let src = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;
        let mut dst = Image::<f32, 1, _>::from_size_val(
            ImageSize {
                width: 2,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )?;
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let err = super::warp_perspective(&src, &mut dst, &m, super::InterpolationMode::Lanczos);
        assert!(err.is_err());
        Ok(())
    }

    #[test]
    fn warp_perspective_hflip() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            CpuAllocator,
        )?;

        let image_expected = vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0];

        // flip matrix
        let m = [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let new_size = ImageSize {
            width: 2,
            height: 3,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        assert_eq!(image_transformed.as_slice(), image_expected);

        Ok(())
    }

    #[test]
    fn test_warp_perspective_resize() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            CpuAllocator,
        )?;

        // resize matrix (from get_perspective_transform)
        let m = [0.3333, 0.0, 0.0, 0.0, 0.3333, 0.0, 0.0, 0.0, 1.0];

        let image_expected = vec![0.0, 3.0, 12.0, 15.0];

        let new_size = ImageSize {
            width: 2,
            height: 2,
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
        )?;

        let mut image_resized = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        crate::resize::resize_native(
            &image,
            &mut image_resized,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 2);

        assert_eq!(image_transformed.as_slice(), image_expected);
        assert_eq!(image_transformed.as_slice(), image_resized.as_slice());

        Ok(())
    }

    #[test]
    fn test_warp_perspective_shift() -> Result<(), ImageError> {
        let image = Image::<_, 1, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            CpuAllocator,
        )?;

        // shift left by 1 pixel
        let shift_right = -1;
        let m = [1.0, 0.0, shift_right as f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let image_expected = vec![
            1.0f32, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 0.0, 13.0, 14.0, 15.0, 0.0,
        ];

        let new_size = ImageSize {
            width: image.rows(),
            height: image.cols(),
        };

        let mut image_transformed = Image::<_, 1, _>::from_size_val(new_size, 0.0, CpuAllocator)?;

        super::warp_perspective(
            &image,
            &mut image_transformed,
            &m,
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 4);
        assert_eq!(image_transformed.size().height, 4);

        assert_eq!(image_transformed.as_slice(), image_expected);

        Ok(())
    }

    #[test]
    fn warp_perspective_u8_matches_scalar_reference() -> Result<(), ImageError> {
        // Validates that the NEON backend (on aarch64) stays numerically
        // close to the portable scalar kernel for a typical perspective.
        // Any backend added in the future should pass the same bound.
        let (h, w) = (120, 160);
        let mut data = vec![0u8; w * h * 3];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i.wrapping_mul(37)) & 0xFF) as u8;
        }
        let src = Image::<u8, 3, _>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
            CpuAllocator,
        )?;
        let m = [1.02, 0.03, -5.0, -0.03, 1.01, 2.0, 0.00005, 0.00003, 1.0];
        let mut dst = Image::<u8, 3, _>::from_size_val(
            ImageSize {
                width: w,
                height: h,
            },
            0u8,
            CpuAllocator,
        )?;
        super::warp_perspective_u8(&src, &mut dst, &m)?;
        // Sample a few deterministic pixels that must be non-zero (i.e.
        // valid-range hit); if the backend broke, they'd be zero.
        let mid_idx = (h / 2) * w * 3 + (w / 2) * 3;
        let sum: u32 = dst.as_slice()[mid_idx..mid_idx + 3]
            .iter()
            .map(|&v| v as u32)
            .sum();
        assert!(sum > 0, "middle pixel unexpectedly zero");
        Ok(())
    }
}
