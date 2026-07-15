use kornia_image::Image;

/// Kernel for bilinear interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel value.
// TODO: add support for other data types. Maybe use a trait? or template?
pub(crate) fn bilinear_interpolation<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());

    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let frac_u = u.fract();
    let frac_v = v.fract();
    let val00 = *image.get_unchecked([iv, iu, c]);
    let val01 = if iu + 1 < cols {
        *image.get_unchecked([iv, iu + 1, c])
    } else {
        val00
    };
    let val10 = if iv + 1 < rows {
        *image.get_unchecked([iv + 1, iu, c])
    } else {
        val00
    };
    let val11 = if iu + 1 < cols && iv + 1 < rows {
        *image.get_unchecked([iv + 1, iu + 1, c])
    } else {
        val00
    };

    let frac_uu = 1. - frac_u;
    let frac_vv = 1. - frac_v;

    // Weights are formed first and the terms summed left to right — the same
    // expression shape as `resize_bilinear_downscale_3c` and
    // `resize_bilinear_normalize_3c` in `cuda/resize.rs`, which compile with
    // `--fmad=false` (uncontracted multiply-adds). Same ops in the same order
    // means bit-identical results, which the CPU/GPU resize parity tests
    // assert. `(val * w1) * w2` is NOT the same rounding as `val * (w1 * w2)`;
    // keep this shape in sync with those kernels.
    //
    // Naming: the weight digit order is (y, x) while the val digit order is
    // (row-offset, col-offset), so w10 (y+0, x+1) pairs with val01 (row+0,
    // col+1) — transposed names, same tap.
    let w00 = frac_vv * frac_uu;
    let w10 = frac_vv * frac_u;
    let w01 = frac_v * frac_uu;
    let w11 = frac_v * frac_u;

    w00 * val00 + w10 * val01 + w01 * val10 + w11 * val11
}
