use kornia_image::Image;

/// Kernel for bilinear interpolation
///
/// This is the single bounds-safe boundary for f32 bilinear sampling: the
/// integer taps are clamped into the image and every read is a (bounds
/// checked) slice index, so any coordinate — negative, huge, NaN — stays in
/// bounds. Negative / NaN coordinates saturate to tap 0 in the `as usize`
/// cast. Validating the channel and the image extent is the caller's job
/// (see `interpolate_pixel`, `remap`).
///
/// # Arguments
///
/// * `image` - The input image container. Must be non-empty.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate. Must be `< C`.
///
/// # Returns
///
/// The interpolated pixel value.
// TODO: add support for other data types. Maybe use a trait? or template?
// Per-pixel, per-channel hot path: the bounds-check panic paths would
// otherwise push it past the inlining threshold at some call sites (an
// outlined call per channel measured ~+80% on the f32 bilinear resize).
#[inline(always)]
pub(crate) fn bilinear_interpolation<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());
    debug_assert!(rows > 0 && cols > 0 && c < C);

    let iu = (u.trunc() as usize).min(cols - 1);
    let iv = (v.trunc() as usize).min(rows - 1);

    let frac_u = u.fract();
    let frac_v = v.fract();

    // Row-major (H, W, C). A neighbour past the last column/row replicates
    // `val00` (the historical rule the CUDA kernels mirror — note `val11`
    // falls back to `val00`, not to `val01`/`val10`).
    let data = image.as_slice();
    let i00 = (iv * cols + iu) * C + c;
    let has_right = iu + 1 < cols;
    let has_down = iv + 1 < rows;
    let row_step = cols * C;
    let val00 = data[i00];
    let val01 = if has_right { data[i00 + C] } else { val00 };
    let val10 = if has_down {
        data[i00 + row_step]
    } else {
        val00
    };
    let val11 = if has_right && has_down {
        data[i00 + row_step + C]
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
