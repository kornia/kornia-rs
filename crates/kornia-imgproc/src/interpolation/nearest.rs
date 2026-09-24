use kornia_image::Image;

/// Kernel for nearest neighbor interpolation
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
pub(crate) fn nearest_neighbor_interpolation<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());

    // Empty image or out-of-range channel: nothing valid to sample.
    if rows == 0 || cols == 0 || c >= C {
        return 0.0;
    }

    // Negative / NaN coordinates saturate to 0 in the `as usize` cast; the
    // upper bound is clamped explicitly.
    let iu = (u.round() as usize).min(cols - 1);
    let iv = (v.round() as usize).min(rows - 1);

    // Row-major (H, W, C) read with a single slice bounds check (see bilinear).
    image
        .as_slice()
        .get((iv * cols + iu) * C + c)
        .copied()
        .unwrap_or(0.0)
}
