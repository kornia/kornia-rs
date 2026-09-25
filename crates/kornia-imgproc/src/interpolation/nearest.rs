use kornia_image::Image;

/// Kernel for nearest neighbor interpolation
///
/// Bounds-safe for any coordinate: the tap is clamped into the image and read
/// with a bounds-checked slice index (negative / NaN coordinates saturate to
/// tap 0). Validating the channel and the image extent is the caller's job
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
pub(crate) fn nearest_neighbor_interpolation<const C: usize>(
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());
    debug_assert!(rows > 0 && cols > 0 && c < C);

    let iu = (u.round() as usize).min(cols - 1);
    let iv = (v.round() as usize).min(rows - 1);

    image.as_slice()[(iv * cols + iu) * C + c]
}
