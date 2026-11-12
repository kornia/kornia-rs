use kornia_image::Image;

/// Kernel for nearest neighbor interpolation
///
/// Bounds-safe for any coordinate: the tap is clamped into the image and read
/// with a bounds-checked slice index (negative / NaN coordinates saturate to
/// tap 0). An empty image or an out-of-range channel returns `0.0`; public entry
/// points (`interpolate_pixel`, `remap`) report those as errors first.
///
/// # Arguments
///
/// * `image` - The input image container. Empty images yield `0.0`.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate. Out-of-range channels yield `0.0`.
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
    // Cheap, well-predicted guard so release builds never read a neighbouring
    // pixel's channel or index an empty image, whatever the caller did.
    if rows == 0 || cols == 0 || c >= C {
        return 0.0;
    }

    let iu = (u.round() as usize).min(cols - 1);
    let iv = (v.round() as usize).min(rows - 1);

    image.as_slice()[(iv * cols + iu) * C + c]
}
