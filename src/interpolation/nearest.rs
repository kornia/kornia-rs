use crate::image::ImageDtype;
use ndarray::Array3;

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
pub(crate) fn nearest_neighbor_interpolation<T: ImageDtype>(
    image: &Array3<T>,
    u: f32,
    v: f32,
    c: usize,
) -> T {
    let (height, width, _) = image.dim();

    let iu = u.round() as usize;
    let iv = v.round() as usize;

    let iu = iu.clamp(0, width - 1);
    let iv = iv.clamp(0, height - 1);

    image[[iv, iu, c]]
}
