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
    //image: &ArrayView3<T>,
    image: &Image<f32, C>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    //let (height, width, _) = image.dim();
    let (height, width) = (image.height(), image.width());

    let iu = u.round() as usize;
    let iv = v.round() as usize;

    let iu = iu.clamp(0, width - 1);
    let iv = iv.clamp(0, height - 1);

    //image[[iv, iu, c]]
    *image.get_unchecked([iv, iu, c])
}
