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
    let (height, width) = (image.height(), image.width());

    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let frac_u = u.fract();
    let frac_v = v.fract();
    //let val00: f32 = image[[iv, iu, c]].into();
    let val00 = *image.get_unchecked([iv, iu, c]);
    let val01 = if iu + 1 < width {
        //image[[iv, iu + 1, c]].into()
        *image.get_unchecked([iv, iu + 1, c])
    } else {
        val00
    };
    let val10 = if iv + 1 < height {
        //image[[iv + 1, iu, c]].into()
        *image.get_unchecked([iv + 1, iu, c])
    } else {
        val00
    };
    let val11 = if iu + 1 < width && iv + 1 < height {
        //image[[iv + 1, iu + 1, c]].into()
        *image.get_unchecked([iv + 1, iu + 1, c])
    } else {
        val00
    };

    let frac_uu = 1. - frac_u;
    let frac_vv = 1. - frac_v;

    val00 * frac_uu * frac_vv
        + val01 * frac_u * frac_vv
        + val10 * frac_uu * frac_v
        + val11 * frac_u * frac_v
}
