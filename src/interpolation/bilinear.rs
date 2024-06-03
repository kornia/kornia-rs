use crate::image::ImageDtype;
use ndarray::Array3;

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
pub(crate) fn bilinear_interpolation<T: ImageDtype>(
    image: &Array3<T>,
    u: f32,
    v: f32,
    c: usize,
) -> T {
    let (height, width, _) = image.dim();

    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let frac_u = u.fract();
    let frac_v = v.fract();
    let val00: f32 = image[[iv, iu, c]].into();
    let val01: f32 = if iu + 1 < width {
        image[[iv, iu + 1, c]].into()
    } else {
        val00
    };
    let val10: f32 = if iv + 1 < height {
        image[[iv + 1, iu, c]].into()
    } else {
        val00
    };
    let val11: f32 = if iu + 1 < width && iv + 1 < height {
        image[[iv + 1, iu + 1, c]].into()
    } else {
        val00
    };

    let frac_uu = 1. - frac_u;
    let frac_vv = 1. - frac_v;

    T::from_f32(
        val00 * frac_uu * frac_vv
            + val01 * frac_u * frac_vv
            + val10 * frac_uu * frac_v
            + val11 * frac_u * frac_v,
    )
}
