use kornia_image::{allocator::ImageAllocator, Image};

/// Kernel for bilinear interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel values.
// TODO: add support for other data types. Maybe use a trait? or template?
pub(crate) fn bilinear_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
) -> [f32; C] {
    let (rows, cols) = (image.rows(), image.cols());

    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let iu0 = iu.min(cols - 1);
    let iv0 = iv.min(rows - 1);

    let frac_u = u.fract();
    let frac_v = v.fract();

    let frac_uu = 1.0 - frac_u;
    let frac_vv = 1.0 - frac_v;

    let w00 = frac_uu * frac_vv;
    let w01 = frac_u * frac_vv;
    let w10 = frac_uu * frac_v;
    let w11 = frac_u * frac_v;

    let iu1 = if iu0 + 1 < cols { iu0 + 1 } else { iu0 };
    let iv1 = if iv0 + 1 < rows { iv0 + 1 } else { iv0 };

    let base00 = (iv0 * cols + iu0) * C;
    let base01 = (iv0 * cols + iu1) * C;
    let base10 = (iv1 * cols + iu0) * C;

    let base11 = (iv1 * cols + iu1) * C;

    let data = image.as_slice();

    let p00 = unsafe { data.get_unchecked(base00..base00 + C) };
    let p01 = unsafe { data.get_unchecked(base01..base01 + C) };
    let p10 = unsafe { data.get_unchecked(base10..base10 + C) };
    let p11 = unsafe { data.get_unchecked(base11..base11 + C) };

    let mut pixel = [0.0; C];
    for k in 0..C {
        pixel[k] = p00[k] * w00 + p01[k] * w01 + p10[k] * w10 + p11[k] * w11;
    }

    pixel
}
