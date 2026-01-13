use kornia_image::{allocator::ImageAllocator, Image};

/// Catmull-Rom (bicubic) filter
///
/// Cubic convolution kernel with configurable parameter
#[inline]
fn cubic_kernel(mut x: f32, a: f32) -> f32 {
    x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Kernel for bicubic interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
/// * `a` - Cubic parameter (-0.5 for Catmull-Rom).
///
/// # Returns
///
/// The interpolated pixel value.
pub(crate) fn bicubic_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
    a: f32,
) -> f32 {
    let (rows, cols) = (image.rows() as i32, image.cols() as i32);

    let iu = u.trunc() as i32;
    let iv = v.trunc() as i32;

    let frac_u = u.fract();
    let frac_v = v.fract();

    let mut result = 0.0;

    // 4x4 neighborhood from (iu-1, iv-1) to (iu+2, iv+2)
    for dy in -1..=2 {
        for dx in -1..=2 {
            let y = iv + dy;
            let x = iu + dx;

            let y_clamped = y.clamp(0, rows - 1) as usize;
            let x_clamped = x.clamp(0, cols - 1) as usize;

            let kx = cubic_kernel(frac_u - dx as f32, a);
            let ky = cubic_kernel(frac_v - dy as f32, a);
            let pixel = *image.get_unchecked([y_clamped, x_clamped, c]);

            result += pixel * kx * ky;
        }
    }

    result
}
