use kornia_image::{allocator::ImageAllocator, Image};

/// Cubic weight function (Catmull-Rom spline, a = -0.5)
#[inline]
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        (1.5 * t - 2.5) * t * t + 1.0
    } else if t < 2.0 {
        ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    } else {
        0.0
    }
}

/// Clamp index to valid image bounds
#[inline]
fn clamp_index(val: i32, max: i32) -> usize {
    val.max(0).min(max) as usize
}

/// Kernel for bicubic interpolation
///
/// Uses a 4x4 neighborhood with Catmull-Rom cubic weights.
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
pub(crate) fn bicubic_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let rows = image.rows() as i32;
    let cols = image.cols() as i32;

    let iu = u.floor() as i32;
    let iv = v.floor() as i32;

    let frac_u = u - iu as f32;
    let frac_v = v - iv as f32;

    let mut result = 0.0;

    for j in -1..=2_i32 {
        let y = clamp_index(iv + j, rows - 1);
        let wy = cubic_weight(frac_v - j as f32);

        for i in -1..=2_i32 {
            let x = clamp_index(iu + i, cols - 1);
            let wx = cubic_weight(frac_u - i as f32);

            result += *image.get_unchecked([y, x, c]) * wx * wy;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::Image;

    #[test]
    fn test_bicubic_at_integer_coords() {
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let image = Image::<f32, 1, CpuAllocator>::new([4, 4].into(), data, CpuAllocator).unwrap();
        let val = bicubic_interpolation(&image, 1.0, 1.0, 0);
        assert!((val - 6.0).abs() < 1e-5, "expected 6.0, got {val}");
    }

    #[test]
    fn test_bicubic_midpoint() {
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 10.0, 0.0,
            0.0, 10.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let image = Image::<f32, 1, CpuAllocator>::new([4, 4].into(), data, CpuAllocator).unwrap();
        let val = bicubic_interpolation(&image, 1.5, 1.5, 0);
        assert!(val > -2.0 && val < 12.0, "unexpected value: {val}");
    }

    #[test]
    fn test_bicubic_boundary() {
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let image = Image::<f32, 1, CpuAllocator>::new([3, 3].into(), data, CpuAllocator).unwrap();
        let val = bicubic_interpolation(&image, 0.0, 0.0, 0);
        assert!((val - 1.0).abs() < 1e-5, "expected 1.0, got {val}");
        let val = bicubic_interpolation(&image, 2.0, 2.0, 0);
        assert!((val - 9.0).abs() < 1e-5, "expected 9.0, got {val}");
    }

    #[test]
    fn test_bicubic_multichannel() {
        let data: Vec<f32> = vec![
            1.0, 10.0, 100.0,  2.0, 20.0, 200.0,  3.0, 30.0, 300.0,
            4.0, 40.0, 400.0,  5.0, 50.0, 500.0,  6.0, 60.0, 600.0,
            7.0, 70.0, 700.0,  8.0, 80.0, 800.0,  9.0, 90.0, 900.0,
        ];
        let image = Image::<f32, 3, CpuAllocator>::new([3, 3].into(), data, CpuAllocator).unwrap();
        let val_c0 = bicubic_interpolation(&image, 1.0, 1.0, 0);
        let val_c1 = bicubic_interpolation(&image, 1.0, 1.0, 1);
        let val_c2 = bicubic_interpolation(&image, 1.0, 1.0, 2);
        assert!((val_c0 - 5.0).abs() < 1e-5, "ch0: expected 5.0, got {val_c0}");
        assert!((val_c1 - 50.0).abs() < 1e-5, "ch1: expected 50.0, got {val_c1}");
        assert!((val_c2 - 500.0).abs() < 1e-5, "ch2: expected 500.0, got {val_c2}");
    }
}