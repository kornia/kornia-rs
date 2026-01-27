use kornia_image::{allocator::ImageAllocator, Image};

/// Kernel for nearest neighbor interpolation
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
pub(crate) fn nearest_neighbor_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
) -> [f32; C] {
    let (rows, cols) = (image.rows(), image.cols());

    let iu = u.round() as usize;
    let iv = v.round() as usize;

    let iu = iu.clamp(0, cols - 1);
    let iv = iv.clamp(0, rows - 1);

    let base = (iv * cols + iu) * C;

    let mut pixel = [0.0; C];
    unsafe {
        let src = image.as_slice().get_unchecked(base..base + C);
        pixel.copy_from_slice(src);
    }

    pixel
}
