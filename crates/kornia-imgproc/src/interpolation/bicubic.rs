use kornia_image::{allocator::ImageAllocator, Image};

pub(crate) fn bicubic_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());

    0.0
}