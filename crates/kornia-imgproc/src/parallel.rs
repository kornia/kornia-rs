use rayon::prelude::*;

use kornia_image::Image;
use kornia_tensor::{CpuAllocator, Tensor2};

/// Apply a function to each pixel in the image in parallel.
///
/// # Arguments
///
/// * `src` - The input image.
/// * `dst` - The output image.
/// * `f` - The function to apply to each pixel.
pub fn par_iter_rows<T1, const C1: usize, T2, const C2: usize>(
    src: &Image<T1, C1>,
    dst: &mut Image<T2, C2>,
    f: impl Fn(&[T1], &mut [T2]) + Send + Sync,
) where
    T1: Clone + Send + Sync,
    T2: Clone + Send + Sync,
{
    src.as_slice()
        .par_chunks_exact(C1 * src.cols())
        .zip(dst.as_slice_mut().par_chunks_exact_mut(C2 * src.cols()))
        .for_each(|(src_chunk, dst_chunk)| {
            src_chunk
                .chunks_exact(C1)
                .zip(dst_chunk.chunks_exact_mut(C2))
                .for_each(|(src_pixel, dst_pixel)| {
                    f(src_pixel, dst_pixel);
                });
        });
}

/// Apply a function to each pixel in the image in parallel with a value.
pub fn par_iter_rows_val<T1, const C1: usize, T2, const C2: usize>(
    src: &Image<T1, C1>,
    dst: &mut Image<T2, C2>,
    f: impl Fn(&T1, &mut T2) + Send + Sync,
) where
    T1: Clone + Send + Sync,
    T2: Clone + Send + Sync,
{
    src.as_slice()
        .par_chunks_exact(C1 * src.cols())
        .zip(dst.as_slice_mut().par_chunks_exact_mut(C2 * src.cols()))
        .for_each(|(src_chunk, dst_chunk)| {
            src_chunk
                .iter()
                .zip(dst_chunk.iter_mut())
                .for_each(|(src_pixel, dst_pixel)| {
                    f(src_pixel, dst_pixel);
                });
        });
}

/// Apply a function to each pixel in the image in parallel with two values.
pub fn par_iter_rows_val_two<T1, const C1: usize, T2, const C2: usize, T3, const C3: usize>(
    src1: &Image<T1, C1>,
    src2: &Image<T2, C2>,
    dst: &mut Image<T3, C3>,
    f: impl Fn(&T1, &T2, &mut T3) + Send + Sync,
) where
    T1: Clone + Send + Sync,
    T2: Clone + Send + Sync,
    T3: Clone + Send + Sync,
{
    src1.as_slice()
        .par_chunks_exact(C1 * src1.cols())
        .zip(src2.as_slice().par_chunks_exact(C2 * src1.cols()))
        .zip(dst.as_slice_mut().par_chunks_exact_mut(C3 * src1.cols()))
        .for_each(|((src1_chunk, src2_chunk), dst_chunk)| {
            src1_chunk
                .iter()
                .zip(src2_chunk.iter())
                .zip(dst_chunk.iter_mut())
                .for_each(|((src1_pixel, src2_pixel), dst_pixel)| {
                    f(src1_pixel, src2_pixel, dst_pixel);
                });
        });
}

/// Apply a function to each pixel for grid sampling in parallel.
pub fn par_iter_rows_resample<const C: usize>(
    dst: &mut Image<f32, C>,
    map_x: &Tensor2<f32, CpuAllocator>,
    map_y: &Tensor2<f32, CpuAllocator>,
    f: impl Fn(&f32, &f32, &mut [f32]) + Send + Sync,
) {
    let cols = dst.cols();
    let dst_slice = dst.as_slice_mut();
    let map_x_slice = map_x.as_slice();
    let map_y_slice = map_y.as_slice();

    dst_slice
        .par_chunks_exact_mut(C * cols)
        .zip(map_x_slice.par_chunks_exact(cols))
        .zip(map_y_slice.par_chunks_exact(cols))
        .for_each(|((dst_chunk, map_x_chunk), map_y_chunk)| {
            dst_chunk
                .chunks_exact_mut(C)
                .zip(map_x_chunk.iter().zip(map_y_chunk.iter()))
                .for_each(|(dst_pixel, (x, y))| {
                    f(x, y, dst_pixel);
                });
        });
}
