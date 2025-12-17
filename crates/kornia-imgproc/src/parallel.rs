use rayon::prelude::*;

use kornia_image::{allocator::ImageAllocator, Image};
use kornia_tensor::{CpuAllocator, Tensor2};

/// Apply a function to each pixel in the image in parallel.
///
/// # Arguments
///
/// * `src` - The input image.
/// * `dst` - The output image.
/// * `f` - The function to apply to each pixel.
pub fn par_iter_rows<
    T1,
    const C1: usize,
    A1: ImageAllocator,
    T2,
    const C2: usize,
    A2: ImageAllocator,
>(
    src: &Image<T1, C1, A1>,
    dst: &mut Image<T2, C2, A2>,
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
pub fn par_iter_rows_val<
    T1,
    const C1: usize,
    A1: ImageAllocator,
    T2,
    const C2: usize,
    A2: ImageAllocator,
>(
    src: &Image<T1, C1, A1>,
    dst: &mut Image<T2, C2, A2>,
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
pub fn par_iter_rows_val_two<
    T1,
    const C1: usize,
    A1: ImageAllocator,
    T2,
    const C2: usize,
    A2: ImageAllocator,
    T3,
    const C3: usize,
    A3: ImageAllocator,
>(
    src1: &Image<T1, C1, A1>,
    src2: &Image<T2, C2, A2>,
    dst: &mut Image<T3, C3, A3>,
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
pub fn par_iter_rows_resample<const C: usize, A: ImageAllocator>(
    dst: &mut Image<f32, C, A>,
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

/// Apply a function to each row in the image in parallel, providing the row index.
///
/// # Arguments
///
/// * `dst` - The destination image whose rows will be iterated over.
/// * `f` - A function that receives the row index and a mutable slice corresponding
///   to the row's pixel data.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::parallel::par_iter_rows_indexed_mut;
///
/// let mut image = Image::<f32, 3, _>::from_size_val(
///     ImageSize {
///         width: 4,
///         height: 4,
///     },
///     0.0,
///     CpuAllocator,
/// )
/// .unwrap();
///
/// par_iter_rows_indexed_mut(&mut image, |row_idx, row| {
///     for v in row.iter_mut() {
///         *v += row_idx as f32;
///     }
/// });
/// ```

pub fn par_iter_rows_indexed_mut<T, const C: usize, A: ImageAllocator>(
    dst: &mut Image<T, C, A>,
    f: impl Fn(usize, &mut [T]) + Send + Sync,
) where
    T: Send + Sync,
{
    let cols = dst.cols();
    dst.as_slice_mut()
        .par_chunks_exact_mut(C * cols)
        .enumerate()
        .for_each(|(row_idx, dst_row_slice)| {
            f(row_idx, dst_row_slice);
        });
}
