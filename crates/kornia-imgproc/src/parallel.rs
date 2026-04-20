use rayon::prelude::*;

use kornia_image::{allocator::ImageAllocator, Image};
use kornia_tensor::{CpuAllocator, Tensor2};

/// Row-group granularity for per-pixel rayon sharding. At 1 row per task,
/// spawn overhead (~2-5 μs on the 8-core Orin pool) rivals per-row work for
/// memory-bound ops like u8→f32 normalize, killing throughput. 16-row chunks
/// put ~68 tasks on an 8-core pool for 1080p (~8 per worker) — enough for
/// work-stealing balance without drowning in task overhead.
const ROWS_PER_TASK: usize = 16;

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
    let src_row_len = C1 * src.cols();
    let dst_row_len = C2 * src.cols();
    src.as_slice()
        .par_chunks(ROWS_PER_TASK * src_row_len)
        .zip(dst.as_slice_mut().par_chunks_mut(ROWS_PER_TASK * dst_row_len))
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
    let src_row_len = C1 * src.cols();
    let dst_row_len = C2 * src.cols();
    src.as_slice()
        .par_chunks(ROWS_PER_TASK * src_row_len)
        .zip(dst.as_slice_mut().par_chunks_mut(ROWS_PER_TASK * dst_row_len))
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
    let cols = src1.cols();
    let s1_row = C1 * cols;
    let s2_row = C2 * cols;
    let d_row = C3 * cols;
    src1.as_slice()
        .par_chunks(ROWS_PER_TASK * s1_row)
        .zip(src2.as_slice().par_chunks(ROWS_PER_TASK * s2_row))
        .zip(dst.as_slice_mut().par_chunks_mut(ROWS_PER_TASK * d_row))
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
        .par_chunks_mut(ROWS_PER_TASK * C * cols)
        .zip(map_x_slice.par_chunks(ROWS_PER_TASK * cols))
        .zip(map_y_slice.par_chunks(ROWS_PER_TASK * cols))
        .for_each(|((dst_chunk, map_x_chunk), map_y_chunk)| {
            dst_chunk
                .chunks_exact_mut(C)
                .zip(map_x_chunk.iter().zip(map_y_chunk.iter()))
                .for_each(|(dst_pixel, (x, y))| f(x, y, dst_pixel))
        });
}

/// Apply a spatial mapping function to each pixel in parallel without pre-allocating coordinate tensors.
pub fn par_iter_rows_spatial_mapping<const C: usize, A: ImageAllocator>(
    dst: &mut Image<f32, C, A>,
    map_coord: impl Fn(usize, usize) -> (f32, f32) + Send + Sync,
    f: impl Fn(f32, f32, &mut [f32]) + Send + Sync,
) {
    let cols = dst.cols();
    let row_len = C * cols;
    let dst_slice = dst.as_slice_mut();

    dst_slice
        .par_chunks_mut(ROWS_PER_TASK * row_len)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let r_base = chunk_idx * ROWS_PER_TASK;
            dst_chunk
                .chunks_exact_mut(row_len)
                .enumerate()
                .for_each(|(dr, row)| {
                    let r = r_base + dr;
                    row.chunks_exact_mut(C).enumerate().for_each(|(c, dst_pixel)| {
                        let (x, y) = map_coord(c, r);
                        f(x, y, dst_pixel)
                    });
                });
        });
}
