use rayon::prelude::*;

use kornia_image::{allocator::ImageAllocator, Image};
use kornia_tensor::{CpuAllocator, Tensor2};

/// Controls how parallel operations are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionStrategy {
    /// Use the global Rayon thread pool.
    ///
    /// This is the default strategy. It is generally efficient for heavy workloads
    /// but may have overhead for very small tasks compared to [`ExecutionStrategy::Serial`].
    #[default]
    Auto,
    /// Run sequentially on the current thread.
    ///
    /// Useful for small images, debugging, or when the overhead of parallelization
    /// outweighs the benefits (e.g., simple thresholding on small/medium images).
    Serial,
    /// Run on a local thread pool with `n` threads.
    ///
    /// # Warning
    /// Creates a new thread pool on every call, which has significant overhead.
    /// Use this primarily for benchmarking or specific isolation needs, not for
    /// tight loops.
    Fixed(usize),
}

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

/// Trait to execute operations on a slice with a given strategy.
pub trait ExecuteExt<T> {
    /// Execute an operation on the slice with the given strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The execution strategy.
    /// * `dst` - The destination slice.
    /// * `op` - The operation to perform on each pixel.
    fn execute_with<F>(self, strategy: ExecutionStrategy, dst: &mut [T], op: F)
    where
        F: Fn((&T, &mut T)) + Sync + Send;
}

impl<T: Sync + Send> ExecuteExt<T> for &[T] {
    fn execute_with<F>(self, strategy: ExecutionStrategy, dst: &mut [T], op: F)
    where
        F: Fn((&T, &mut T)) + Sync + Send,
    {
        match strategy {
            ExecutionStrategy::Serial => {
                self.iter().zip(dst.iter_mut()).for_each(op);
            }
            ExecutionStrategy::Auto => {
                self.par_iter().zip(dst.par_iter_mut()).for_each(op);
            }
            ExecutionStrategy::Fixed(n) => {
                if n == 0 {
                    panic!("ExecutionStrategy::Fixed(n) requires n > 0");
                }
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("Failed to create thread pool with {} threads: {}", n, e);
                    });

                pool.install(|| self.par_iter().zip(dst.par_iter_mut()).for_each(op));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_serial() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice().execute_with(
            ExecutionStrategy::Serial,
            &mut dst,
            |(s, d)| *d = *s * 2,
        );
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_execute_auto() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice().execute_with(
            ExecutionStrategy::Auto,
            &mut dst,
            |(s, d)| *d = *s * 2,
        );
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_execute_fixed() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice().execute_with(
            ExecutionStrategy::Fixed(2),
            &mut dst,
            |(s, d)| *d = *s * 2,
        );
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    #[should_panic(expected = "ExecutionStrategy::Fixed(n) requires n > 0")]
    fn test_execute_fixed_zero() {
        let src = vec![1];
        let mut dst = vec![0];
        src.as_slice().execute_with(
            ExecutionStrategy::Fixed(0),
            &mut dst,
            |(_, _)| {},
        );
    }
}