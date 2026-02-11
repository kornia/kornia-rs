use rayon::prelude::*;
use thiserror::Error;

use kornia_image::{allocator::ImageAllocator, Image};
use kornia_tensor::{CpuAllocator, Tensor2};

/// Errors that can occur during parallel execution.
#[derive(Error, Debug, PartialEq)]
pub enum ParallelError {
    /// The thread pool failed to build.
    #[error("failed to build thread pool: {0}")]
    BuildError(String),

    /// The requested thread count is invalid.
    #[error("thread count must be > 0, got {0}")]
    InvalidThreadCount(usize),

    /// The row stride for AutoRows must be valid.
    #[error("row stride must be > 0 for AutoRows strategy")]
    InvalidRowStride(usize),

    /// Input and output sizes do not match.
    #[error("source and destination slices must have the same length")]
    SizeMismatch,
}

/// Controls how parallel operations are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionStrategy {
    /// Use the global Rayon thread pool to process every element in parallel.
    ///
    /// This maximizes parallelism but may have overhead for small operations.
    #[default]
    ParallelElements,

    /// Use the global Rayon thread pool to process rows (chunks) in parallel.
    ///
    /// You must provide the row stride (width * channels).
    /// This is often more cache-friendly than [`ExecutionStrategy::ParallelElements`].
    AutoRows(usize),

    /// Run sequentially on the current thread.
    ///
    /// Useful for small images, debugging, or when the overhead of parallelization
    /// outweighs the benefits.
    Serial,

    /// Run on a local thread pool with `n` threads.
    ///
    /// # Warning
    /// Creates a new thread pool on every call, which has significant overhead.
    /// Use this primarily for benchmarking or specific isolation needs.
    Fixed(usize),
}

/// Apply a function to each pixel in the image in parallel.
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
    /// * `op` - The operation to perform on each (source, destination) element pair.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
    fn execute_with<F>(
        &self,
        strategy: ExecutionStrategy,
        dst: &mut [T],
        op: F,
    ) -> Result<(), ParallelError>
    where
        F: Fn((&T, &mut T)) + Sync + Send;
}

impl<T: Sync + Send> ExecuteExt<T> for &[T] {
    fn execute_with<F>(
        &self,
        strategy: ExecutionStrategy,
        dst: &mut [T],
        op: F,
    ) -> Result<(), ParallelError>
    where
        F: Fn((&T, &mut T)) + Sync + Send,
    {
        if self.len() != dst.len() {
            return Err(ParallelError::SizeMismatch);
        }

        match strategy {
            ExecutionStrategy::Serial => {
                self.iter().zip(dst.iter_mut()).for_each(op);
            }
            ExecutionStrategy::ParallelElements => {
                self.par_iter().zip(dst.par_iter_mut()).for_each(op);
            }
            ExecutionStrategy::AutoRows(stride) => {
                if stride == 0 {
                    return Err(ParallelError::InvalidRowStride(stride));
                }
                self.par_chunks(stride)
                    .zip(dst.par_chunks_mut(stride))
                    .for_each(|(src_row, dst_row)| {
                        src_row.iter().zip(dst_row.iter_mut()).for_each(&op);
                    });
            }
            ExecutionStrategy::Fixed(n) => {
                if n == 0 {
                    return Err(ParallelError::InvalidThreadCount(n));
                }
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| ParallelError::BuildError(e.to_string()))?;

                pool.install(|| {
                    self.par_iter().zip(dst.par_iter_mut()).for_each(op);
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_serial() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice()
            .execute_with(ExecutionStrategy::Serial, &mut dst, |(s, d)| *d = *s * 2)
            .unwrap();
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_execute_parallel_elements() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice()
            .execute_with(ExecutionStrategy::ParallelElements, &mut dst, |(s, d)| {
                *d = *s * 2
            })
            .unwrap();
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_execute_auto_rows() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice()
            .execute_with(ExecutionStrategy::AutoRows(2), &mut dst, |(s, d)| {
                *d = *s * 2
            })
            .unwrap();
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_execute_auto_rows_invalid() {
        let src = vec![1];
        let mut dst = vec![0];
        let res =
            src.as_slice()
                .execute_with(ExecutionStrategy::AutoRows(0), &mut dst, |(_, _)| {});
        assert!(matches!(res, Err(ParallelError::InvalidRowStride(0))));
    }

    #[test]
    fn test_execute_fixed_success() {
        let src = vec![1, 2, 3, 4];
        let mut dst = vec![0; 4];
        src.as_slice()
            .execute_with(ExecutionStrategy::Fixed(2), &mut dst, |(s, d)| *d = *s * 2)
            .unwrap();
        assert_eq!(dst, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_execute_fixed_error() {
        let src = vec![1];
        let mut dst = vec![0];
        let res = src
            .as_slice()
            .execute_with(ExecutionStrategy::Fixed(0), &mut dst, |(_, _)| {});
        assert!(matches!(res, Err(ParallelError::InvalidThreadCount(0))));
    }
}
