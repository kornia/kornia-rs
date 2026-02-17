use crate::parallel::ExecutionStrategy;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use num_traits::Zero;
use rayon::prelude::*;

/// Trait for floating point casting
pub trait FloatConversion {
    /// Convert the type to f32
    fn to_f32(&self) -> f32;
    /// Convert the type from f32
    fn from_f32(val: f32) -> Self;
}

impl FloatConversion for f32 {
    fn to_f32(&self) -> f32 {
        *self
    }

    fn from_f32(val: f32) -> Self {
        val
    }
}

impl FloatConversion for f64 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }

    fn from_f32(val: f32) -> Self {
        val as f64
    }
}

impl FloatConversion for u8 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }

    fn from_f32(val: f32) -> Self {
        val.clamp(0.0, 255.0) as u8
    }
}

/// A separable 2D filter that applies horizontal and vertical 1D convolutions sequentially.
///
/// This struct caches the kernel data and precomputed offsets for efficient filtering.
struct SeparableFilter {
    kernel_x: Vec<f32>,
    kernel_y: Vec<f32>,
    offsets_x: Vec<isize>,
    offsets_y: Vec<isize>,
}

/// Macro for generic convolution pass
macro_rules! run_pass {
    (
        $out_row:expr, $src_data:expr, $cols:expr, $C:expr,
        $kernels:expr, $offsets:expr,
        $base_coord:expr, $limit:expr,
        | $curr_c:ident, $offset:ident | $idx_calc:expr,
        | $in_val:ident | $in_conv:expr,
        | $out_val:ident | $out_conv:expr
    ) => {{
        for $curr_c in 0..$cols {
            let mut acc = [0.0f32; $C];
            for (&k, &off) in $kernels.iter().zip($offsets.iter()) {
                let coord = $base_coord as isize + off;
                if coord >= 0 && coord < $limit as isize {
                    let $offset = off;
                    let idx = $idx_calc;
                    for (ch, acc_val) in acc.iter_mut().enumerate().take($C) {
                        let $in_val = unsafe { $src_data.get_unchecked(idx + ch) };
                        *acc_val += $in_conv * k;
                    }
                }
            }
            let out_idx = $curr_c * $C;
            for (ch, &acc_val) in acc.iter().enumerate().take($C) {
                let $out_val = acc_val;
                $out_row[out_idx + ch] = $out_conv;
            }
        }
    }};
}

impl SeparableFilter {
    /// Create a new separable filter with the given kernels.
    ///
    /// # Arguments
    ///
    /// * `kernel_x` - The horizontal convolution kernel
    /// * `kernel_y` - The vertical convolution kernel
    fn new(kernel_x: &[f32], kernel_y: &[f32]) -> Self {
        let half_x = kernel_x.len() / 2;
        let half_y = kernel_y.len() / 2;

        let offsets_x = (0..kernel_x.len())
            .map(|i| i as isize - half_x as isize)
            .collect();

        let offsets_y = (0..kernel_y.len())
            .map(|i| i as isize - half_y as isize)
            .collect();

        Self {
            kernel_x: kernel_x.to_vec(),
            kernel_y: kernel_y.to_vec(),
            offsets_x,
            offsets_y,
        }
    }

    /// Apply the filter serially (single-threaded).
    ///
    /// This version does not require `Send + Sync` bounds, making it suitable for
    /// non-thread-safe types.
    ///
    /// # Arguments
    ///
    /// * `src` - The source image
    /// * `dst` - The destination image (must be same size as source)
    fn apply_serial<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
        &self,
        src: &Image<T, C, A1>,
        dst: &mut Image<T, C, A2>,
    ) -> Result<(), ImageError>
    where
        T: FloatConversion + Clone + Zero,
    {
        let rows = src.rows();
        let cols = src.cols();
        let src_data = src.as_slice();
        let dst_data = dst.as_slice_mut();
        let mut temp = vec![0.0f32; src_data.len()];

        // Horizontal pass
        for r in 0..rows {
            let row_offset = r * cols * C;
            let out_row = &mut temp[row_offset..row_offset + cols * C];
            run_pass!(
                out_row,
                src_data,
                cols,
                C,
                self.kernel_x,
                self.offsets_x,
                c,
                cols,
                |c, off| (r * cols * C) + (c as isize + off) as usize * C,
                |val| val.to_f32(),
                |acc| acc
            );
        }

        // Vertical pass
        for r in 0..rows {
            let row_offset = r * cols * C;
            let out_row = &mut dst_data[row_offset..row_offset + cols * C];
            run_pass!(
                out_row,
                temp,
                cols,
                C,
                self.kernel_y,
                self.offsets_y,
                r,
                rows,
                |c, off| (r as isize + off) as usize * cols * C + c * C,
                |val| *val,
                |acc| <T>::from_f32(acc)
            );
        }

        Ok(())
    }

    /// Apply the filter to an image.
    ///
    /// Performs horizontal filtering followed by vertical filtering using a temporary buffer.
    ///
    /// # Arguments
    ///
    /// * `src` - The source image
    /// * `dst` - The destination image (must be same size as source)
    /// * `strategy` - The execution strategy to use
    fn apply<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
        &self,
        src: &Image<T, C, A1>,
        dst: &mut Image<T, C, A2>,
        strategy: ExecutionStrategy,
    ) -> Result<(), ImageError>
    where
        T: FloatConversion + Clone + Zero + Send + Sync,
    {
        if src.size() != dst.size() {
            return Err(ImageError::InvalidImageSize(
                src.cols(),
                src.rows(),
                dst.cols(),
                dst.rows(),
            ));
        }

        if src.cols() == 0 || src.rows() == 0 {
            return Ok(());
        }

        let rows = src.rows();
        let cols = src.cols();
        let src_data = src.as_slice();
        let dst_data = dst.as_slice_mut();
        let mut temp = vec![0.0f32; src_data.len()];

        match strategy {
            ExecutionStrategy::Serial => {
                return self.apply_serial(src, dst);
            }
            ExecutionStrategy::Fixed(n) => {
                if n == 0 {
                    return Err(ImageError::Parallel("thread count must be > 0".to_string()));
                }
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| ImageError::Parallel(e.to_string()))?;

                pool.install(|| {
                    // Horizontal
                    temp.par_chunks_mut(cols * C)
                        .enumerate()
                        .for_each(|(r, row)| {
                            run_pass!(
                                row,
                                src_data,
                                cols,
                                C,
                                self.kernel_x,
                                self.offsets_x,
                                c,
                                cols,
                                |c, off| (r * cols * C) + (c as isize + off) as usize * C,
                                |val| val.to_f32(),
                                |acc| acc
                            );
                        });

                    // Vertical
                    dst_data
                        .par_chunks_mut(cols * C)
                        .enumerate()
                        .for_each(|(r, row)| {
                            run_pass!(
                                row,
                                temp,
                                cols,
                                C,
                                self.kernel_y,
                                self.offsets_y,
                                r,
                                rows,
                                |c, off| (r as isize + off) as usize * cols * C + c * C,
                                |val| *val,
                                |acc| <T>::from_f32(acc)
                            );
                        });
                });
            }

            ExecutionStrategy::AutoRows(stride) => {
                if stride == 0 {
                    return Err(ImageError::Parallel("row stride must be > 0".to_string()));
                }

                // Horizontal
                temp.par_chunks_mut(stride * cols * C).enumerate().for_each(
                    |(chunk_idx, chunk)| {
                        chunk.chunks_exact_mut(cols * C).enumerate().for_each(
                            |(r_in_chunk, row)| {
                                let r = chunk_idx * stride + r_in_chunk;
                                run_pass!(
                                    row,
                                    src_data,
                                    cols,
                                    C,
                                    self.kernel_x,
                                    self.offsets_x,
                                    c,
                                    cols,
                                    |c, off| (r * cols * C) + (c as isize + off) as usize * C,
                                    |val| val.to_f32(),
                                    |acc| acc
                                );
                            },
                        );
                    },
                );

                // Vertical
                dst_data
                    .par_chunks_mut(stride * cols * C)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        chunk.chunks_exact_mut(cols * C).enumerate().for_each(
                            |(r_in_chunk, row)| {
                                let r = chunk_idx * stride + r_in_chunk;
                                run_pass!(
                                    row,
                                    temp,
                                    cols,
                                    C,
                                    self.kernel_y,
                                    self.offsets_y,
                                    r,
                                    rows,
                                    |c, off| (r as isize + off) as usize * cols * C + c * C,
                                    |val| *val,
                                    |acc| <T>::from_f32(acc)
                                );
                            },
                        );
                    });
            }

            ExecutionStrategy::ParallelElements => {
                temp.par_iter_mut().enumerate().for_each(|(i, temp_val)| {
                    let r = i / (cols * C);
                    let c = (i % (cols * C)) / C;
                    let ch = i % C;

                    let row_offset = r * cols * C;
                    let mut acc = 0.0f32;
                    for (&k, &off) in self.kernel_x.iter().zip(self.offsets_x.iter()) {
                        let x = c as isize + off;
                        if x >= 0 && x < cols as isize {
                            let idx = row_offset + x as usize * C + ch;
                            acc += unsafe { src_data.get_unchecked(idx).to_f32() } * k;
                        }
                    }
                    *temp_val = acc;
                });

                dst_data
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, dst_val)| {
                        let r = i / (cols * C);
                        let c = (i % (cols * C)) / C;
                        let ch = i % C;

                        let mut acc = 0.0f32;
                        for (&k, &off) in self.kernel_y.iter().zip(self.offsets_y.iter()) {
                            let y = r as isize + off;
                            if y >= 0 && y < rows as isize {
                                let idx = y as usize * cols * C + c * C + ch;
                                acc += unsafe { *temp.get_unchecked(idx) } * k;
                            }
                        }
                        *dst_val = T::from_f32(acc);
                    });
            }
        }

        Ok(())
    }
}

/// Apply a separable filter to an image.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_x` - The horizontal kernel.
/// * `kernel_y` - The vertical kernel.
/// * `strategy` - The execution strategy.
pub fn separable_filter<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel_x: &[f32],
    kernel_y: &[f32],
    strategy: ExecutionStrategy,
) -> Result<(), ImageError>
where
    T: FloatConversion + Clone + Zero + Send + Sync,
{
    if kernel_x.is_empty() || kernel_y.is_empty() {
        return Err(ImageError::InvalidKernelLength(
            kernel_x.len(),
            kernel_y.len(),
        ));
    }

    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let filter = SeparableFilter::new(kernel_x, kernel_y);
    filter.apply(src, dst, strategy)
}

/// Apply a 1D separable filter (serial execution only, no parallelism).
///
/// This version does not require `Send + Sync` bounds on the pixel type,
///
/// # Arguments
///
/// * `src` - Source image
/// * `dst` - Destination image (must have same size as source)
/// * `kernel_x` - Horizontal filter kernel
/// * `kernel_y` - Vertical filter kernel
pub fn separable_filter_serial<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError>
where
    T: FloatConversion + Clone + Zero,
{
    if kernel_x.is_empty() || kernel_y.is_empty() {
        return Err(ImageError::InvalidKernelLength(
            kernel_x.len(),
            kernel_y.len(),
        ));
    }

    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let filter = SeparableFilter::new(kernel_x, kernel_y);
    filter.apply_serial(src, dst)
}

/// Apply a fast filter horizontally using cumulative kernel
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with transposed shape (W, H, C).
/// * `half_kernel_x_size` - Half of the kernel at weight 1. The total size would be 2*this+1
pub(crate) fn fast_horizontal_filter<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    half_kernel_x_size: usize,
) -> Result<(), ImageError> {
    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();
    let mut row_acc = [0.0; C];

    let mut leftmost_pixel = [0.0; C];
    let mut rightmost_pixel = [0.0; C];

    let pixels_between_first_last_cols = (src.cols() - 1) * C;
    let kernel_pix_offset_diffs: Vec<usize> =
        (0..half_kernel_x_size).map(|p| (p + 1) * C).collect();
    for (pix_offset, source_pixel) in src_data.iter().enumerate() {
        let ch = pix_offset % C;
        let rc = pix_offset / C;
        let c = rc % src.cols();
        let r = rc / src.cols();

        let transposed_r = c;
        let transposed_c = r;
        let transposed_pix_offset = transposed_r * src.rows() * C + transposed_c * C + ch;

        if c == 0 {
            row_acc[ch] = *source_pixel * (half_kernel_x_size + 1) as f32;
            for pix_diff in &kernel_pix_offset_diffs {
                row_acc[ch] += src_data[pix_offset + pix_diff]
            }
            leftmost_pixel[ch] = *source_pixel;
            rightmost_pixel[ch] = src_data[pix_offset + pixels_between_first_last_cols];
        } else {
            row_acc[ch] -= match c.checked_sub(half_kernel_x_size + 1) {
                Some(_) => {
                    let prv_leftmost_pix_offset = pix_offset - C * (half_kernel_x_size + 1);
                    src_data[prv_leftmost_pix_offset]
                }
                None => leftmost_pixel[ch],
            };

            let rightmost_x = c + half_kernel_x_size;

            row_acc[ch] += match rightmost_x {
                x if x < src.cols() => {
                    let rightmost_pix_offset = pix_offset + C * half_kernel_x_size;
                    src_data[rightmost_pix_offset]
                }
                _ => rightmost_pixel[ch],
            };
        }
        dst_data[transposed_pix_offset] = row_acc[ch] / (half_kernel_x_size * 2 + 1) as f32;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_separable_filter_f32() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut dst = Image::<_, 1, _>::from_size_val(img.size(), 0f32, CpuAllocator)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let xsum = dst.as_slice().iter().sum::<f32>();
        assert_eq!(xsum, 9.0);

        Ok(())
    }

    #[test]
    fn test_separable_filter_u8() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 255, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            CpuAllocator
        )?;

        let mut dst = Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0, 0, 0, 0, 0,
                0, 255, 255, 255, 0,
                0, 255, 255, 255, 0,
                0, 255, 255, 255, 0,
                0, 0, 0, 0, 0,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_separable_filter_u8_max_val() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];

        let mut img = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        img.as_slice_mut()[12] = 255;

        let mut dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[0, 0, 0, 0, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 0, 0, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn test_fast_horizontal_filter() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 9.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut transposed = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        fast_horizontal_filter(&img, &mut transposed, 1)?;

        #[rustfmt::skip]
        assert_eq!(
            transposed.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let mut dst = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        fast_horizontal_filter(&transposed, &mut dst, 1)?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
        let xsum = dst.as_slice().iter().sum::<f32>();
        assert_eq!(xsum, 9.0);

        Ok(())
    }

    #[test]
    fn test_parallel_strategies_consistency() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 10,
            height: 10,
        };

        // Create test image with pattern
        let mut data = vec![0.0f32; 100];
        data[44] = 1.0; // Center pixel
        data[33] = 0.5; // Another pixel
        data[67] = 0.8;

        let img = Image::new(size, data, CpuAllocator)?;
        let kernel_x = vec![0.25, 0.5, 0.25];
        let kernel_y = vec![0.25, 0.5, 0.25];

        // Serial (reference)
        let mut dst_serial = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_serial,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        // Fixed(4)
        let mut dst_fixed = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_fixed,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Fixed(4),
        )?;

        // AutoRows
        let mut dst_auto = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_auto,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::AutoRows(2),
        )?;

        // ParallelElements
        let mut dst_elements = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_elements,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::ParallelElements,
        )?;

        // All strategies should produce identical results
        assert_eq!(
            dst_serial.as_slice(),
            dst_fixed.as_slice(),
            "Fixed strategy mismatch"
        );
        assert_eq!(
            dst_serial.as_slice(),
            dst_auto.as_slice(),
            "AutoRows strategy mismatch"
        );
        assert_eq!(
            dst_serial.as_slice(),
            dst_elements.as_slice(),
            "ParallelElements strategy mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_parallel_strategies_u8() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 8,
            height: 8,
        };

        let mut data = vec![0u8; 64];
        data[27] = 255;
        data[36] = 128;

        let img = Image::new(size, data, CpuAllocator)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];

        // Test strategies
        let mut dst_serial = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_serial,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        let mut dst_fixed = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_fixed,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Fixed(2),
        )?;

        let mut dst_auto = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_auto,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::AutoRows(2),
        )?;

        let mut dst_elements = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_elements,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::ParallelElements,
        )?;

        assert_eq!(dst_serial.as_slice(), dst_fixed.as_slice());
        assert_eq!(dst_serial.as_slice(), dst_auto.as_slice());
        assert_eq!(dst_serial.as_slice(), dst_elements.as_slice());

        Ok(())
    }

    #[test]
    fn test_fixed_threadpool_validation() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let img = Image::<f32, 1, _>::from_size_val(size, 0.5, CpuAllocator)?;
        let mut dst = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        let kernel = vec![1.0];

        // Fixed(0) should error
        let result = separable_filter(
            &img,
            &mut dst,
            &kernel,
            &kernel,
            ExecutionStrategy::Fixed(0),
        );

        match result {
            Err(ImageError::Parallel(msg)) => {
                assert!(
                    msg.contains("thread count must be > 0"),
                    "unexpected error message: {msg}"
                );
            }
            Err(e) => panic!("unexpected error type: {e:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }

        Ok(())
    }
}
