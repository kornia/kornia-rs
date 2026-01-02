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

struct SeparableFilter<'a> {
    kernel_x: &'a [f32],
    kernel_y: &'a [f32],
    offsets_x: Vec<isize>,
    offsets_y: Vec<isize>,
}

impl<'a> SeparableFilter<'a> {
    fn new(kernel_x: &'a [f32], kernel_y: &'a [f32]) -> Self {
        let half_x = kernel_x.len() / 2;
        let half_y = kernel_y.len() / 2;

        let offsets_x = (0..kernel_x.len())
            .map(|i| i as isize - half_x as isize)
            .collect();

        let offsets_y = (0..kernel_y.len())
            .map(|i| i as isize - half_y as isize)
            .collect();

        Self {
            kernel_x,
            kernel_y,
            offsets_x,
            offsets_y,
        }
    }

    fn apply<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
        &self,
        src: &Image<T, C, A1>,
        dst: &mut Image<T, C, A2>,
    ) -> Result<(), ImageError>
    where
        T: FloatConversion + Clone + Zero + Send + Sync,
    {
        let rows = src.rows();
        let cols = src.cols();

        let src_data = src.as_slice();
        let dst_data = dst.as_slice_mut();
        let mut temp = vec![0.0f32; src_data.len()];

        const TILE_SIZE: usize = 64;

        // Horizontal
        temp.par_chunks_mut(TILE_SIZE * cols * C)
            .enumerate()
            .for_each(|(tile_idx, tile)| {
                let start = tile_idx * TILE_SIZE;
                let end = (start + TILE_SIZE).min(rows);

                for r in start..end {
                    let local_r = r - start;
                    self.apply_kernel_row::<T, C>(src_data, tile, r, local_r, cols);
                }
            });

        // Vertical
        dst_data
            .par_chunks_mut(TILE_SIZE * cols * C)
            .enumerate()
            .for_each(|(tile_idx, tile)| {
                let start = tile_idx * TILE_SIZE;
                let end = (start + TILE_SIZE).min(rows);

                for r in start..end {
                    let local_r = r - start;
                    self.apply_kernel_col::<T, C>(&temp, tile, r, local_r, rows, cols);
                }
            });

        Ok(())
    }

    #[inline]
    fn apply_kernel_row<T: FloatConversion, const C: usize>(
        &self,
        src: &[T],
        dst: &mut [f32],
        r: usize,
        local_r: usize,
        cols: usize,
    ) {
        let global_row = r * cols * C;
        let local_row = local_r * cols * C;

        for c in 0..cols {
            let mut acc = [0.0f32; C];

            for (&k, &off) in self.kernel_x.iter().zip(self.offsets_x.iter()) {
                let x = c as isize + off;
                if x >= 0 && x < cols as isize {
                    let idx = global_row + x as usize * C;
                    for ch in 0..C {
                        acc[ch] += unsafe { src.get_unchecked(idx + ch).to_f32() } * k;
                    }
                }
            }

            let out = local_row + c * C;
            for ch in 0..C {
                unsafe {
                    *dst.get_unchecked_mut(out + ch) = acc[ch];
                }
            }
        }
    }

    #[inline]
    fn apply_kernel_col<T: FloatConversion, const C: usize>(
        &self,
        src: &[f32],
        dst: &mut [T],
        r: usize,
        local_r: usize,
        rows: usize,
        cols: usize,
    ) {
        let local_row = local_r * cols * C;

        for c in 0..cols {
            let mut acc = [0.0f32; C];

            for (&k, &off) in self.kernel_y.iter().zip(self.offsets_y.iter()) {
                let y = r as isize + off;
                if y >= 0 && y < rows as isize {
                    let idx = y as usize * cols * C + c * C;
                    for ch in 0..C {
                        acc[ch] += unsafe { *src.get_unchecked(idx + ch) } * k;
                    }
                }
            }

            let out = local_row + c * C;
            for ch in 0..C {
                unsafe {
                    *dst.get_unchecked_mut(out + ch) = T::from_f32(acc[ch]);
                }
            }
        }
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
pub fn separable_filter<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError>
where
    T: FloatConversion
        + Clone
        + Zero
        + Send
        + Sync
        + std::ops::Mul<Output = T>
        + std::ops::AddAssign,
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
    filter.apply(src, dst)
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
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;

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
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;

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
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;

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
}
