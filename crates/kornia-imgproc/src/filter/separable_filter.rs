use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use num_traits::Zero;
use wide::f32x4;

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

    /// Apply the filter to an image.
    ///
    /// Performs horizontal filtering followed by vertical filtering using a temporary buffer.
    ///
    /// # Arguments
    ///
    /// * `src` - The source image
    /// * `dst` - The destination image (must be same size as source)
    fn apply<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
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

        // Horizontal
        for r in 0..rows {
            let row_offset = r * cols * C;

            for c in 0..cols {
                let mut acc = [0.0f32; C];

                for (&k, &off) in self.kernel_x.iter().zip(self.offsets_x.iter()) {
                    let x = c as isize + off;
                    if x >= 0 && x < cols as isize {
                        let idx = row_offset + x as usize * C;
                        for (ch, acc_val) in acc.iter_mut().enumerate().take(C) {
                            *acc_val += unsafe { src_data.get_unchecked(idx + ch).to_f32() } * k;
                        }
                    }
                }

                let out_idx = row_offset + c * C;
                for (ch, &acc_val) in acc.iter().enumerate().take(C) {
                    unsafe {
                        *temp.get_unchecked_mut(out_idx + ch) = acc_val;
                    }
                }
            }
        }

        // Vertical
        for r in 0..rows {
            let row_offset = r * cols * C;

            for c in 0..cols {
                let mut acc = [0.0f32; C];

                for (&k, &off) in self.kernel_y.iter().zip(self.offsets_y.iter()) {
                    let y = r as isize + off;
                    if y >= 0 && y < rows as isize {
                        let idx = y as usize * cols * C + c * C;
                        for (ch, acc_val) in acc.iter_mut().enumerate().take(C) {
                            *acc_val += unsafe { *temp.get_unchecked(idx + ch) } * k;
                        }
                    }
                }

                let out_idx = row_offset + c * C;
                for (ch, &acc_val) in acc.iter().enumerate().take(C) {
                    unsafe {
                        *dst_data.get_unchecked_mut(out_idx + ch) = T::from_f32(acc_val);
                    }
                }
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
pub fn separable_filter<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
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

#[inline(always)]
unsafe fn load4(p: *const f32) -> f32x4 {
    // SAFETY: caller ensures at least 4 elements remain at p.
    f32x4::new([*p, *p.add(1), *p.add(2), *p.add(3)])
}

#[inline(always)]
unsafe fn store4(p: *mut f32, v: f32x4) {
    // SAFETY: caller ensures at least 4 writable elements remain at p.
    let a = v.to_array();
    *p = a[0];
    *p.add(1) = a[1];
    *p.add(2) = a[2];
    *p.add(3) = a[3];
}

#[inline]
fn update_row<const C: usize>(
    row_idx: usize,
    add: bool,
    cols: usize,
    src_data: &[f32],
    col_sums: &mut [f32],
) {
    debug_assert!(
        row_idx * cols * C + cols * C <= src_data.len(),
        "update_row: src_data too short for row_idx={row_idx}, cols={cols}, C={C}"
    );
    debug_assert_eq!(
        col_sums.len(),
        cols * C,
        "update_row: col_sums length mismatch"
    );

    unsafe {
        if C == 1 {
            let base = row_idx * cols;
            let mut c = 0usize;

            while c + 4 <= cols {
                // SAFETY: while condition guarantees 4 elements remain; bounds checked by debug_assert.
                let sv = load4(src_data.as_ptr().add(base + c));
                let cv = load4(col_sums.as_ptr().add(c));
                let r = if add { cv + sv } else { cv - sv };
                store4(col_sums.as_mut_ptr().add(c), r);
                c += 4;
            }

            while c < cols {
                // SAFETY: c < cols keeps both accesses in bounds; checked by debug_assert.
                let v = *src_data.get_unchecked(base + c);
                if add {
                    *col_sums.get_unchecked_mut(c) += v;
                } else {
                    *col_sums.get_unchecked_mut(c) -= v;
                }
                c += 1;
            }
        } else if C == 4 {
            let base = row_idx * cols * 4;
            for c in 0..cols {
                // SAFETY: c < cols keeps 4-wide src and col_sums reads in bounds; checked by debug_assert.
                let src_p = src_data.as_ptr().add(base + c * 4);
                let col_p = col_sums.as_mut_ptr().add(c * 4);
                let sv = load4(src_p);
                let cv = load4(col_p);
                let r = if add { cv + sv } else { cv - sv };
                store4(col_p, r);
            }
        } else {
            for c in 0..cols {
                let idx = (row_idx * cols + c) * C;
                for ch in 0..C {
                    // SAFETY: c < cols and ch < C keep both accesses in bounds; checked by debug_assert.
                    let v = *src_data.get_unchecked(idx + ch);
                    let out = col_sums.get_unchecked_mut(c * C + ch);
                    if add {
                        *out += v;
                    } else {
                        *out -= v;
                    }
                }
            }
        }
    }
}

/// Apply box blur using incremental column sums.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_size` - The size of the kernel (kernel_x, kernel_y).
pub fn columnar_sat<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    kernel_size: (usize, usize),
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidKernelLength(
            kernel_size.0,
            kernel_size.1,
        ));
    }
    if kernel_size.0 == 0
        || kernel_size.1 == 0
        || kernel_size.0 > src.cols()
        || kernel_size.1 > src.rows()
    {
        return Err(ImageError::InvalidKernelLength(
            kernel_size.0,
            kernel_size.1,
        ));
    }

    let cols = src.cols();
    let rows = src.rows();

    let half_x = kernel_size.0 / 2;
    let half_y = kernel_size.1 / 2;

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    let mut col_sums = vec![0.0f32; cols * C];
    let mut row_acc = [0.0f32; C];

    for r in 0..rows {
        let y_start = r.saturating_sub(half_y);
        let y_end = (r + half_y + 1).min(rows);
        let row_offset = r * cols * C;

        // vertical incremental update
        if r == 0 {
            for y in y_start..y_end {
                update_row::<C>(y, true, cols, src_data, &mut col_sums);
            }
        } else {
            if r > half_y {
                update_row::<C>(r - half_y - 1, false, cols, src_data, &mut col_sums);
            }
            if r + half_y < rows {
                update_row::<C>(r + half_y, true, cols, src_data, &mut col_sums);
            }
        }

        // horizontal sliding window
        debug_assert_eq!(col_sums.len(), cols * C, "col_sums length mismatch");
        debug_assert!(
            row_offset + cols * C <= dst_data.len(),
            "dst_data too short: row_offset={row_offset}, cols={cols}, C={C}"
        );
        for c in 0..cols {
            let x_start = c.saturating_sub(half_x);
            let x_end = (c + half_x + 1).min(cols);

            if c == 0 {
                row_acc.fill(0.0);
                for x in x_start..x_end {
                    let base = x * C;
                    for (ch, acc) in row_acc.iter_mut().enumerate() {
                        // SAFETY: x is clamped to cols so base + ch stays within col_sums; checked by debug_assert.
                        *acc += unsafe { *col_sums.get_unchecked(base + ch) };
                    }
                }
            } else {
                let prev_x_start = (c - 1).saturating_sub(half_x);
                if x_start > prev_x_start {
                    let base = prev_x_start * C;
                    for (ch, acc) in row_acc.iter_mut().enumerate() {
                        // SAFETY: prev_x_start < cols so base + ch stays within col_sums; checked by debug_assert.
                        *acc -= unsafe { *col_sums.get_unchecked(base + ch) };
                    }
                }

                let prev_x_end = (c + half_x).min(cols);
                if x_end > prev_x_end {
                    let base = (x_end - 1) * C;
                    for (ch, acc) in row_acc.iter_mut().enumerate() {
                        // SAFETY: x_end clamped to cols makes x_end - 1 a valid index; checked by debug_assert.
                        *acc += unsafe { *col_sums.get_unchecked(base + ch) };
                    }
                }
            }

            let inv_area = 1.0 / ((x_end - x_start) * (y_end - y_start)) as f32;
            let out_idx = row_offset + c * C;

            if C == 4 {
                let v = f32x4::new([row_acc[0], row_acc[1], row_acc[2], row_acc[3]])
                    * f32x4::splat(inv_area);
                // SAFETY: r < rows and c < cols guarantee 4 elements remain at out_idx; checked by debug_assert.
                unsafe {
                    store4(dst_data.as_mut_ptr().add(out_idx), v);
                }
            } else {
                for (ch, &acc) in row_acc.iter().enumerate() {
                    // SAFETY: r < rows and c < cols keep out_idx + ch within dst_data; checked by debug_assert.
                    unsafe {
                        *dst_data.get_unchecked_mut(out_idx + ch) = acc * inv_area;
                    }
                }
            }
        }
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

    #[test]
    fn test_columnar_sat_box_blur() -> Result<(), ImageError> {
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
                0.0, 0.0, 255.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            CpuAllocator,
        )?;

        let mut dst = Image::<f32, 1, _>::from_size_val(img.size(), 0.0, CpuAllocator)?;

        columnar_sat(&img, &mut dst, (3, 3))?;

        let center_val = 255.0 / 9.0;

        assert!((dst.as_slice()[6] - center_val).abs() < 1e-3);
        assert!((dst.as_slice()[7] - center_val).abs() < 1e-3);
        assert!((dst.as_slice()[12] - center_val).abs() < 1e-3);
        assert_eq!(dst.as_slice()[0], 0.0);

        Ok(())
    }
}
