use kornia_image::{Image, ImageError};
//use num_traits::Zero;

/// Trait for element-wise operations.
pub trait ElementOp {
    /// The type of the element.
    type ElementType: Clone;
    /// Return the zero element.
    fn zero() -> Self::ElementType;
    /// Multiply two elements.
    fn mul(a: Self, b: Self) -> Self::ElementType;
    /// Multiply an element with another element.
    fn mul_elem(a: Self::ElementType, b: Self) -> Self::ElementType;
    /// Add an element to another element in place.
    fn add_assign(a: &mut Self::ElementType, b: Self::ElementType);
    /// Convert an element to another type.
    fn from_elem(a: Self::ElementType) -> Self;
}

impl ElementOp for f32 {
    type ElementType = f32;
    fn mul(a: Self, b: Self) -> Self::ElementType {
        a * b
    }
    fn mul_elem(a: Self::ElementType, b: Self) -> Self::ElementType {
        a * b
    }
    fn add_assign(a: &mut Self::ElementType, b: Self::ElementType) {
        *a += b;
    }
    fn zero() -> Self::ElementType {
        0.0
    }
    fn from_elem(a: Self::ElementType) -> Self {
        a
    }
}

impl ElementOp for u8 {
    type ElementType = u32;
    fn mul(a: Self, b: Self) -> Self::ElementType {
        a as u32 * b as u32
    }
    fn mul_elem(a: Self::ElementType, b: Self) -> Self::ElementType {
        a * b as u32
    }
    fn add_assign(a: &mut Self::ElementType, b: Self::ElementType) {
        *a += b;
    }
    fn zero() -> Self::ElementType {
        0
    }
    fn from_elem(a: Self::ElementType) -> Self {
        a as u8
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
pub fn separable_filter<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel_x: &[T],
    kernel_y: &[T],
) -> Result<(), ImageError>
where
    //T: Zero + std::ops::Mul<Output = T> + std::ops::AddAssign + Copy,
    T: ElementOp + Copy + Clone,
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

    let half_kernel_x = kernel_x.len() / 2;
    let half_kernel_y = kernel_y.len() / 2;

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    // preallocate the temporary buffer for intermediate results
    // TODO: use a better buffer allocation strategy
    let mut temp = vec![T::zero(); src_data.len()];

    // Row-wise filtering
    for r in 0..src.rows() {
        let row_offset = r * src.cols();
        for c in 0..src.cols() {
            let col_offset = (row_offset + c) * C;
            for ch in 0..C {
                let pix_offset = col_offset + ch;
                let mut row_acc = T::zero();
                for (k_idx, k_val) in kernel_x.iter().enumerate() {
                    let x_pos = c as isize + k_idx as isize - half_kernel_x as isize;
                    if x_pos >= 0 && x_pos < src.cols() as isize {
                        let neighbor_idx = (row_offset + x_pos as usize) * C + ch;
                        let neighbor_val = unsafe { src_data.get_unchecked(neighbor_idx) };
                        //row_acc += unsafe { *src_data.get_unchecked(neighbor_idx) } * *k_val;
                        T::add_assign(&mut row_acc, T::mul(*neighbor_val, *k_val));
                    }
                }

                unsafe {
                    *temp.get_unchecked_mut(pix_offset) = row_acc;
                }
            }
        }
    }

    // Column-wise filtering
    for r in 0..src.rows() {
        let row_offset = r * src.cols();
        for c in 0..src.cols() {
            let col_offset = (row_offset + c) * C;
            for ch in 0..C {
                let pix_offset = col_offset + ch;
                let mut col_acc = T::zero();
                for (k_idx, k_val) in kernel_y.iter().enumerate() {
                    let y_pos = r as isize + k_idx as isize - half_kernel_y as isize;
                    if y_pos >= 0 && y_pos < src.rows() as isize {
                        let neighbor_idx = (y_pos as usize * src.cols() + c) * C + ch;
                        //col_acc += unsafe { *temp.get_unchecked(neighbor_idx) } * *k_val;
                        let neighbor_val = unsafe { temp.get_unchecked(neighbor_idx) };
                        T::add_assign(&mut col_acc, T::mul_elem(neighbor_val.clone(), *k_val));
                    }
                }
                unsafe {
                    *dst_data.get_unchecked_mut(pix_offset) = T::from_elem(col_acc);
                }
            }
        }
    }

    Ok(())
}

/// Apply a fast filter horizontally using cumulative kernel
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with transposed shape (W, H, C).
/// * `half_kernel_x_size` - Half of the kernel at weight 1. The total size would be 2*this+1
pub(crate) fn fast_horizontal_filter<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
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
        )?;

        let mut dst = Image::<_, 1>::from_size_val(img.size(), 0f32)?;
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
                0, 0, 1, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
        )?;

        let mut dst = Image::<u8, 1>::from_size_val(img.size(), 0)?;
        let kernel_x = vec![1, 1, 1];
        let kernel_y = vec![1, 1, 1];
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0, 0, 0, 0, 0,
                0, 1, 1, 1, 0,
                0, 1, 1, 1, 0,
                0, 1, 1, 1, 0,
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

        let kernel_x = vec![1, 1, 1];
        let kernel_y = vec![1, 1, 1];

        let mut img = Image::<u8, 1>::from_size_val(size, 0)?;
        img.as_slice_mut()[12] = 255;

        let mut dst = Image::<u8, 1>::from_size_val(size, 0)?;
        separable_filter(&img, &mut dst, &kernel_x, &kernel_y)?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[0, 0, 0, 0, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 0, 0, 0, 0,
            ]
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
        )?;

        let mut transposed = Image::<_, 1>::from_size_val(size, 0.0)?;

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

        let mut dst = Image::<_, 1>::from_size_val(size, 0.0)?;

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
