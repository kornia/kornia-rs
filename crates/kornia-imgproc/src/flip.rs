use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

/// Flip the input image horizontally.
///
/// # Arguments
///
/// * `src` - The input image with shape (H, W, C).
/// * `dst` - The output image with shape (H, W, C).
///
/// Precondition: the input and output images must have the same size.
///
/// # Errors
///
/// Returns an error if the sizes of `src` and `dst` do not match.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::flip::horizontal_flip;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0f32; 2 * 3 * 3],
///     CpuAllocator
/// )
/// .unwrap();
///
/// let mut flipped = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// horizontal_flip(&image, &mut flipped).unwrap();
/// ```
pub fn horizontal_flip<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    dst.as_slice_mut()
        .par_chunks_exact_mut(src.cols() * C)
        .zip_eq(src.as_slice().par_chunks_exact(src.cols() * C))
        .for_each(|(dst_row, src_row)| {
            dst_row
                .chunks_exact_mut(C)
                .zip(src_row.chunks_exact(C).rev())
                .for_each(|(dst_pixel, src_pixel)| {
                    dst_pixel.copy_from_slice(src_pixel);
                })
        });

    Ok(())
}

/// Flip the input image vertically.
///
/// # Arguments
///
/// * `src` - The input image with shape (H, W, C).
/// * `dst` - The output image with shape (H, W, C).
///
/// Precondition: the input and output images must have the same size.
///
/// # Errors
///
/// Returns an error if the sizes of `src` and `dst` do not match.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::flip::vertical_flip;
///
/// let image = Image::<f32, 3, _>::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0f32; 2 * 3 * 3],
///     CpuAllocator,
/// )
/// .unwrap();
///
/// let mut flipped = Image::<f32, 3, _>::from_size_val(image.size(), 0.0, CpuAllocator).unwrap();
///
/// vertical_flip(&image, &mut flipped).unwrap();
///
/// ```
pub fn vertical_flip<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let row_len = src.cols() * C;

    dst.as_slice_mut()
        .par_chunks_exact_mut(row_len)
        .zip(src.as_slice().par_chunks_exact(row_len).rev())
        .for_each(|(dst_row, src_row)| {
            dst_row.copy_from_slice(src_row);
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_hflip() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };
        let image = Image::<_, 3, _>::new(
            image_size,
            vec![
                0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            ],
            CpuAllocator,
        )?;
        let data_expected = vec![
            3u8, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14,
        ];
        let mut flipped = Image::<_, 3, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::horizontal_flip(&image, &mut flipped)?;
        assert_eq!(flipped.as_slice(), &data_expected);
        Ok(())
    }

    #[test]
    fn test_vflip() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };
        let image = Image::<_, 3, _>::new(
            image_size,
            vec![
                0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            ],
            CpuAllocator,
        )?;
        let data_expected = vec![
            12u8, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5,
        ];
        let mut flipped = Image::<_, 3, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::vertical_flip(&image, &mut flipped)?;
        assert_eq!(flipped.as_slice(), &data_expected);
        Ok(())
    }

    #[test]
    fn test_horizontal_flip_single_row() -> Result<(), ImageError> {
        // A single-row (rows=1) image flipped horizontally must reverse the pixel order.
        // Image layout: 1 row × 4 columns × 1 channel.
        let image_size = ImageSize {
            width: 4,
            height: 1,
        };
        let image = Image::<_, 1, _>::new(image_size, vec![10u8, 20, 30, 40], CpuAllocator)?;
        let mut flipped = Image::<_, 1, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::horizontal_flip(&image, &mut flipped)?;
        // Columns reversed: [40, 30, 20, 10]
        assert_eq!(flipped.as_slice(), &[40u8, 30, 20, 10]);
        Ok(())
    }

    #[test]
    fn test_vertical_flip_single_column() -> Result<(), ImageError> {
        // A single-column (cols=1) image flipped vertically must reverse the row order.
        // Image layout: 4 rows × 1 column × 1 channel.
        let image_size = ImageSize {
            width: 1,
            height: 4,
        };
        let image = Image::<_, 1, _>::new(image_size, vec![10u8, 20, 30, 40], CpuAllocator)?;
        let mut flipped = Image::<_, 1, _>::from_size_val(image_size, 0u8, CpuAllocator)?;
        super::vertical_flip(&image, &mut flipped)?;
        // Rows reversed: [40, 30, 20, 10]
        assert_eq!(flipped.as_slice(), &[40u8, 30, 20, 10]);
        Ok(())
    }
}
