use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

/// Crops a rectangular region from an image.
///
/// Extracts a sub-region from the source image starting at coordinates `(x, y)`
/// with dimensions determined by the destination image size. The operation is
/// performed in parallel for efficiency.
///
/// # Arguments
///
/// * `src` - The source image to crop from
/// * `dst` - The destination image to store the cropped region  
///   (its size determines the crop dimensions)
/// * `x` - The column (x-coordinate) of the top-left corner of the crop region
/// * `y` - The row (y-coordinate) of the top-left corner of the crop region
///
/// # Panics
///
/// Panics if the crop region extends beyond the source image boundaries.
/// Ensure that `x + dst.width() <= src.width()` and `y + dst.height() <= src.height()`.
///
/// # Examples
///
/// Cropping a 2×2 region from a 4×4 grayscale image:
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::crop::crop_image;
///
/// let image = Image::<_, 1, _>::new(ImageSize { width: 4, height: 4 }, vec![
///     0u8, 1, 2, 3,
///     4u8, 5, 6, 7,
///     8u8, 9, 10, 11,
///     12u8, 13, 14, 15
/// ], CpuAllocator).unwrap();
///
/// let mut cropped = Image::<_, 1, _>::from_size_val(ImageSize { width: 2, height: 2 }, 0u8, CpuAllocator).unwrap();
///
/// crop_image(&image, &mut cropped, 1, 1).unwrap();
///
/// assert_eq!(cropped.as_slice(), &[5u8, 6, 9, 10]);
/// ```
pub fn crop_image<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    x: usize,
    y: usize,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync,
{
    let dst_cols = dst.cols();

    dst.as_slice_mut()
        .par_chunks_exact_mut(dst_cols * C)
        .enumerate()
        .for_each(|(i, dst_row)| {
            // get the slice at the top left corner
            let offset = (y + i) * src.cols() * C + x * C;
            let src_slice = &src.as_slice()[offset..offset + dst_cols * C];

            // copy the slice to the destination
            dst_row.copy_from_slice(src_slice);
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_crop() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };

        #[rustfmt::skip]
        let image = Image::<_, 3, _>::new(
            image_size,
            vec![
                0u8, 1, 2, 3, 4, 5,
                6u8, 7, 8, 9, 10, 11,
                12u8, 13, 14, 15, 16, 17,
            ],
            CpuAllocator
        )?;

        let data_expected = vec![9u8, 10, 11, 15, 16, 17];

        let crop_size = ImageSize {
            width: 1,
            height: 2,
        };

        let mut cropped = Image::<_, 3, _>::from_size_val(crop_size, 0u8, CpuAllocator)?;

        super::crop_image(&image, &mut cropped, 1, 1)?;

        assert_eq!(cropped.as_slice(), &data_expected);

        Ok(())
    }
}
