use kornia_core::SafeTensorType;
use kornia_image::{Image, ImageError};
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
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::flip::horizontal_flip;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0f32; 2 * 3 * 3],
/// )
/// .unwrap();
///
/// let mut flipped = Image::<f32, 3>::from_size_val(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     0.0
/// )
/// .unwrap();
///
/// horizontal_flip(&image, &mut flipped).unwrap();
/// ```
pub fn horizontal_flip<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
) -> Result<(), ImageError>
where
    T: SafeTensorType,
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
            let n = src.cols() - 1;
            for i in 0..=n {
                for c in 0..C {
                    dst_row[i * C + c] = src_row[(n - i) * C + c];
                }
            }
        });

    Ok(())
}

/// Flip the input image vertically.
///
/// # Arguments
///
/// * `src` - The input image with shape (H, W, C).
///
/// # Returns
///
/// The flipped image.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::flip::vertical_flip;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 3,
///     },
///     vec![0f32; 2 * 3 * 3],
/// )
/// .unwrap();
///
/// let flipped: Image<f32, 3> = vertical_flip(&image).unwrap();
///
/// assert_eq!(flipped.size().width, 2);
/// assert_eq!(flipped.size().height, 3);
/// ```
pub fn vertical_flip<T, const C: usize>(src: &Image<T, C>) -> Result<Image<T, C>, ImageError>
where
    T: SafeTensorType,
{
    let mut dst = src.clone();

    // TODO: improve this implementation
    for i in 0..src.cols() {
        let mut j = src.rows() - 1;
        for k in 0..src.rows() / 2 {
            for c in 0..C {
                let idx_i = i * C + c + k * src.cols() * C;
                let idx_j = i * C + c + j * src.cols() * C;
                dst.as_slice_mut().swap(idx_i, idx_j);
            }
            j -= 1;
        }
    }

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_hflip() -> Result<(), ImageError> {
        let image_size = ImageSize {
            width: 2,
            height: 3,
        };
        let image = Image::<_, 1>::new(image_size, vec![0u8, 1, 2, 3, 4, 5])?;
        let data_expected = vec![1u8, 0, 3, 2, 5, 4];
        let mut flipped = Image::<_, 1>::from_size_val(image_size, 0u8)?;
        super::horizontal_flip(&image, &mut flipped)?;
        assert_eq!(flipped.as_slice(), &data_expected);
        Ok(())
    }

    #[test]
    fn test_vflip() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0u8, 1, 2, 3, 4, 5],
        )?;
        let data_expected = vec![4u8, 5, 2, 3, 0, 1];
        let flipped = super::vertical_flip(&image)?;
        assert_eq!(flipped.as_slice(), &data_expected);
        Ok(())
    }
}
