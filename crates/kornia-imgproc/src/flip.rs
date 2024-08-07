use kornia_image::{Image, ImageError};

/// Flip the input image horizontally.
///
/// # Arguments
///
/// * `image` - The input image with shape (H, W, C).
///
/// # Returns
///
/// The flipped image.
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
/// let flipped: Image<f32, 3> = horizontal_flip(&image).unwrap();
///
/// assert_eq!(flipped.size().width, 2);
/// assert_eq!(flipped.size().height, 3);
/// ```
pub fn horizontal_flip<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
) -> Result<Image<T, CHANNELS>, ImageError>
where
    T: Copy,
{
    let mut img = image.clone();

    img.data
        .axis_iter_mut(ndarray::Axis(0))
        .for_each(|mut row| {
            let mut i = 0;
            let mut j = image.width() - 1;
            while i < j {
                for c in 0..CHANNELS {
                    row.swap((i, c), (j, c));
                }
                i += 1;
                j -= 1;
            }
        });

    Ok(img)
}

/// Flip the input image vertically.
///
/// # Arguments
///
/// * `image` - The input image with shape (H, W, C).
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
pub fn vertical_flip<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
) -> Result<Image<T, CHANNELS>, ImageError>
where
    T: Copy,
{
    let mut img = image.clone();

    img.data
        .axis_iter_mut(ndarray::Axis(1))
        .for_each(|mut col| {
            let mut i = 0;
            let mut j = image.height() - 1;
            while i < j {
                for c in 0..CHANNELS {
                    col.swap((i, c), (j, c));
                }
                i += 1;
                j -= 1;
            }
        });

    Ok(img)
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_hflip() -> Result<(), ImageError> {
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0u8, 1, 2, 3, 4, 5],
        )?;
        let data_expected = vec![1u8, 0, 3, 2, 5, 4];
        let flipped = super::horizontal_flip(&image)?;
        assert_eq!(
            flipped.data.as_slice().expect("could not convert to slice"),
            &data_expected
        );
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
        assert_eq!(
            flipped.data.as_slice().expect("could not convert to slice"),
            &data_expected
        );
        Ok(())
    }
}
