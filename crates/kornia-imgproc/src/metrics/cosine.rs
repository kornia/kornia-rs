use kornia_image::{Image, ImageError};

/// Compute the cosine distance between two images.
///
/// The cosine distance is defined as:
///
/// $ cosine_dist = 1 - \frac {a \cdot b} {\left\| a\right\| \cdot \left\| b\right\|} $
///
/// where `a` and `b` are the pixel values in vectors of two images.
///
/// # Arguments
///
/// * `image1` - The first input image with shape (H, W, C).
/// * `image2` - The second input image with shape (H, W, C).
///
/// # Returns
///
/// The cosine distance between the two images.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::metrics::cosine_dist;
///
/// let image1 = Image::<f32, 1>::new(
///    ImageSize {
///      width: 2,
///      height: 3,
///    },
///    vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
/// )
/// .unwrap();
///
/// let image2 = Image::<f32, 1>::new(
///    ImageSize {
///      width: 2,
///      height: 3,
///    },
///    vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
/// )
/// .unwrap();
///
/// let cosine_dist = cosine_dist(&image1, &image2).unwrap();
/// assert_eq!(cosine_dist, 0f32);
/// ```
///
/// # Panics
///
/// Panics if the two images have different shapes.
///
/// # References
///
/// [LinkedIn post ](https://www.linkedin.com/posts/ashvardanian_100x-10000x-numerical-error-reduction-activity-7242002088803663872-kVsx?utm_source=share&utm_medium=member_desktop)
pub fn cosine_dist<const C: usize>(
    image1: &Image<f32, C>,
    image2: &Image<f32, C>,
) -> Result<f32, ImageError> {
    if image1.size() != image2.size() {
        return Err(ImageError::InvalidImageSize(
            image1.rows(),
            image1.cols(),
            image2.rows(),
            image2.cols(),
        ));
    }

    let (ab, a2, b2) = image1.as_slice().iter().zip(image2.as_slice().iter()).fold(
        (0f32, 0f32, 0f32),
        |(mut ab, mut a2, mut b2), (a, b)| {
            ab += a * b;
            a2 += a * a;
            b2 += b * b;
            (ab, a2, b2)
        },
    );

    let cosine_dist = if a2 == 0. && b2 == 0. {
        0.
    } else if ab == 0. {
        1.
    } else {
        1. - ab / (a2.sqrt() * b2.sqrt())
    };

    Ok(cosine_dist)
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_equal() -> Result<(), ImageError> {
        let image1 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
        )?;
        let image2 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
        )?;
        let cosine_dist = crate::metrics::cosine_dist(&image1, &image2)?;
        assert!((cosine_dist - 0.).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_not_equal() -> Result<(), ImageError> {
        let image1 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0f32, 1f32, 2f32, 3f32],
        )?;
        let image2 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0f32, 3f32, 2f32, 3f32],
        )?;
        let cosine_dist = crate::metrics::cosine_dist(&image1, &image2)?;
        assert!((cosine_dist - 0.088315).abs() < 1e-6);

        Ok(())
    }
}
