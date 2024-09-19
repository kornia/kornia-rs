use kornia_image::{Image, ImageError};

/// Compute the pixel intensity histogram of an image.
///
/// NOTE: this is limited to 8-bit 1-channel images.
///
/// # Arguments
///
/// * `src` - The input image to compute the histogram.
/// * `hist` - The output histogram.
/// * `num_bins` - The number of bins to use for the histogram.
///
/// # Returns
///
/// A vector of size `num_bins` containing the histogram.
///
/// # Errors
///
/// Returns an error if the number of bins is invalid.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::histogram::compute_histogram;
///
/// let image = Image::<u8, 1>::new(
///   ImageSize {
///     width: 3,
///     height: 3,
///   },
///   vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
/// ).unwrap();
///
/// let mut histogram = vec![0; 3];
///
/// compute_histogram(&image, &mut histogram, 3).unwrap();
/// assert_eq!(histogram, vec![3, 3, 3]);
/// ```
pub fn compute_histogram(
    src: &Image<u8, 1>,
    hist: &mut Vec<usize>,
    num_bins: usize,
) -> Result<(), ImageError> {
    if num_bins == 0 || num_bins > 256 {
        return Err(ImageError::InvalidHistogramBins(num_bins));
    }

    if hist.len() != num_bins {
        return Err(ImageError::InvalidHistogramBins(num_bins));
    }

    // we assume 8-bit images for now and range [0, 255]
    let scale = 256.0 / num_bins as f32;

    // TODO: check if this can be done in parallel
    src.as_slice().iter().fold(hist, |histogram, &pixel| {
        let bin_pos = (pixel as f32 / scale).floor();
        histogram[bin_pos as usize] += 1;
        histogram
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn test_compute_histogram() -> Result<(), ImageError> {
        let image = Image::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
        )?;

        let mut histogram = vec![0; 3];

        super::compute_histogram(&image, &mut histogram, 3)?;
        assert_eq!(histogram, vec![3, 3, 3]);

        Ok(())
    }
}
