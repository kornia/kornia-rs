use anyhow::Result;

use crate::image::Image;

/// Compute the pixel intensity histogram of an image.
///
/// NOTE: this is limited to 8-bit 1-channel images.
///
/// # Arguments
///
/// * `image` - The input image to compute the histogram.
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
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::histogram::compute_histogram;
///
/// let image = Image::<u8, 1>::new(
///    ImageSize {
///      width: 3,
///      height: 3,
///  },
/// vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
/// ).unwrap();
///
/// let histogram = compute_histogram(&image, 3).unwrap();
/// assert_eq!(histogram, vec![3, 3, 3]);
/// ```
pub fn compute_histogram(image: &Image<u8, 1>, num_bins: usize) -> Result<Vec<usize>> {
    if num_bins == 0 || num_bins > 256 {
        return Err(anyhow::anyhow!(
            "Invalid number of bins. Must be in the range [1, 256]."
        ));
    }

    // we assume 8-bit images for now and range [0, 255]
    let scale = 256.0 / num_bins as f32;

    // TODO: check if this can be done in parallel
    let mut histogram = vec![0; num_bins];
    image.data.fold(&mut histogram, |histogram, &pixel| {
        let bin_pos = (pixel as f32 / scale).floor();
        histogram[bin_pos as usize] += 1;
        histogram
    });

    Ok(histogram)
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};
    use anyhow::Result;

    #[test]
    fn test_compute_histogram() -> Result<()> {
        let image = Image::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
        )?;
        let histogram = super::compute_histogram(&image, 3)?;
        assert_eq!(histogram, vec![3, 3, 3]);

        Ok(())
    }
}
