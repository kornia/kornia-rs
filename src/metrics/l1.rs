use crate::image::Image;

/// Compute the L1 loss between two images.
///
/// The L1 loss is defined as the sum of the absolute differences between the two images.
///
/// The L1 loss is defined as:
///
/// $ L1(a, b) = \frac{1}{N} \sum_{i=1}^{N} |a_i - b_i| $
///
/// where `a` and `b` are the two images and `N` is the number of pixels.
///
/// # Arguments
///
/// * `image1` - The first input image with shape (H, W, C).
/// * `image2` - The second input image with shape (H, W, C).
///
/// # Returns
///
/// The L1 loss between the two images.
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image1 = Image::<f32, 1>::new(
///   ImageSize {
///    width: 2,
///   height: 3,
/// },
/// vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
/// )
/// .unwrap();
///
/// let image2 = Image::<f32, 1>::new(
///  ImageSize {
///   width: 2,
/// height: 3,
/// },
/// vec![5f32, 4f32, 3f32, 2f32, 1f32, 0f32],
/// )
/// .unwrap();
///
/// let l1_loss = kornia_rs::metrics::l1_loss(&image1, &image2);
/// assert_eq!(l1_loss, 3.0);
/// ```
///
/// # Panics
///
/// Panics if the two images have different shapes.
///
/// # References
///
/// [Wikipedia - L1 loss](https://en.wikipedia.org/wiki/Huber_loss)
pub fn l1_loss<const CHANNELS: usize>(
    image1: &Image<f32, CHANNELS>,
    image2: &Image<f32, CHANNELS>,
) -> f32 {
    assert_eq!(image1.size(), image2.size());

    ndarray::Zip::from(&image1.data)
        .and(&image2.data)
        .fold(0f32, |acc, &a, &b| acc + (a - b).abs())
        / (image1.data.len() as f32)
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};

    #[test]
    fn test_l1_loss() {
        let image1 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
        )
        .unwrap();

        let image2 = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![5f32, 4f32, 3f32, 2f32, 1f32, 0f32],
        )
        .unwrap();

        let l1_loss = crate::metrics::l1_loss(&image1, &image2);
        assert_eq!(l1_loss, 3.0);
    }
}
