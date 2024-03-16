use crate::image::Image;

/// Compute the Huber loss between two images.
///
/// The Huber loss is a robust loss function that is less sensitive to outliers in data than the squared error loss.
///
/// The Huber loss is defined as:
///
/// $ L_{\delta}(a, b) = \begin{cases} \frac{1}{2}(a - b)^2 & \text{if } |a - b| \leq \delta \\ \delta(|a - b| - \frac{1}{2}\delta) & \text{otherwise} \end{cases} $
///
/// where `a` and `b` are the two images and `delta` is the threshold.
///
/// # Arguments
///
/// * `image1` - The first input image with shape (H, W, C).
/// * `image2` - The second input image with shape (H, W, C).
/// * `delta` - The threshold value.
///
/// # Returns
///
/// The Huber loss between the two images.
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image1 = Image::<f32, 1>::new(
///    ImageSize {
///       width: 2,
///      height: 3,
/// },
/// vec![0f32, 1f32, 2f32, 3f32, 4f32, 5f32],
/// )
/// .unwrap();
///
/// let image2 = Image::<f32, 1>::new(
///   ImageSize {
///     width: 2,
///   height: 3,
/// },
/// vec![5f32, 4f32, 3f32, 2f32, 1f32, 0f32],
/// )
/// .unwrap();
///
/// let huber = kornia_rs::metrics::huber(&image1, &image2, 1.0);
/// assert_eq!(huber, 2.5);
/// ```
///
/// # Panics
///
/// Panics if the two images have different shapes.
///
/// # References
///
/// [Wikipedia - Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
pub fn huber<const CHANNELS: usize>(
    image1: &Image<f32, CHANNELS>,
    image2: &Image<f32, CHANNELS>,
    delta: f32,
) -> f32 {
    assert_eq!(image1.size(), image2.size());

    ndarray::Zip::from(&image1.data)
        .and(&image2.data)
        .fold(0f32, |acc, &a, &b| {
            let diff = a - b;
            if diff.abs() <= delta {
                acc + 0.5 * diff.powi(2)
            } else {
                acc + delta * (diff.abs() - 0.5 * delta)
            }
        })
        / (image1.data.len() as f32)
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};

    #[test]
    fn test_huber() {
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
        let huber = super::huber(&image1, &image2, 1.0);
        assert_eq!(huber, 2.5);
    }
}
