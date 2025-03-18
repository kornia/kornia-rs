use crate::distance_transform::distance_transform;
use anyhow::Result;
use kornia_image::{Image, ImageError};

/// Performs morphological dilation on a grayscale image.
///
/// This function sets all pixels within a distance `k` of any foreground pixel
/// to white (`255.0`). A pixel is considered part of the foreground if it has
/// a nonzero intensity.
///
/// # Arguments
///
/// * `src` - A reference to the source grayscale image (with `f32` pixel values).
/// * `k` - The dilation radius, specifying the maximum distance from a foreground
///         pixel that should also be set to white.
///
/// # Returns
///
/// * A new dilated image, where pixels within `k` distance from the foreground
///   are set to `255.0`, and others remain `0.0`.
///
pub fn dilate(src: &Image<f32, 1>, k: u8) -> Result<Image<f32, 1>> {
    let mut transformed = distance_transform(&src)?;
    for p in transformed.data.iter_mut() {
        *p = if *p <= k as f32 { 255.0 } else { 0.0 };
    }
    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;

    #[test]
    fn test_dilate() {
        let image = Image::<f32, 1>::new(
            kornia_image::ImageSize {
                width: 5,
                height: 5,
            },
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 0.0,
                0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();

        let result = dilate(&image, 1).unwrap();

        println!("{:?}", result.data);

        // checking if the pixels near the original are activated.
        let expected = vec![
            0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 0.0, 255.0, 255.0, 255.0, 255.0,
            255.0, 0.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0,
        ];

        assert_eq!(result.data.as_slice().unwrap(), expected.as_slice());
    }
}
