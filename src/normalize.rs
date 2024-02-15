use crate::image::Image;
use anyhow::Result;

pub fn normalize_mean_std<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
    mean: &[T; CHANNELS],
    std: &[T; CHANNELS],
) -> Result<Image<T, CHANNELS>, std::io::Error>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    let mut output = ndarray::Array3::<T>::zeros(image.data.dim());

    ndarray::Zip::from(output.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            for i in 0..CHANNELS {
                out[i] = (inp[i] - mean[i]) / std[i];
            }
        });

    Ok(Image { data: output })
}

/// Find the minimum and maximum values in an image.
///
/// Arguments:
///
/// * `image` - The input image of shape (height, width, channels).
///
/// Returns:
///
/// A tuple containing the minimum and maximum values in the image.
///
/// # Errors
///
/// If the image is empty, an error is returned.
pub fn find_min_max<T: PartialOrd, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
) -> Result<(T, T)>
where
    T: Copy,
{
    // get the first element in the image
    let first_element = match image.data.iter().next() {
        Some(x) => x,
        None => return Err(anyhow::anyhow!("Empty image")),
    };

    let mut min = first_element;
    let mut max = first_element;

    for x in image.data.iter() {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }

    Ok((*min, *max))
}
//
//pub fn normalize_min_max(image: &Image<f32>, min: f32, max: f32) -> Image<f32> {
//    let mut output = ndarray::Array3::<f32>::zeros(image.data.dim());
//
//    let (min_val, max_val) = find_min_max(&image);
//
//    ndarray::Zip::from(output.rows_mut())
//        .and(image.data.rows())
//        .par_for_each(|mut out, inp| {
//            for i in 0..image.num_channels() {
//                out[i] = (inp[i] - min_val) * (max - min) / (max_val - min_val) + min;
//            }
//        });
//
//    Image::<f32> { data: output }
//}
//
#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};
    //
    //    #[test]
    //    fn normalize_mean_std() {
    //        let image_data = vec![
    //            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
    //        ];
    //        let image_expected = vec![
    //            -0.5f32, 0.0, -0.5, 0.5, 1.0, 2.5, -0.5, 0.0, -0.5, 0.5, 1.0, 2.5,
    //        ];
    //        let image = Image::from_shape_vec([2, 2, 3], image_data);
    //        let mean = [0.5, 1.0, 0.5];
    //        let std = [1.0, 1.0, 1.0];
    //
    //        let normalized = super::normalize_mean_std(&image, &mean, &std);
    //
    //        assert_eq!(normalized.num_channels(), 3);
    //        assert_eq!(normalized.image_size().width, 2);
    //        assert_eq!(normalized.image_size().height, 2);
    //
    //        normalized
    //            .data
    //            .iter()
    //            .zip(image_expected.iter())
    //            .for_each(|(a, b)| {
    //                assert!((a - b).abs() < 1e-6);
    //            });
    //    }
    //
    #[test]
    fn find_min_max() {
        let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            image_data,
        )
        .unwrap();

        let (min, max) = super::find_min_max(&image).unwrap();

        assert_eq!(min, 0);
        assert_eq!(max, 3);
    }
    //
    //    #[test]
    //    fn normalize_min_max() {
    //        let image_data = vec![
    //            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
    //        ];
    //        let image_expected = vec![
    //            0.0f32, 0.33333334, 0.0, 0.33333334, 0.6666667, 1.0, 0.0, 0.33333334, 0.0, 0.33333334,
    //            0.6666667, 1.0,
    //        ];
    //        let image = Image::from_shape_vec([2, 2, 3], image_data);
    //
    //        let normalized = super::normalize_min_max(&image, 0.0, 1.0);
    //
    //        assert_eq!(normalized.num_channels(), 3);
    //        assert_eq!(normalized.image_size().width, 2);
    //        assert_eq!(normalized.image_size().height, 2);
    //
    //        normalized
    //            .data
    //            .iter()
    //            .zip(image_expected.iter())
    //            .for_each(|(a, b)| {
    //                assert!((a - b).abs() < 1e-6);
    //            });
    //    }
}
