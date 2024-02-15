use crate::image::Image;
use anyhow::Result;

pub fn threshold_binary<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
    threshold: T,
    max_value: T,
) -> Result<Image<T, CHANNELS>>
where
    T: Copy + Clone + Default + Send + Sync + num_traits::NumCast + std::fmt::Debug + PartialOrd,
{
    let mut output = Image::<T, CHANNELS>::from_size(image.image_size())?;

    ndarray::Zip::from(&mut output.data)
        .and(&image.data)
        .par_for_each(|out, &inp| {
            *out = if inp > threshold {
                max_value
            } else {
                num_traits::NumCast::from(0).unwrap()
            };
        });

    Ok(output)
}

//pub fn threshold_binary_inverse(image: &Image, threshold: u8, max_value: u8) -> Image {
//    let mut output = ndarray::Array3::<u8>::zeros(image.data.dim());
//
//    ndarray::Zip::from(&mut output)
//        .and(&image.data)
//        .par_for_each(|out, &inp| {
//            *out = if inp > threshold { 0 } else { max_value };
//        });
//
//    Image { data: output }
//}
//
//pub fn threshold_truncate(image: &Image, threshold: u8) -> Image {
//    let mut output = ndarray::Array3::<u8>::zeros(image.data.dim());
//
//    ndarray::Zip::from(&mut output)
//        .and(&image.data)
//        .par_for_each(|out, &inp| {
//            *out = if inp > threshold { threshold } else { inp };
//        });
//
//    Image { data: output }
//}
//
//pub fn threshold_to_zero(image: &Image, threshold: u8) -> Image {
//    let mut output = ndarray::Array3::<u8>::zeros(image.data.dim());
//
//    ndarray::Zip::from(&mut output)
//        .and(&image.data)
//        .par_for_each(|out, &inp| {
//            *out = if inp > threshold { inp } else { 0 };
//        });
//
//    Image { data: output }
//}
//
//pub fn threshold_to_zero_inverse(image: &Image, threshold: u8) -> Image {
//    let mut output = ndarray::Array3::<u8>::zeros(image.data.dim());
//
//    ndarray::Zip::from(&mut output)
//        .and(&image.data)
//        .par_for_each(|out, &inp| {
//            *out = if inp > threshold { 0 } else { inp };
//        });
//
//    Image { data: output }
//}

// TODO: outsu, triangle

//#[cfg(test)]
//mod tests {
//use crate::image::Image;

//#[test]
//fn threshold_binary() {
//    let data = vec![100u8, 200, 50, 150, 200, 250];
//    let data_expected = vec![0u8, 255, 0, 255, 255, 255];
//    let image = Image::from_shape_vec([1, 2, 3], data);

//    let thresholded = super::threshold_binary(&image, 100, 255);
//    assert_eq!(thresholded.num_channels(), 3);
//    assert_eq!(thresholded.image_size().width, 2);
//    assert_eq!(thresholded.image_size().height, 1);

//    thresholded
//        .data
//        .iter()
//        .zip(data_expected.iter())
//        .for_each(|(x, y)| {
//            assert_eq!(x, y);
//        });
//}

//#[test]
//fn threshold_binary_inverse() {
//    let data = vec![100u8, 200, 50, 150, 200, 250];
//    let data_expected = vec![255u8, 0, 255, 0, 0, 0];
//    let image = Image::from_shape_vec([1, 2, 3], data);

//    let thresholded = super::threshold_binary_inverse(&image, 100, 255);
//    assert_eq!(thresholded.num_channels(), 3);
//    assert_eq!(thresholded.image_size().width, 2);
//    assert_eq!(thresholded.image_size().height, 1);

//    thresholded
//        .data
//        .iter()
//        .zip(data_expected.iter())
//        .for_each(|(x, y)| {
//            assert_eq!(x, y);
//        });
//}

//#[test]
//fn threshold_truncate() {
//    let data = vec![100u8, 200, 50, 150, 200, 250];
//    let data_expected = vec![100u8, 150, 50, 150, 150, 150];
//    let image = Image::from_shape_vec([1, 2, 3], data);

//    let thresholded = super::threshold_truncate(&image, 150);
//    assert_eq!(thresholded.num_channels(), 3);
//    assert_eq!(thresholded.image_size().width, 2);
//    assert_eq!(thresholded.image_size().height, 1);

//    thresholded
//        .data
//        .iter()
//        .zip(data_expected.iter())
//        .for_each(|(x, y)| {
//            assert_eq!(x, y);
//        });
//}

//#[test]
//fn threshold_to_zero() {
//    let data = vec![100u8, 200, 50, 150, 200, 250];
//    let data_expected = vec![0u8, 200, 0, 0, 200, 250];
//    let image = Image::from_shape_vec([1, 2, 3], data);

//    let thresholded = super::threshold_to_zero(&image, 150);
//    assert_eq!(thresholded.num_channels(), 3);
//    assert_eq!(thresholded.image_size().width, 2);
//    assert_eq!(thresholded.image_size().height, 1);

//    thresholded
//        .data
//        .iter()
//        .zip(data_expected.iter())
//        .for_each(|(x, y)| {
//            assert_eq!(x, y);
//        });
//}

//#[test]
//fn threshold_to_zero_inverse() {
//    let data = vec![100u8, 200, 50, 150, 200, 250];
//    let data_expected = vec![100u8, 0, 50, 150, 0, 0];
//    let image = Image::from_shape_vec([1, 2, 3], data);

//    let thresholded = super::threshold_to_zero_inverse(&image, 150);
//    assert_eq!(thresholded.num_channels(), 3);
//    assert_eq!(thresholded.image_size().width, 2);
//    assert_eq!(thresholded.image_size().height, 1);

//    thresholded
//        .data
//        .iter()
//        .zip(data_expected.iter())
//        .for_each(|(x, y)| {
//            assert_eq!(x, y);
//        });
//}
//}
