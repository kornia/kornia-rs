use crate::image::Image;

pub fn normalize_mean_std(image: &Image<f32>, mean: &[f32; 3], std: &[f32; 3]) -> Image<f32> {
    assert_eq!(image.num_channels(), 3);
    let mut output = ndarray::Array3::<f32>::zeros(image.data.dim());

    ndarray::Zip::from(output.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            for i in 0..image.num_channels() {
                out[i] = (inp[i] - mean[i]) / std[i];
            }
        });

    Image::<f32> { data: output }
}

// TODO: implement this with generics
pub fn find_min_max(image: &Image) -> (u8, u8) {
    let mut min = &u8::MAX;
    let mut max = &u8::MIN;

    for x in image.data.iter() {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }

    (*min, *max)
}

pub fn normalize_min_max(image: &Image<f32>, min: f32, max: f32) -> Image<f32> {
    let mut output = ndarray::Array3::<f32>::zeros(image.data.dim());

    let (min_val, max_val) = find_min_max(&image.cast());

    let min_val = min_val as f32;
    let max_val = max_val as f32;

    ndarray::Zip::from(output.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            for i in 0..image.num_channels() {
                out[i] = (inp[i] - min_val) * (max - min) / (max_val - min_val) + min;
            }
        });

    Image::<f32> { data: output }
}

#[cfg(test)]
mod tests {
    use crate::image::Image;

    #[test]
    fn normalize_mean_std() {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let image_expected = vec![
            -0.5f32, 0.0, -0.5, 0.5, 1.0, 2.5, -0.5, 0.0, -0.5, 0.5, 1.0, 2.5,
        ];
        let image = Image::from_shape_vec([2, 2, 3], image_data);
        let mean = [0.5, 1.0, 0.5];
        let std = [1.0, 1.0, 1.0];

        let normalized = super::normalize_mean_std(&image, &mean, &std);

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.image_size().width, 2);
        assert_eq!(normalized.image_size().height, 2);

        normalized
            .data
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
    }

    #[test]
    fn find_min_max() {
        let image_data = vec![0u8, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3];
        let image = Image::from_shape_vec([2, 2, 3], image_data);

        let (min, max) = super::find_min_max(&image);

        assert_eq!(min, 0);
        assert_eq!(max, 3);
    }

    #[test]
    fn normalize_min_max() {
        let image_data = vec![
            0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let image_expected = vec![
            0.0f32, 0.33333334, 0.0, 0.33333334, 0.6666667, 1.0, 0.0, 0.33333334, 0.0, 0.33333334,
            0.6666667, 1.0,
        ];
        let image = Image::from_shape_vec([2, 2, 3], image_data);

        let normalized = super::normalize_min_max(&image, 0.0, 1.0);

        assert_eq!(normalized.num_channels(), 3);
        assert_eq!(normalized.image_size().width, 2);
        assert_eq!(normalized.image_size().height, 2);

        normalized
            .data
            .iter()
            .zip(image_expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });
    }
}
