use anyhow::Result;
use kornia_image::Image;

pub(crate) fn euclidean_distance(x1: Vec<f32>, x2: Vec<f32>) -> f32 {
    ((x1[0] - x2[0]).powi(2) + (x1[1] - x2[1]).powi(2)).sqrt()
}

// NOTE: only for testing, extremely slow
pub fn distance_transform_vanilla(image: &Image<f32, 1>) -> Image<f32, 1> {
    let mut output = ndarray::Array3::<f32>::zeros(image.data.dim());
    for y in 0..image.height() - 1 {
        for x in 0..image.width() - 1 {
            let mut min_distance = std::f32::MAX;
            for j in 0..image.height() - 1 {
                for i in 0..image.width() - 1 {
                    //println!("{:?} {:?}", i, j);
                    if image.data[[j, i, 0]] > 0.0 {
                        // TODO: pass as array or reference
                        let distance =
                            euclidean_distance(vec![x as f32, y as f32], vec![i as f32, j as f32]);
                        if distance < min_distance {
                            min_distance = distance;
                        }
                    }
                }
            }
            output[[y, x, 0]] = min_distance;
        }
    }
    Image { data: output }
}

// TODO: not fully working
//pub fn distance_transform(image: &Image<f32>) -> Image<f32> {
pub fn distance_transform(image: &Image<f32, 1>) -> Result<Image<f32, 1>> {
    //let mut distance = ndarray::Array3::<f32>::zeros(image.data.dim());
    let mut distance = Image::from_size_val(image.size(), 0.0f32).unwrap();

    // forwards pass

    for i in 0..image.height() {
        for j in 0..image.width() {
            if image.data[[i, j, 0]] > 0.0 {
                distance.data[[i, j, 0]] = 0.0;
            } else {
                if i > 0 {
                    distance.data[[i, j, 0]] =
                        distance.data[[i, j, 0]].min(distance.data[[i - 1, j, 0]] + 1.0)
                }
                if j > 0 {
                    distance.data[[i, j, 0]] =
                        distance.data[[i, j, 0]].min(distance.data[[i, j - 1, 0]] + 1.0)
                }
            }
        }
    }

    // backwards pass

    for i in (0..image.height()).rev() {
        for j in (0..image.width()).rev() {
            if i < image.height() - 1 {
                distance.data[[i, j, 0]] =
                    distance.data[[i, j, 0]].min(distance.data[[i + 1, j, 0]] + 1.0)
            }
            if j < image.width() - 1 {
                distance.data[[i, j, 0]] =
                    distance.data[[i, j, 0]].min(distance.data[[i, j + 1, 0]] + 1.0)
            }
        }
    }

    //for i in (0..image.width()).rev() {
    //    for j in (0..image.height()).rev() {
    //        if i < image.width() - 1 {
    //            distance[[j, i, 0]] = distance[[j, i, 0]].min(distance[[j, i + 1, 0]] + 1.0)
    //        }
    //        if j < image.height() - 1 {
    //            distance[[j, i, 0]] = distance[[j, i, 0]].min(distance[[j + 1, i, 0]] + 1.0)
    //        }
    //    }
    //}

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use crate::distance_transform::distance_transform_vanilla;
    use kornia_image::Image;

    #[test]
    fn distance_transform_vanilla_smoke() {
        let image = Image::<f32, 1>::new(
            kornia_image::ImageSize {
                width: 3,
                height: 4,
            },
            vec![
                0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
        let output = distance_transform_vanilla(&image);
        println!("{:?}", output.data);
    }
}
