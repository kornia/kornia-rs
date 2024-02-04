use crate::image::Image;

pub(crate) fn euclidean_distance(x1: Vec<f64>, x2: Vec<f64>) -> f64 {
    ((x1[0] - x2[0]).powi(2) + (x1[1] - x2[1]).powi(2)).sqrt()
}

// NOTE: only for testing, extremely slow
pub fn distance_transform_vanilla(image: &Image<f64>) -> Image<f64> {
    let mut output = ndarray::Array3::<f64>::zeros(image.data.dim());
    for y in 0..image.height() - 1 {
        for x in 0..image.width() - 1 {
            let mut min_distance = std::f64::MAX;
            for j in 0..image.height() - 1 {
                for i in 0..image.width() - 1 {
                    //println!("{:?} {:?}", i, j);
                    if image.data[[j, i, 0]] > 0.0 {
                        // TODO: pass as array or reference
                        let distance =
                            euclidean_distance(vec![x as f64, y as f64], vec![i as f64, j as f64]);
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
pub fn distance_transform(image: &Image<f32>) -> Image<f32> {
    let mut distance = ndarray::Array3::<f32>::zeros(image.data.dim());

    // forwards pass

    for i in 0..image.height() {
        for j in 0..image.width() {
            if image.data[[i, j, 0]] > 0.0 {
                distance[[i, j, 0]] = 0.0;
            } else {
                if i > 0 {
                    distance[[i, j, 0]] = distance[[i, j, 0]].min(distance[[i - 1, j, 0]] + 1.0)
                }
                if j > 0 {
                    distance[[i, j, 0]] = distance[[i, j, 0]].min(distance[[i, j - 1, 0]] + 1.0)
                }
            }
        }
    }

    // backwards pass

    for i in (0..image.width()).rev() {
        for j in (0..image.height()).rev() {
            if i < image.width() - 1 {
                distance[[j, i, 0]] = distance[[j, i, 0]].min(distance[[j, i + 1, 0]] + 1.0)
            }
            if j < image.height() - 1 {
                distance[[j, i, 0]] = distance[[j, i, 0]].min(distance[[j + 1, i, 0]] + 1.0)
            }
        }
    }

    Image { data: distance }
}

#[cfg(test)]
mod tests {
    use crate::distance_transform::distance_transform_vanilla;
    use crate::image::Image;

    #[test]
    fn distance_transform_vanilla_smoke() {
        let image = Image::from_shape_vec(
            [2, 2, 3],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let output = distance_transform_vanilla(&image);
        println!("{:?}", output.data);
    }
}
