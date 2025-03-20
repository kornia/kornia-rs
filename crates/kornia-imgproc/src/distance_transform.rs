use kornia_image::Image;

pub(crate) fn euclidean_distance(x1: Vec<f32>, x2: Vec<f32>) -> f32 {
    ((x1[0] - x2[0]).powi(2) + (x1[1] - x2[1]).powi(2)).sqrt()
}

// NOTE: only for testing, extremely slow
/// Finds the distance transform of the input image using a O(N^4) vanilla method.
///
/// # Arguements
///
/// * `image`: The input image whose distance transform you want to find.
///
/// # Returns
///
/// An image whose distance transform is found.
///
pub fn distance_transform_vanilla(image: &Image<f32, 1>) -> Image<f32, 1> {
    let mut output = Image::from_size_val(image.size(), f32::INFINITY).unwrap();
    let width = image.width();

    for y in 0..image.height() {
        for x in 0..width {
            let mut min_distance = f32::INFINITY;
            for j in 0..image.height() {
                for i in 0..width {
                    let idx = j * width + i;
                    if image.storage.as_slice()[idx] > 0.0 {
                        let d =
                            euclidean_distance(vec![x as f32, y as f32], vec![i as f32, j as f32]);
                        min_distance = min_distance.min(d);
                    }
                }
            }
            let idx = y * width + x;
            output.storage.as_mut_slice()[idx] = min_distance;
        }
    }
    output
}

/// Finds the distance transform using the algorithm described by Huttenlocher and Felzenszwaib.
/// This method is relatively quite fast as it computes the transform in O(N^2).
///
/// # Arguements
///
/// * `image`: The input image whose distance transform you want to find.
///
/// # Returns
///
/// An image whose distance transform is found.
///
pub fn distance_transform(image: &Image<f32, 1>) -> Image<f32, 1> {
    let mut distance = Image::from_size_val(image.size(), f32::INFINITY).unwrap();
    let width = image.width();

    // Initialize distances
    for i in 0..image.height() {
        for j in 0..width {
            let idx = i * width + j;
            if image.storage.as_slice()[idx] > 0.0 {
                distance.storage.as_mut_slice()[idx] = 0.0;
            }
        }
    }

    // Forward pass
    for i in 0..image.height() {
        for j in 0..width {
            let idx = i * width + j;
            // If already a foreground pixel, skip.
            if distance.storage.as_slice()[idx] == 0.0 {
                continue;
            }
            if i > 0 {
                let idx_up = (i - 1) * width + j;
                distance.storage.as_mut_slice()[idx] =
                    distance.storage.as_slice()[idx].min(distance.storage.as_slice()[idx_up] + 1.0);
            }
            if j > 0 {
                let idx_left = i * width + (j - 1);
                distance.storage.as_mut_slice()[idx] = distance.storage.as_slice()[idx]
                    .min(distance.storage.as_slice()[idx_left] + 1.0);
            }
        }
    }

    // Backward pass
    for i in (0..image.height()).rev() {
        for j in (0..width).rev() {
            let idx = i * width + j;
            if i < image.height() - 1 {
                let idx_down = (i + 1) * width + j;
                distance.storage.as_mut_slice()[idx] = distance.storage.as_slice()[idx]
                    .min(distance.storage.as_slice()[idx_down] + 1.0);
            }
            if j < width - 1 {
                let idx_right = i * width + (j + 1);
                distance.storage.as_mut_slice()[idx] = distance.storage.as_slice()[idx]
                    .min(distance.storage.as_slice()[idx_right] + 1.0);
            }
        }
    }

    distance
}

#[cfg(test)]
mod tests {
    use crate::distance_transform::{distance_transform, distance_transform_vanilla};
    use kornia_image::Image;

    #[test]
    fn distance_transform_vanilla_smoke() {
        let image = Image::<f32, 1>::new(
            kornia_image::ImageSize {
                width: 3,
                height: 4,
            },
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let output = distance_transform_vanilla(&image);
        println!("{:?}", output.storage.as_slice());
    }

    #[test]
    fn distance_transform_smoke() {
        let image = Image::<f32, 1>::new(
            kornia_image::ImageSize {
                width: 4,
                height: 3,
            },
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let output = distance_transform(&image);
        println!("{:?}", output.storage.as_slice());
    }
}
