use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use kornia_image::{Image, ImageError, ImageSize};

const INF: f32 = 1e20;

pub(crate) fn euclidean_distance(x1: Vec<f32>, x2: Vec<f32>) -> f32 {
    ((x1[0] - x2[0]).powi(2) + (x1[1] - x2[1]).powi(2)).sqrt()
}

/// NOTE: only for testing, extremely slow
pub fn distance_transform_vanilla<A>(image: &Image<f32, 1, A>) -> Image<f32, 1, CpuAllocator>
where
    A: ImageAllocator,
{
    let mut output = vec![0.0f32; image.width() * image.height()];
    let slice = image.as_slice();

    for y in 0..image.height() {
        for x in 0..image.width() {
            let mut min_distance = std::f32::MAX;
            for j in 0..image.height() {
                for i in 0..image.width() {
                    // image.data[[j, i, 0]] > 0.0
                    if slice[j * image.width() + i] > 0.0 {
                        let distance =
                            euclidean_distance(vec![x as f32, y as f32], vec![i as f32, j as f32]);
                        if distance < min_distance {
                            min_distance = distance;
                        }
                    }
                }
            }
            output[y * image.width() + x] = min_distance;
        }
    }

    Image::new(image.size(), output, CpuAllocator).unwrap()
}

/// Computes the Euclidean Distance Transform of a binary image.
///
/// This implementation uses the linear-time algorithm O(N) by Felzenszwalb & Huttenlocher
/// described in "Distance Transforms of Sampled Functions" (2012).
///
/// # Arguments
///
/// * `image` - Input binary image. Non-zero pixels are considered features (distance 0).
///             Zero pixels are considered background.
///
/// # Returns
///
/// A float image where each pixel value is the Euclidean distance to the nearest feature.
/// The output image is always allocated on the CPU.
pub fn distance_transform<A>(
    image: &Image<f32, 1, A>,
) -> Result<Image<f32, 1, CpuAllocator>, ImageError>
where
    A: ImageAllocator,
{
    let width = image.width();
    let height = image.height();
    let num_pixels = width * height;

    // Initialization
    let mut grid = vec![INF; num_pixels];
    let slice = image.as_slice();

    for (i, &val) in slice.iter().enumerate() {
        if val > 0.0 {
            grid[i] = 0.0;
        }
    }

    // Transform along Columns
    let mut f_col = vec![0.0; height];
    let mut d_col = vec![0.0; height];
    let mut v_col = vec![0usize; height];
    let mut z_col = vec![0.0f32; height + 1];

    for x in 0..width {
        for y in 0..height {
            f_col[y] = grid[y * width + x];
        }
        distance_transform_1d(&f_col, &mut d_col, &mut v_col, &mut z_col);
        for y in 0..height {
            grid[y * width + x] = d_col[y];
        }
    }

    // Transform along Rows
    let mut f_row = vec![0.0; width];
    let mut d_row = vec![0.0; width];
    let mut v_row = vec![0usize; width];
    let mut z_row = vec![0.0f32; width + 1];

    for y in 0..height {
        let row_start = y * width;
        for x in 0..width {
            f_row[x] = grid[row_start + x];
        }
        distance_transform_1d(&f_row, &mut d_row, &mut v_row, &mut z_row);
        for x in 0..width {
            grid[row_start + x] = d_row[x];
        }
    }

    // Finalize
    for val in grid.iter_mut() {
        *val = val.sqrt();
    }

    Image::new(ImageSize { width, height }, grid, CpuAllocator)
}

/// Helper function - 1D distance transform using parabolic lower envelope
fn distance_transform_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }

    let mut k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;

    for q in 1..n {
        loop {
            let r = v[k];
            // Calculate intersection of parabola 'r' and 'q'
            let s =
                ((f[q] + (q * q) as f32) - (f[r] + (r * r) as f32)) / (2.0 * (q as f32 - r as f32));

            if s <= z[k] {
                if k == 0 {
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = INF;
                break;
            }
        }
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let r = v[k];
        d[q] = (q as f32 - r as f32).powi(2) + f[r];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn test_accuracy_vs_vanilla() {
        let width = 20;
        let height = 20;
        let mut data = vec![0.0; width * height];
        data[45] = 1.0;
        data[102] = 1.0;
        data[300] = 1.0;

        let image = Image::new(ImageSize { width, height }, data, CpuAllocator).unwrap();

        let expected = distance_transform_vanilla(&image);
        let actual = distance_transform(&image).unwrap();

        for i in 0..actual.as_slice().len() {
            let diff = (expected.as_slice()[i] - actual.as_slice()[i]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at index {}: Vanilla={}, New={}",
                i,
                expected.as_slice()[i],
                actual.as_slice()[i]
            );
        }
    }

    #[test]
    fn distance_transform_smoke() {
        let image = Image::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            vec![
                0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            CpuAllocator,
        )
        .unwrap();

        let output = distance_transform(&image).unwrap();

        println!("Output: {:?}", output.as_slice());
    }
}
