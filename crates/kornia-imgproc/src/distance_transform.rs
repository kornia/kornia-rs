use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use kornia_image::{Image, ImageError, ImageSize};
use rayon::prelude::*;

// We use a large finite number instead of f32::INFINITY to avoid NaN generation
// during arithmetic operations (INF - INF = NaN).
// 1e20 is sufficient for images up to ~1e10 pixels in dimension.
const INF: f32 = 1e20;

/// Computes the Euclidean Distance Transform of a binary image.
///
/// This implementation is highly optimized:
/// 1. Uses the O(N) linear-time algorithm by Felzenszwalb & Huttenlocher (Distance Transforms of Sampled Functions, 2012).
/// 2. Uses Rayon for multi-threading (processing rows in parallel).
/// 3. Uses Transposition to ensure cache-friendly memory access.
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

    // Parallel initialization
    grid.par_iter_mut()
        .zip(slice.par_iter())
        .for_each(|(g, &s)| {
            if s > 0.0 {
                *g = 0.0;
            }
        });

    // Transform along Rows (Parallel)
    // We allocate thread-local scratch buffers to avoid repeated allocations.
    grid.par_chunks_mut(width).for_each_init(
        || (vec![0.0; width], vec![0usize; width], vec![0.0; width + 1]),
        |(f, v, z), row| {
            f.copy_from_slice(row);
            distance_transform_1d(f, row, v, z);
        },
    );

    // Transpose Grid (Cache Optimization)
    let mut grid_t = transpose(&grid, width, height);

    // Transform along Rows (which are actually Columns)
    grid_t.par_chunks_mut(height).for_each_init(
        || {
            (
                vec![0.0; height],
                vec![0usize; height],
                vec![0.0; height + 1],
            )
        },
        |(f, v, z), row| {
            f.copy_from_slice(row);
            distance_transform_1d(f, row, v, z);
        },
    );

    //Transpose Back
    let mut final_grid = transpose(&grid_t, height, width);

    //Finalize: Square Root (Parallel)
    final_grid.par_iter_mut().for_each(|val| {
        *val = val.sqrt();
    });

    Image::new(ImageSize { width, height }, final_grid, CpuAllocator)
}

/// Helper function - 1D distance transform using parabolic lower envelope
///
/// Based on Algorithm from "Distance Transforms of Sampled Functions" (2012).
///
/// # Arguments
/// * f - Source function (1D grid row/column).
/// * d - Output distance buffer.
/// * v - Scratch buffer for parabola vertices (locations).
/// * z - Scratch buffer for parabola boundaries (intersections).
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
            debug_assert!(q != r, "q and r should never be equal");
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
    for (q, d_val) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let r = v[k];
        let dist = q as f32 - r as f32;
        *d_val = dist * dist + f[r];
    }
}

/// Transpose a flat vector representing a 2D grid.
fn transpose<T: Copy + Send + Sync>(data: &[T], width: usize, height: usize) -> Vec<T> {
    let mut output = vec![data[0]; data.len()];
    // Parallel transpose
    output
        .par_chunks_mut(height)
        .enumerate()
        .for_each(|(x, out_row)| {
            for y in 0..height {
                out_row[y] = data[y * width + x];
            }
        });
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::allocator::CpuAllocator;
    use kornia_image::{Image, ImageSize};

    fn euclidean_distance(x1: &[f32], x2: &[f32]) -> f32 {
        ((x1[0] - x2[0]).powi(2) + (x1[1] - x2[1]).powi(2)).sqrt()
    }

    /// NOTE: only for testing, extremely slow
    fn distance_transform_vanilla<A>(image: &Image<f32, 1, A>) -> Image<f32, 1, CpuAllocator>
    where
        A: ImageAllocator,
    {
        let mut output = vec![0.0f32; image.width() * image.height()];
        let slice = image.as_slice();

        for y in 0..image.height() {
            for x in 0..image.width() {
                let mut min_distance = f32::MAX;
                for j in 0..image.height() {
                    for i in 0..image.width() {
                        if slice[j * image.width() + i] > 0.0 {
                            let distance =
                                euclidean_distance(&[x as f32, y as f32], &[i as f32, j as f32]);
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
            // Check feature pixel is exactly 0.0
            if image.as_slice()[i] > 0.0 {
                assert_eq!(actual.as_slice()[i], 0.0, "Feature pixel should be 0.0");
            }

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

        assert_eq!(output.size().width, 3);
        assert_eq!(output.size().height, 4);

        // Check known feature pixels are 0
        assert_eq!(output.as_slice()[2], 0.0); // (2,0)
        assert_eq!(output.as_slice()[6], 0.0); // (0,2)
        assert_eq!(output.as_slice()[10], 0.0); // (1,3)

        // Check a known background pixel
        // (1,1) -> closest is (2,0) dist sqrt(2)=1.414 or (0,2) dist sqrt(2)
        let val = output.as_slice()[4];
        assert!(val > 1.4 && val < 1.5);
    }

    #[test]
    fn test_edge_cases() {
        // All features (all 1.0) -> Output should be all 0.0
        let image = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![1.0; 25],
            CpuAllocator,
        )
        .unwrap();
        let output = distance_transform(&image).unwrap();
        assert!(output.as_slice().iter().all(|&x| x == 0.0));

        // 1xN image
        let image = Image::new(
            ImageSize {
                width: 5,
                height: 1,
            },
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            CpuAllocator,
        )
        .unwrap();
        let output = distance_transform(&image).unwrap();
        let slice = output.as_slice();
        assert_eq!(slice[2], 0.0);
        assert_eq!(slice[1], 1.0);
        assert_eq!(slice[0], 2.0);

        // Single pixel image
        let image = Image::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![1.0],
            CpuAllocator,
        )
        .unwrap();
        let output = distance_transform(&image).unwrap();
        assert_eq!(output.as_slice()[0], 0.0);
    }
}
