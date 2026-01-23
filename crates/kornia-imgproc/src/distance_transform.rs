use kornia_image::allocator::{CpuAllocator, ImageAllocator};
use kornia_image::{Image, ImageError, ImageSize};
use rayon::prelude::*;

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
/// This implementation is highly optimized:
/// 1. Uses the O(N) linear-time algorithm by Felzenszwalb & Huttenlocher.
/// 2. Uses **Rayon** for multi-threading (processing rows in parallel).
/// 3. Uses **Transposition** to ensure cache-friendly memory access.
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
    // This allows us to process "columns" as "rows", enabling SIMD/Cache benefits and Rayon parallelism.
    let mut grid_t = transpose(&grid, width, height);

    // Transform along "Rows" (which are actually Columns)
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
        let dist = q as f32 - r as f32;
        d[q] = dist * dist + f[r];
    }
}

/// Transpose a flat vector representing a 2D grid.
/// This enables efficient memory access for column-wise operations.
fn transpose<T: Copy + Send + Sync>(data: &[T], width: usize, height: usize) -> Vec<T> {
    let mut output = vec![data[0]; data.len()];
    // Parallel transpose
    // We iterate over the rows of the output
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
    }
}
