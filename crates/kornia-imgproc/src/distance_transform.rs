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
            let mut min_distance = f32::MAX;
            for j in 0..image.height() {
                for i in 0..image.width() {
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

/// Executor for computing the Euclidean Distance Transform.
///
/// This struct maintains internal buffers that can be reused across multiple
/// calls to avoid the overhead of frequent memory allocations.
pub struct DistanceTransformExecutor {
    grid: Vec<f32>,
    scratch: Vec<f32>,
}
impl Default for DistanceTransformExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceTransformExecutor {
    /// Creates a new `DistanceTransformExecutor` with empty buffers.
    pub fn new() -> Self {
        Self {
            grid: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Computes the Euclidean Distance Transform of a binary image.
    pub fn execute<A>(
        &mut self,
        image: &Image<f32, 1, A>,
    ) -> Result<Image<f32, 1, CpuAllocator>, ImageError>
    where
        A: ImageAllocator,
    {
        let width = image.width();
        let height = image.height();
        let num_pixels = width * height;

        // Resize internal workspace if the image dimensions have changed.
        if self.grid.len() != num_pixels {
            self.grid.resize(num_pixels, 0.0);
            self.scratch.resize(num_pixels, 0.0);
        }

        let src_slice = image.as_slice();

        // Parallelize over rows and fuse binary-to-INF conversion with the transform.
        self.grid
            .par_chunks_mut(width)
            .zip(src_slice.par_chunks(width))
            .for_each_init(
                || (vec![0.0; width], vec![0usize; width], vec![0.0; width + 1]),
                |(f, v, z), (grid_row, src_row)| {
                    for (i, &val) in src_row.iter().enumerate() {
                        f[i] = if val > 0.0 { 0.0 } else { INF };
                    }
                    distance_transform_1d(f, grid_row, v, z);
                },
            );

        transpose_map(&self.grid, &mut self.scratch, width, height, |x| x);

        self.scratch.par_chunks_mut(height).for_each_init(
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

        // create a fresh vector for the final Image to be returned.
        let mut final_data = vec![0.0f32; num_pixels];
        transpose_map(&self.scratch, &mut final_data, height, width, |x| x.sqrt());

        Image::new(ImageSize { width, height }, final_data, CpuAllocator)
    }
}

/// 1D Parabolic Distance Transform (Safe Rust)
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
            let numerator = (f[q] + (q * q) as f32) - (f[r] + (r * r) as f32);
            let denominator = 2.0 * (q as f32 - r as f32);
            let s = numerator / denominator;

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

/// Generic Parallel Transpose with Mapper
fn transpose_map<F>(src: &[f32], dst: &mut [f32], width: usize, height: usize, op: F)
where
    F: Fn(f32) -> f32 + Sync + Send,
{
    dst.par_chunks_mut(height)
        .enumerate()
        .for_each(|(x, dst_row)| {
            for y in 0..height {
                dst_row[y] = op(src[y * width + x]);
            }
        });
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

        let image =
            Image::<f32, 1, _>::new(ImageSize { width, height }, data, CpuAllocator).unwrap();
        let expected = distance_transform_vanilla(&image);

        let mut executor = DistanceTransformExecutor::new();
        let actual = executor.execute(&image).unwrap();

        for i in 0..actual.as_slice().len() {
            let diff = (expected.as_slice()[i] - actual.as_slice()[i]).abs();
            assert!(diff < 1e-4);
        }
    }

    #[test]
    fn distance_transform_smoke() {
        let image = Image::<f32, 1, _>::new(
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

        let mut executor = DistanceTransformExecutor::new();
        let output = executor.execute(&image).unwrap();

        assert_eq!(output.size().width, 3);
        assert_eq!(output.size().height, 4);
        assert_eq!(output.as_slice()[2], 0.0);
    }
}
