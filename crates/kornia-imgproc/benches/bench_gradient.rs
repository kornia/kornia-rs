use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::{Image, ImageError};
use kornia_imgproc::filter::{
    kernels, spatial_gradient_float, spatial_gradient_float_by_separable_filter,
    spatial_gradient_float_parallel,
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_rayon_parallel<const C: usize>(
    src: &Image<f32, C>,
    dx: &mut Image<f32, C>,
    dy: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .par_chunks_mut(cols * C)
        .zip(dy.as_slice_mut().par_chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            dx_row
                .par_chunks_mut(C)
                .zip(dy_row.par_chunks_mut(C))
                .enumerate()
                .for_each(|(c, (dx_c, dy_c))| {
                    for ch in 0..C {
                        let mut sum_x = 0.0;
                        let mut sum_y = 0.0;
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let row = (r + dy).min(src.rows()).max(1) - 1;
                                let col = (c + dx).min(src.cols()).max(1) - 1;
                                let src_pix_offset = (row * src.cols() + col) * C + ch;
                                let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                                sum_x += val * sobel_x[dy][dx];
                                sum_y += val * sobel_y[dy][dx];
                            }
                        }
                        dx_c[ch] = sum_x;
                        dy_c[ch] = sum_y;
                    }
                });
        });

    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_rayon_row_parallel<const C: usize>(
    src: &Image<f32, C>,
    dx: &mut Image<f32, C>,
    dy: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .par_chunks_mut(cols * C)
        .zip(dy.as_slice_mut().par_chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            for c in 0..cols {
                let mut sum_x = [0.0; C];
                let mut sum_y = [0.0; C];
                for dy in 0..3 {
                    for dx in 0..3 {
                        let row = (r + dy).min(src.rows()).max(1) - 1;
                        let col = (c + dx).min(src.cols()).max(1) - 1;
                        for ch in 0..C {
                            let src_pix_offset = (row * src.cols() + col) * C + ch;
                            let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                            sum_x[ch] += val * sobel_x[dy][dx];
                            sum_y[ch] += val * sobel_y[dy][dx];
                        }
                    }
                }
                dx_row[c * C..(c + 1) * C].copy_from_slice(&sum_x);
                dy_row[c * C..(c + 1) * C].copy_from_slice(&sum_y);
            }
        });
    Ok(())
}

/// Compute the first order image derivative in both x and y using a Sobel operator.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W).
/// * `dst` - The destination image with shape (H, W, 2).
pub fn spatial_gradient_float_row_parallel<const C: usize>(
    src: &Image<f32, C>,
    dx: &mut Image<f32, C>,
    dy: &mut Image<f32, C>,
) -> Result<(), ImageError> {
    if src.size() != dx.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dx.cols(),
            dx.rows(),
        ));
    }

    if src.size() != dy.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dy.cols(),
            dy.rows(),
        ));
    }

    let (sobel_x, sobel_y) = kernels::normalized_sobel_kernel3();
    let cols = src.cols();

    let src_data = src.as_slice();

    dx.as_slice_mut()
        .chunks_mut(cols * C)
        .zip(dy.as_slice_mut().chunks_mut(cols * C))
        .enumerate()
        .for_each(|(r, (dx_row, dy_row))| {
            for c in 0..cols {
                let mut sum_x = [0.0; C];
                let mut sum_y = [0.0; C];
                for dy in 0..3 {
                    for dx in 0..3 {
                        let row = (r + dy).min(src.rows()).max(1) - 1;
                        let col = (c + dx).min(src.cols()).max(1) - 1;
                        for ch in 0..C {
                            let src_pix_offset = (row * src.cols() + col) * C + ch;
                            let val = unsafe { src_data.get_unchecked(src_pix_offset) };
                            sum_x[ch] += val * sobel_x[dy][dx];
                            sum_y[ch] += val * sobel_y[dy][dx];
                        }
                    }
                }
                dx_row[c * C..(c + 1) * C].copy_from_slice(&sum_x);
                dy_row[c * C..(c + 1) * C].copy_from_slice(&sum_y);
            }
        });
    Ok(())
}

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spatial Gradient");

    for (width, height) in [(512, 512), (1024, 1024), (2048, 2048)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_data = vec![0f32; width * height * 3];
        let image_size = [*width, *height].into();

        let image = Image::<_, 3>::new(image_size, image_data).unwrap();

        // output image
        let output_dx = Image::from_size_val(image.size(), 0.0).unwrap();
        let output_dy = Image::from_size_val(image.size(), 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "spatial_gradient_float_by_separable_filter",
                &parameter_string,
            ),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    black_box(spatial_gradient_float_by_separable_filter(
                        src, &mut dx, &mut dy,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float_parallel(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_rayon_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float_rayon_parallel(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "spatial_gradient_float_rayon_row_parallel",
                &parameter_string,
            ),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    black_box(spatial_gradient_float_rayon_row_parallel(
                        src, &mut dx, &mut dy,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_row_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float_row_parallel(src, &mut dx, &mut dy)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gradient);
criterion_main!(benches);
