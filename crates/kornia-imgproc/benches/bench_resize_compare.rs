use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use kornia_image::{allocator::ImageAllocator, Image, ImageSize};
use kornia_tensor::CpuAllocator;

use kornia_imgproc::{
    interpolation::{grid::meshgrid_from_fn, interpolate_pixel, InterpolationMode},
    parallel,
};

// Old code that allocated mesh grid
pub fn resize_native_old<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), kornia_image::ImageError> {
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());

    // Added division-by-zero guards
    let step_x = if dst_cols > 1 {
        (src.cols() - 1) as f32 / (dst_cols - 1) as f32
    } else {
        0.0
    };
    let step_y = if dst_rows > 1 {
        (src.rows() - 1) as f32 / (dst_rows - 1) as f32
    } else {
        0.0
    };

    let (map_x, map_y) = meshgrid_from_fn(dst_cols, dst_rows, |x, y| {
        Ok((x as f32 * step_x, y as f32 * step_y))
    })?;

    parallel::par_iter_rows_resample(dst, &map_x, &map_y, |&x, &y, dst_pixel| {
        dst_pixel.iter_mut().enumerate().for_each(|(k, pixel)| {
            *pixel = interpolate_pixel(src, x, y, k, interpolation);
        });
    });

    Ok(())
}

// New code that does on-the-fly calculation for coordinates
pub fn resize_native_new<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    interpolation: InterpolationMode,
) -> Result<(), kornia_image::ImageError> {
    if src.size() == dst.size() {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return Ok(());
    }

    let (dst_rows, dst_cols) = (dst.rows(), dst.cols());

    // Added division-by-zero guards
    let step_x = if dst_cols > 1 {
        (src.cols() - 1) as f32 / (dst_cols - 1) as f32
    } else {
        0.0
    };
    let step_y = if dst_rows > 1 {
        (src.rows() - 1) as f32 / (dst_rows - 1) as f32
    } else {
        0.0
    };

    kornia_imgproc::parallel::par_iter_rows_indexed_mut(dst, |row_idx, row| {
        let y_src = row_idx as f32 * step_y;

        for (col_idx, pix) in row.chunks_exact_mut(C).enumerate() {
            let x_src = col_idx as f32 * step_x;

            pix.iter_mut().enumerate().for_each(|(k, p)| {
                *p = kornia_imgproc::interpolation::interpolate_pixel(
                    src,
                    x_src,
                    y_src,
                    k,
                    interpolation,
                );
            });
        }
    });

    Ok(())
}

// BENCHMARK

fn bench_resize_native(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResizeNativeCompare");

    let sizes = [(256, 256), (512, 512), (1024, 1024)];

    for (w, h) in sizes {
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize {
                width: w,
                height: h,
            },
            0.5,
            CpuAllocator,
        )
        .unwrap();

        let mut dst_old = Image::<f32, 3, _>::from_size_val(
            ImageSize {
                width: w / 2,
                height: h / 2,
            },
            0.0,
            CpuAllocator,
        )
        .unwrap();
        let mut dst_new = dst_old.clone();

        let label = format!("{w}x{h}");

        // OLD
        group.bench_with_input(
            BenchmarkId::new("old_resize_native", &label),
            &src,
            |b, src| {
                b.iter(|| {
                    resize_native_old(
                        black_box(src),
                        black_box(&mut dst_old),
                        black_box(InterpolationMode::Nearest),
                    )
                    .unwrap();
                })
            },
        );

        // NEW
        group.bench_with_input(
            BenchmarkId::new("new_resize_native", &label),
            &src,
            |b, src| {
                b.iter(|| {
                    resize_native_new(
                        black_box(src),
                        black_box(&mut dst_new),
                        black_box(InterpolationMode::Nearest),
                    )
                    .unwrap();
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_resize_native);
criterion_main!(benches);
