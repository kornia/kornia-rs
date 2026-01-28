use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{allocator::CpuAllocator, allocator::ImageAllocator, Image, ImageSize};
use kornia_imgproc::distance_transform::distance_transform;
use opencv::{core, imgproc, prelude::*};

fn euclidean_distance(x1: &[f32], x2: &[f32]) -> f32 {
    ((x1[0] - x2[0]).powi(2) + (x1[1] - x2[1]).powi(2)).sqrt()
}

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

fn bench_distance_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("DistanceTransform");

    // Test Small Sizes (include Vanilla for comparison)
    for (width, height) in [(64, 64), (128, 128)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        // Rust Image Setup
        let mut data = vec![0.0f32; width * height];
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                data[i * width + i] = 1.0;
            }
        }
        let image = Image::new(
            ImageSize {
                width: *width,
                height: *height,
            },
            data.clone(),
            CpuAllocator,
        )
        .unwrap();

        // OpenCV Mat Setup
        let mut cv_src = core::Mat::zeros(*height as i32, *width as i32, core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                *cv_src.at_2d_mut::<u8>(i as i32, i as i32).unwrap() = 255;
            }
        }
        let mut cv_dst = core::Mat::default();

        // 1. Bench Vanilla (Rust O(N^2))
        group.bench_with_input(
            BenchmarkId::new("vanilla_on2", &parameter_string),
            &image,
            |b, i| b.iter(|| std::hint::black_box(distance_transform_vanilla(i))),
        );

        // 2. Bench New (Rust O(N))
        group.bench_with_input(
            BenchmarkId::new("kornia_linear", &parameter_string),
            &image,
            |b, i| b.iter(|| std::hint::black_box(distance_transform(i))),
        );

        // 3. Bench OpenCV
        group.bench_function(BenchmarkId::new("opencv", &parameter_string), |b| {
            b.iter(|| {
                imgproc::distance_transform(
                    &cv_src,
                    &mut cv_dst,
                    imgproc::DIST_L2,
                    imgproc::DIST_MASK_5,
                    core::CV_32F,
                )
                .unwrap()
            })
        });
    }

    // Test Large Sizes (Exclude Vanilla because it's too slow)
    for (width, height) in [(512, 512), (1024, 1024)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        // Rust Setup
        let mut data = vec![0.0f32; width * height];
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                data[i * width + i] = 1.0;
            }
        }
        let image = Image::new(
            ImageSize {
                width: *width,
                height: *height,
            },
            data,
            CpuAllocator,
        )
        .unwrap();

        // OpenCV Setup
        let mut cv_src = core::Mat::zeros(*height as i32, *width as i32, core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                *cv_src.at_2d_mut::<u8>(i as i32, i as i32).unwrap() = 255;
            }
        }
        let mut cv_dst = core::Mat::default();

        group.bench_with_input(
            BenchmarkId::new("kornia_linear", &parameter_string),
            &image,
            |b, i| b.iter(|| std::hint::black_box(distance_transform(i))),
        );

        group.bench_function(BenchmarkId::new("opencv", &parameter_string), |b| {
            b.iter(|| {
                imgproc::distance_transform(
                    &cv_src,
                    &mut cv_dst,
                    imgproc::DIST_L2,
                    imgproc::DIST_MASK_5,
                    core::CV_32F,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_distance_transform);
criterion_main!(benches);
