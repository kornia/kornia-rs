use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::distance_transform::{distance_transform, distance_transform_vanilla};
use kornia_tensor::CpuAllocator;
use opencv::{core, imgproc, prelude::*}; // Import OpenCV modules

fn bench_distance_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("DistanceTransform");

    for (width, height) in [(64, 64), (128, 128)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        // Rust Image
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

        // -OpenCV Mat
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
            BenchmarkId::new("new_linear_on", &parameter_string),
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
            BenchmarkId::new("new_linear_on", &parameter_string),
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
